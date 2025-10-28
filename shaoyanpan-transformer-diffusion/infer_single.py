"""
MRI→CT inference (single file, with optional mask).
If --mask infer is used, automatically loads 'mask.mha' from the same folder as input.
"""

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message="torch.meshgrid")
warnings.filterwarnings("ignore", message="Using a non-tuple sequence for multidimensional indexing is deprecated")

import argparse
from pathlib import Path
import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
import SimpleITK as sitk

# Model + diffusion imports
from diffusion.Create_diffusion import create_gaussian_diffusion
from diffusion.resampler import UniformSampler
from network.Diffusion_model_transformer import SwinVITModel


def parse_args():
    p = argparse.ArgumentParser(description="MRI→CT inference for a single file to .mha")

    p.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--input", required=True, help="Path to a .npz or .mha/.mhd file")
    p.add_argument("--output", required=True, help="Output .mha filepath")

    p.add_argument("--mask", type=str, default=None,
                   help="Optional .mha mask path, or 'infer' to auto-use mask.mha in input directory")
    p.add_argument("--steps", type=int, default=2, help="Diffusion steps at inference (timestep_respacing)")
    p.add_argument("--overlap", type=float, default=0.5, help="Sliding-window overlap [0,1)")
    p.add_argument("--sw-batch", type=int, default=512, help="Sliding-window batch size")
    p.add_argument("--ref-mha", type=str, default=None, help="Optional reference .mha to copy geometry")
    p.add_argument("--fp16", action="store_true", help="Enable autocast float16 (CUDA)")
    p.add_argument("--device", default="cuda:0", help="Device, e.g., cuda:0 or cpu")
    return p.parse_args()


# ---------------- config (match training) ----------------
num_channels = 64
attention_resolutions = "32,16,8"
channel_mult = (1, 2, 3, 4)
num_heads = [4, 4, 8, 16]
window_size = [[4, 4, 2]] * 4
num_res_blocks = [1] * 4
sample_kernel = ([2, 2, 2], [2, 2, 1], [2, 2, 1], [2, 2, 1]),
attention_ds = [int(x) for x in attention_resolutions.split(",")]

img_size = (256, 256, 128)
patch_size = (64, 64, 2)
patch_num = 1

# Diffusion hyperparams
DIFFUSION_STEPS = 1000
LEARN_SIGMA = True
SIGMA_SMALL = False
NOISE_SCHEDULE = "linear"
USE_KL = False
PREDICT_XSTART = True
RESCALE_TIMESTEPS = True
RESCALE_LEARNED_SIGMAS = True


def build_model(device):
    return SwinVITModel(
        image_size=patch_size,
        in_channels=2,
        model_channels=num_channels,
        out_channels=2,
        dims=3,
        sample_kernel=sample_kernel,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=0,
        channel_mult=channel_mult,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=num_heads,
        window_size=window_size,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
    ).to(device)


def build_diffusion(steps):
    return create_gaussian_diffusion(
        steps=DIFFUSION_STEPS,
        learn_sigma=LEARN_SIGMA,
        sigma_small=SIGMA_SMALL,
        noise_schedule=NOISE_SCHEDULE,
        use_kl=USE_KL,
        predict_xstart=PREDICT_XSTART,
        rescale_timesteps=RESCALE_TIMESTEPS,
        rescale_learned_sigmas=RESCALE_LEARNED_SIGMAS,
        timestep_respacing=str(steps),
    )


def diffusion_sampling_with(diffusion_obj, condition, model):
    return diffusion_obj.p_sample_loop(
        model,
        (condition.shape[0], 1, condition.shape[2], condition.shape[3], condition.shape[4]),
        condition=condition,
        clip_denoised=True,
    )


def load_input(path: Path):
    ext = path.suffix.lower()
    if ext == ".npz":
        d = np.load(str(path))
        vol = d["image"].astype(np.float32)
        spacing = tuple(d.get("spacing", ())) or None
        origin = tuple(d.get("origin", ())) or None
        direction = tuple(d.get("direction", ())) or None
        return vol, spacing, origin, direction
    elif ext in [".mha", ".mhd"]:
        img = sitk.ReadImage(str(path))
        vol = sitk.GetArrayFromImage(img).astype(np.float32)
        return vol, img.GetSpacing(), img.GetOrigin(), img.GetDirection()
    else:
        raise ValueError(f"Unsupported input type: {ext}")


def save_mha(volume_dhw, out_path, spacing=None, origin=None, direction=None, ref_mha=None):
    img = sitk.GetImageFromArray(volume_dhw.astype(np.float32))
    if ref_mha:
        ref = sitk.ReadImage(str(ref_mha))
        img.CopyInformation(ref)
    else:
        if spacing: img.SetSpacing(spacing)
        if origin: img.SetOrigin(origin)
        if direction: img.SetDirection(direction)
    sitk.WriteImage(img, str(out_path))


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    cuda_available = torch.cuda.is_available()

    # --- Parameter summary ---
    print("\n===== Inference Configuration =====")
    print(f"Device:              {device}")
    print(f"CUDA available:      {cuda_available}")
    print(f"Checkpoint:          {args.ckpt}")
    print(f"Input file:          {args.input}")
    print(f"Output file:         {args.output}")
    print(f"Mask:                {args.mask}")
    print(f"Steps:               {args.steps}")
    print(f"Sliding-window batch:{args.sw_batch}")
    print(f"Overlap:             {args.overlap}")
    print(f"Reference .mha:      {args.ref_mha}")
    print(f"FP16 enabled:        {args.fp16}")
    print("===================================\n")

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detect mask if --mask infer
    mask_path = None
    if args.mask:
        if args.mask.lower() == "infer":
            candidate = in_path.parent / "mask.mha"
            if candidate.exists():
                mask_path = candidate
                print(f"Auto-detected mask: {mask_path}")
            else:
                print(f"Warning: --mask infer set but no mask.mha found in {in_path.parent}")
        else:
            mask_path = Path(args.mask)

    # Build model
    model = build_model(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # Diffusion + inferer
    eval_diffusion = build_diffusion(args.steps)
    inferer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=args.sw_batch,
        overlap=args.overlap,
        mode="gaussian",
        sigma_scale=0.125,
        padding_mode="constant",
        cval=-1,
    )

    # Load input
    vol_np, spacing, origin, direction = load_input(in_path)

    # Load mask if available
    mask_np = None
    if mask_path:
        mask_img = sitk.ReadImage(str(mask_path))
        mask_np = sitk.GetArrayFromImage(mask_img).astype(np.float32)
        if mask_np.shape != vol_np.shape:
            raise ValueError(f"Mask shape {mask_np.shape} does not match input shape {vol_np.shape}")
        mask_np = (mask_np > 0.5).astype(np.float32)

    # Prepare tensor
    vol_hwd = np.moveaxis(vol_np, 0, -1)
    vol_t = torch.from_numpy(vol_hwd).unsqueeze(0).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        if args.fp16 and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                pred = inferer(vol_t, lambda c, m: diffusion_sampling_with(eval_diffusion, c, m), model)
        else:
            pred = inferer(vol_t, lambda c, m: diffusion_sampling_with(eval_diffusion, c, m), model)

    # Convert and mask
    ct_hwd = pred.squeeze(0).squeeze(0).cpu().numpy()
    ct_dhw = np.moveaxis(ct_hwd, -1, 0).astype(np.float32)

    if mask_np is not None:
        ct_dhw = ct_dhw * mask_np + (-1.0) * (1.0 - mask_np)

    # Map [-1,1] → [-1000,1000] HU
    ct_proc = np.clip(ct_dhw, -1, 1)
    ct_proc = (ct_proc + 1) / 2 * 2000 - 1000
    ct_proc = ct_proc * 0.5 - 200   # halves contrast, darker base

    save_mha(ct_proc, out_path, spacing, origin, direction, args.ref_mha)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()