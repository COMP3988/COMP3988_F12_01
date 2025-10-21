"""
MRI→CT inference script for the transformer-diffusion model.
- Loads a trained checkpoint
- Reads MRI volumes from .npz files (key: "image")
- Runs sliding-window diffusion sampling
- Writes CT predictions to .mha, copying geometry from the .npz (keys: spacing/origin/direction) or an optional reference .mha

Usage examples:
  python infer.py \
    --ckpt synthRAD_checkpoints/Synth_A_to_B.pth \
    --input SynthRAD/imagesTs \
    --outdir outputs_mha \
    --steps 20 --overlap 0.25 --sw-batch 16

Notes:
- This matches the network and diffusion hyperparameters used in training code you provided.
- Input .npz files are expected to be preprocessed to fixed size and normalized to [-1,1] the same way as training.
"""

# TODO make work with .mhas, not npzs (or both)

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
import SimpleITK as sitk

# Model + diffusion imports from this repo
from diffusion.Create_diffusion import create_gaussian_diffusion
from network.Diffusion_model_transformer import SwinVITModel


def parse_args():
    p = argparse.ArgumentParser(description="MRI→CT inference to .mha")
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--input", required=True, help="Path to a .npz file or a directory of .npz files")
    p.add_argument("--outdir", required=True, help="Output directory for .mha")
    p.add_argument("--steps", type=int, default=20, help="Diffusion steps at inference (timestep_respacing)")
    p.add_argument("--overlap", type=float, default=0.25, help="Sliding-window overlap [0,1)")
    p.add_argument("--sw-batch", type=int, default=16, help="Sliding-window batch size")
    p.add_argument("--ref-mha", type=str, default=None, help="Optional reference .mha to copy geometry when .npz lacks metadata")
    p.add_argument("--fp16", action="store_true", help="Enable autocast float16")
    p.add_argument("--device", default="cuda:0", help="Device, e.g., cuda:0 or cpu")
    return p.parse_args()


# Hyperparameters copied from training script
num_channels=64
attention_resolutions="32,16,8"
channel_mult = (1, 2, 3, 4)
num_heads=[4,4,8,16]
window_size = [[4,4,2],[4,4,2],[4,4,2],[4,4,2]]
num_res_blocks = [1,1,1,1]
sample_kernel=([2,2,2],[2,2,1],[2,2,1],[2,2,1])  # keep trailing comma to match training
attention_ds = [int(x) for x in attention_resolutions.split(",")]

# ---------------- config (match training) ----------------
img_size   = (256,256,128)
patch_size = (64,64,2)
patch_num  = 1

# Diffusion hyperparams copied from training script
DIFFUSION_STEPS = 1000
LEARN_SIGMA = True
SIGMA_SMALL = False
NOISE_SCHEDULE = "linear"
USE_KL = False
PREDICT_XSTART = True
RESCALE_TIMESTEPS = True
RESCALE_LEARNED_SIGMAS = True

def pick_device(dev):
    if dev == "mps" and torch.backends.mps.is_available(): return torch.device("mps")
    if dev.startswith("cuda") and torch.cuda.is_available(): return torch.device(dev)
    return torch.device("cpu")


def build_model(device: torch.device) -> torch.nn.Module:
    attention_ds = [int(r) for r in attention_resolutions.split(",")]
    model = SwinVITModel(
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
        num_classes=None,
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
    return model


def build_diffusion(steps: int):
    return create_gaussian_diffusion(
        steps=DIFFUSION_STEPS,
        learn_sigma=LEARN_SIGMA,
        sigma_small=SIGMA_SMALL,
        noise_schedule=NOISE_SCHEDULE,
        use_kl=USE_KL,
        predict_xstart=PREDICT_XSTART,
        rescale_timesteps=RESCALE_TIMESTEPS,
        rescale_learned_sigmas=RESCALE_LEARNED_SIGMAS,
        timestep_respacing=[steps],
    )


def diffusion_sampling_with(diffusion_obj, condition, model):
    # condition: (B,1,D,H,W) in [-1,1]
    return diffusion_obj.p_sample_loop(
        model,
        (condition.shape[0], 1, condition.shape[2], condition.shape[3], condition.shape[4]),
        condition=condition,
        clip_denoised=True,
    )


def list_npz(path: Path):
    if path.is_file() and path.suffix.lower() == ".npz":
        return [path]
    files = sorted([p for p in path.glob("*.npz")])
    if not files:
        raise FileNotFoundError(f"No .npz files found at {path}")
    return files


def load_npz_volume(npz_path: Path):
    d = np.load(str(npz_path))
    if "image" not in d:
        raise KeyError(f"{npz_path} missing 'image' array")
    vol = d["image"].astype(np.float32)  # expected shape (D,H,W) or (H,W,D)
    # Optional geometry
    spacing = tuple(d["spacing"]) if "spacing" in d else None
    origin = tuple(d["origin"]) if "origin" in d else None
    direction = tuple(d["direction"]) if "direction" in d else None
    # Ensure (D,H,W)
    if vol.ndim != 3:
        raise ValueError(f"{npz_path} 'image' must be 3D, got shape {vol.shape}")
    return vol, spacing, origin, direction


def save_mha(volume_np: np.ndarray, out_path: Path, spacing=None, origin=None, direction=None, ref_mha: Path | None = None):
    img = sitk.GetImageFromArray(volume_np.astype(np.float32))
    if ref_mha is not None:
        ref = sitk.ReadImage(str(ref_mha))
        img.SetSpacing(ref.GetSpacing())
        img.SetOrigin(ref.GetOrigin())
        img.SetDirection(ref.GetDirection())
    else:
        if spacing is not None:
            img.SetSpacing(tuple(spacing))
        if origin is not None:
            img.SetOrigin(tuple(origin))
        if direction is not None:
            img.SetDirection(tuple(direction))
    sitk.WriteImage(img, str(out_path))


def main():
    args = parse_args()
    device = pick_device(args.device)
    os.makedirs(args.outdir, exist_ok=True)

    # Build model
    model = build_model(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Diffusion + inferer
    eval_diffusion = build_diffusion(args.steps)
    inferer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=args.sw_batch,
        overlap=args.overlap,
        mode="gaussian",
        sigma_scale=0.125,
        padding_mode="reflect"
    )

    # Inputs
    npz_files = list_npz(Path(args.input))

    autocast_dtype = torch.float16 if args.fp16 and device.type == "cuda" else None

    for npz_path in npz_files:
        vol_np, spacing, origin, direction = load_npz_volume(npz_path)

        vol_hwd = np.moveaxis(vol_np, 0, -1)          # (D,H,W) -> (H,W,D)
        vol_t = torch.from_numpy(vol_hwd).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W,D)

        with torch.no_grad():
            if autocast_dtype is not None:
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    pred = inferer(vol_t, lambda c, m: diffusion_sampling_with(eval_diffusion, c, m), model)
            else:
                pred = inferer(vol_t, lambda c, m: diffusion_sampling_with(eval_diffusion, c, m), model)

        # pred: (1,1,H,W,D) -> (H,W,D)
        ct_hwd = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
        # back to (D,H,W) for SimpleITK
        ct_dhw = np.moveaxis(ct_hwd, -1, 0).astype(np.float32)
        ct_dhw = np.ascontiguousarray(ct_dhw)

        print("pred HWD:", ct_hwd.shape, "saved DHW:", ct_dhw.shape)  # expect same dims, reordered
        Hx, Wx, Dz = ct_hwd.shape
        Dx, Hy, Wy = ct_dhw.shape
        assert (Dx,Hy,Wy) == (Dz,Hx,Wx), "axis swap wrong"

        # expected raw bytes if float32
        exp=Dx*Hy*Wy*4
        print("expected bytes:", exp)

        out_name = npz_path.stem + ".mha"
        out_path = Path(args.outdir) / out_name

        img = sitk.GetImageFromArray(np.ascontiguousarray(ct_dhw).astype(np.float32))
        if args.ref_mha:
            ref = sitk.ReadImage(str(args.ref_mha)); img.CopyInformation(ref)
        else:
            if spacing is not None:   img.SetSpacing(tuple(spacing))
            if origin  is not None:   img.SetOrigin(tuple(origin))
            if direction is not None: img.SetDirection(tuple(direction))

        w = sitk.ImageFileWriter()
        w.SetFileName(str(out_path))
        w.UseCompressionOff()
        w.Execute(img)

        # reopen with SimpleITK, not napari
        im = sitk.ReadImage(str(out_path))
        print("SITK size (x,y,z):", im.GetSize(), "pixel:", im.GetPixelIDTypeAsString())

if __name__ == "__main__":
    main()