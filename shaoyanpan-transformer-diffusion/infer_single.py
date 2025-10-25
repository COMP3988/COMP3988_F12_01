"""
MRI→CT inference script for the transformer-diffusion model.
- Loads a trained checkpoint
- Reads a single MRI volume from .npz or .mha/.mhd file
- Runs sliding-window diffusion sampling
- Writes CT prediction to .mha file, copying geometry from input or optional reference

Usage examples:
  python infer_single.py \
    --ckpt synthRAD_checkpoints/Synth_A_to_B.pth \
    --input SynthRAD/imagesTs/patient001.npz \
    --output patient001_ct_pred.mha \
    --steps 20 --overlap 0.25 --sw-batch 16

Notes:
- Input: Single .npz file (key: "image") or .mha/.mhd file
- Output: Single .mha file with CT prediction
- Uses sliding-window inference for large volumes
- Input volumes should be preprocessed and normalized to [-1,1] range
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from monai.inferers import SlidingWindowInferer
import SimpleITK as sitk

# Import configuration
from config import *
from preprocess_utils import load_and_preprocess_mha, postprocess_ct_output

# Model + diffusion imports from this repo
from diffusion.Create_diffusion import create_gaussian_diffusion
from diffusion.resampler import UniformSampler  # not used directly but kept for parity
from network.Diffusion_model_transformer import SwinVITModel


def parse_args():
    p = argparse.ArgumentParser(description="MRI→CT inference to .mha")

    # required args
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--input", required=True, help="Path to a single .npz or .mha/.mhd file")
    p.add_argument("--output", required=True, help="Output .mha file path")

    # recommended defaults / flags
    p.add_argument("--steps", type=int, default=DEFAULT_INFERENCE_STEPS, help="Diffusion steps at inference (timestep_respacing)")
    p.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP, help="Sliding-window overlap [0,1)")
    p.add_argument("--sw-batch", type=int, default=DEFAULT_SW_BATCH, help="Sliding-window batch size")
    p.add_argument("--ref-mha", type=str, default=None, help="Optional reference .mha to copy geometry when .npz lacks metadata")
    p.add_argument("--fp16", action="store_true", help="Enable autocast float16")
    p.add_argument("--device", default="cuda:0", help="Device, e.g., cuda:0 or cpu")
    p.add_argument("--sanity", action="store_true", help="Use minimal settings for quick testing (steps=2, overlap=0, sw-batch=1)")

    args = p.parse_args()

    # Apply sanity mode overrides
    if args.sanity:
        args.steps = 2
        args.overlap = 0.0
        args.sw_batch = 1
        print("Sanity mode enabled: Using minimal computational settings for quick testing")
        print(f"  - Steps: {args.steps}")
        print(f"  - Overlap: {args.overlap}")
        print(f"  - Sliding window batch size: {args.sw_batch}")

    return args

# ---------------- config (match training) ----------------
device = get_device()


def build_model(device: torch.device) -> torch.nn.Module:
    model = SwinVITModel(**get_model_config()).to(device)
    return model


def build_diffusion(steps: int):
    config = get_diffusion_config()
    config['timestep_respacing'] = [steps]
    return create_gaussian_diffusion(**config)


def diffusion_sampling_with(diffusion_obj, condition, model):
    # condition: (B,1,D,H,W) in [-1,1]
    return diffusion_obj.p_sample_loop(
        model,
        (condition.shape[0], 1, condition.shape[2], condition.shape[3], condition.shape[4]),
        condition=condition,
        clip_denoised=True,
    )




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


# Load .mha or .mhd volume and metadata
def load_mha_volume(mha_path: Path):
    img = sitk.ReadImage(str(mha_path))
    vol = sitk.GetArrayFromImage(img).astype(np.float32)  # (D,H,W)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
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
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")

    # Build model
    model = build_model(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Diffusion + inferer
    eval_diffusion = build_diffusion(args.steps)
    inferer = SlidingWindowInferer(
        roi_size=PATCH_SIZE,
        sw_batch_size=args.sw_batch,
        overlap=args.overlap,
        mode="constant",
    )

    # Single file input
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} not found")
    suffix = input_path.suffix.lower()

    if suffix == ".npz":
        # NPZ files are already preprocessed
        vol_np, spacing, origin, direction = load_npz_volume(input_path)
    elif suffix in [".mha", ".mhd"]:
        # MHA files need preprocessing - use standardized preprocessing
        vol_np, spacing, origin, direction = load_mha_volume(input_path)
        # Apply MRI preprocessing to match training
        from preprocess_utils import preprocess_mri
        vol_np = preprocess_mri(vol_np)
    else:
        raise ValueError(f"Unsupported input format: {suffix}. Expected .npz or .mha/.mhd")

    vol_t = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0).to(device)

    autocast_dtype = torch.float16 if args.fp16 and device.type == "cuda" else None

    with torch.no_grad():
        if autocast_dtype is not None:
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                pred = inferer(vol_t, lambda c, m: diffusion_sampling_with(eval_diffusion, c, m), model)
        else:
            pred = inferer(vol_t, lambda c, m: diffusion_sampling_with(eval_diffusion, c, m), model)

    ct_np = pred.squeeze(0).squeeze(0).detach().cpu().numpy()

    # Postprocess CT output from [-1, 1] back to HU units
    ct_np = postprocess_ct_output(ct_np)

    out_path = Path(args.output)
    ref_mha = Path(args.ref_mha) if args.ref_mha else None
    save_mha(ct_np, out_path, spacing=spacing, origin=origin, direction=direction, ref_mha=ref_mha)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()