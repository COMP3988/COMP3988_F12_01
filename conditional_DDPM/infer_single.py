"""
Conditional DDPM inference script for MRI→CT synthesis.
- Loads a trained checkpoint
- Reads MRI volumes from .npz or .mha files
- Runs conditional diffusion sampling
- Writes CT predictions to .mha

Usage examples:
  python infer_single.py \
    --ckpt Checkpoints/trial_1/ckpt_1000_.pt \
    --input data/sample.npz \
    --output output_ct.mha

  python infer_single.py \
    --ckpt Checkpoints/trial_1/ckpt_1000_.pt \
    --input data/sample.mha \
    --output output_ct.mha
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import SimpleITK as sitk

# Model imports from this repo
from Model_condition import UNet
from Diffusion_condition import GaussianDiffusionSampler_cond


def parse_args():
    p = argparse.ArgumentParser(description="Conditional DDPM MRI→CT inference to .mha")

    # required args
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pt)")
    p.add_argument("--input", required=True, help="Path to a single .npz or .mha/.mhd file")
    p.add_argument("--output", required=True, help="Output path for .mha file")

    # model hyperparameters (should match training)
    p.add_argument("--T", type=int, default=1000, help="Number of diffusion timesteps")
    p.add_argument("--ch", type=int, default=64, help="Base number of channels")
    p.add_argument("--ch-mult", type=int, nargs="+", default=[1, 2, 3], help="Channel multipliers")
    p.add_argument("--attn", type=int, nargs="+", default=[1], help="Attention layer indices")
    p.add_argument("--num-res-blocks", type=int, default=1, help="Number of residual blocks")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    p.add_argument("--beta-1", type=float, default=1e-4, help="Diffusion schedule start")
    p.add_argument("--beta-T", type=float, default=0.02, help="Diffusion schedule end")

    # inference options
    p.add_argument("--fp16", action="store_true", help="Enable autocast float16")
    p.add_argument("--device", default="cuda:0", help="Device, e.g., cuda:0 or cpu")
    p.add_argument("--timesteps", type=int, default=None, help="Number of sampling timesteps (default: full T=1000)")
    return p.parse_args()


def load_npz_volume(npz_path: Path):
    """Load volume from .npz file."""
    d = np.load(str(npz_path))
    if "image" not in d:
        raise KeyError(f"{npz_path} missing 'image' array")
    vol = d["image"].astype(np.float32)
    # Handle both 2D and 3D inputs
    if vol.ndim == 2:
        vol = vol[None, :, :]  # Add channel dimension: (1, H, W)
    elif vol.ndim == 3:
        # Assume (D, H, W) - take middle slice for 2D inference
        D = vol.shape[0]
        vol = vol[D//2, :, :][None, :, :]
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {vol.shape}")
    # Optional geometry
    spacing = tuple(d["spacing"]) if "spacing" in d else (1.0, 1.0)
    origin = tuple(d["origin"]) if "origin" in d else (0.0, 0.0)
    direction = tuple(d["direction"]) if "direction" in d else None
    return vol, spacing, origin, direction


def load_mha_volume(mha_path: Path):
    """Load volume from .mha/.mhd file."""
    img = sitk.ReadImage(str(mha_path))
    vol = sitk.GetArrayFromImage(img).astype(np.float32)
    # Take middle slice for 2D inference
    if vol.ndim == 3:
        D = vol.shape[0]
        vol = vol[D//2, :, :]
    vol = vol[None, :, :]  # Add channel dimension
    spacing = img.GetSpacing()[:2]  # (x, y) spacing
    origin = img.GetOrigin()[:2]    # (x, y) origin
    direction = img.GetDirection()
    return vol, spacing, origin, direction


def save_mha(volume_np: np.ndarray, out_path: Path, spacing=None, origin=None, direction=None):
    """Save volume to .mha file."""
    # Remove channel dimension if present
    if volume_np.ndim == 3 and volume_np.shape[0] == 1:
        volume_np = volume_np[0]

    # Add dummy depth dimension for SimpleITK
    if volume_np.ndim == 2:
        volume_np = volume_np[None, :, :]

    img = sitk.GetImageFromArray(volume_np.astype(np.float32))
    if spacing is not None:
        spacing_3d = (spacing[0], spacing[1], 1.0) if len(spacing) == 2 else spacing
        img.SetSpacing(spacing_3d)
    if origin is not None:
        origin_3d = (origin[0], origin[1], 0.0) if len(origin) == 2 else origin
        img.SetOrigin(origin_3d)
    if direction is not None:
        img.SetDirection(direction)
    sitk.WriteImage(img, str(out_path))


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")

    print(f"Loading model from {args.ckpt}")
    # Build model
    model = UNet(
        T=args.T,
        ch=args.ch,
        ch_mult=args.ch_mult,
        attn=args.attn,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout
    ).to(device)

    # Load checkpoint - handle both raw state_dict and training checkpoints
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        # Training checkpoint with full state
        state_dict = ckpt['model_state_dict']
        print(f"Loaded training checkpoint (epoch {ckpt.get('epoch', 'N/A')})")
    elif 'state_dict' in ckpt:
        # Another checkpoint format
        state_dict = ckpt['state_dict']
    else:
        # Raw state_dict
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Build sampler (always uses full T=1000 diffusion schedule)
    sampler = GaussianDiffusionSampler_cond(
        model, args.beta_1, args.beta_T, args.T
    ).to(device)

    # Optionally reduce inference timesteps
    if args.timesteps and args.timesteps < args.T:
        # Calculate evenly spaced timesteps for acceleration
        step_size = args.T // args.timesteps
        time_steps = list(range(args.T-1, -1, -step_size))
        num_steps = len(time_steps)

        # Monkey patch the forward to use fewer timesteps
        original_forward = sampler.forward
        def reduced_forward(x_T):
            x_t = x_T
            ct = x_t[:,0,:,:]
            cbct = x_t[:,1,:,:]
            ct = torch.unsqueeze(ct,1)
            cbct = torch.unsqueeze(cbct,1)
            for time_step in time_steps:
                t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
                mean, var = sampler.p_mean_variance(x_t=x_t, t=t)
                if time_step > 0:
                    noise = torch.randn_like(ct)
                else:
                    noise = 0
                ct = mean + torch.sqrt(var) * noise
                x_t = torch.cat((ct,cbct),1)
                assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
            return torch.clip(x_t, -1, 1)
        sampler.forward = reduced_forward
        print(f"Using reduced timesteps: {num_steps} steps out of {args.T}")

    print(f"Model loaded successfully")

    # Load input volume
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} not found")

    suffix = input_path.suffix.lower()
    if suffix == ".npz":
        vol_np, spacing, origin, direction = load_npz_volume(input_path)
    elif suffix in [".mha", ".mhd"]:
        vol_np, spacing, origin, direction = load_mha_volume(input_path)
    else:
        raise ValueError(f"Unsupported input format: {suffix}. Expected .npz or .mha/.mhd")

    print(f"Input shape: {vol_np.shape}, spacing: {spacing}, origin: {origin}")

    # Store original shape for later unpadding
    original_shape = vol_np.shape[1:]  # (H, W)
    H, W = original_shape

    # Normalize to match training data preprocessing: [0, 1] range
    # Training data clips to [-1000, 3000] HU and normalizes to [0, 1]
    print(f"Normalizing input to match training data format...")
    vol_np = np.clip(vol_np, -1000, 3000)  # Clip to typical HU range
    vol_np = (vol_np - (-1000)) / (3000 - (-1000))  # Normalize to [0, 1]
    vol_np = np.clip(vol_np, 0, 1)  # Ensure [0, 1] range

    # Keep in [0, 1] range to match training data preprocessing

    # Model was trained on 128x128 images - resize to training size
    target_size = 128
    if H != target_size or W != target_size:
        print(f"Resizing from {H}x{W} to {target_size}x{target_size} to match training size")
        from scipy.ndimage import zoom
        zoom_h = target_size / H
        zoom_w = target_size / W
        vol_np = zoom(vol_np[0], (zoom_h, zoom_w), order=1)[None, :, :]
        H, W = vol_np.shape[1:]
        original_shape = (H, W)  # Update to match resized dimensions

    # Pad to multiple of 16 for UNet compatibility (downsample/upsample operations)
    # Calculate padded dimensions
    # Round up to nearest multiple of 16
    pad_h = ((H + 15) // 16) * 16
    pad_w = ((W + 15) // 16) * 16

    if pad_h != H or pad_w != W:
        print(f"Padding from {H}x{W} to {pad_h}x{pad_w} for model compatibility")
        vol_padded = np.zeros((1, pad_h, pad_w), dtype=vol_np.dtype)
        vol_padded[:, :H, :W] = vol_np
    else:
        vol_padded = vol_np

    # Convert to tensor and add batch dimension
    vol_t = torch.from_numpy(vol_padded).unsqueeze(0).to(device)  # (1, 1, H, W)

    print(f"Running inference...")
    autocast_dtype = torch.float16 if args.fp16 and device.type == "cuda" else None

    with torch.no_grad():
        if autocast_dtype is not None:
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                # Initialize noisy image and concatenate with condition
                noisy_img = torch.randn(size=(1, 1, vol_t.shape[2], vol_t.shape[3]), device=device)
                x_in = torch.cat((noisy_img, vol_t), 1)  # (B, 2, H, W) - [noise, condition]
                # Run sampling
                pred = sampler(x_in)
                # Extract predicted CT (first channel)
                ct_pred = pred[:, 0:1, :, :]
        else:
            noisy_img = torch.randn(size=(1, 1, vol_t.shape[2], vol_t.shape[3]), device=device)
            x_in = torch.cat((noisy_img, vol_t), 1)
            pred = sampler(x_in)
            ct_pred = pred[:, 0:1, :, :]

    # Convert back to numpy
    ct_np = ct_pred.squeeze(0).detach().cpu().numpy()  # (1, H, W)

    # Unpad if we padded
    if pad_h != H or pad_w != W:
        print(f"Unpadding from padded dimensions")
        ct_np = ct_np[:, :H, :W]

    # Denormalize from [-1, 1] (model output) to HU values
    # Model outputs in [-1, 1] due to final clip in Diffusion_condition.py line 103
    # Convert to [0, 1] then to HU: [-1, 1] -> [0, 1] -> HU
    print(f"Denormalizing output to HU values...")
    ct_np = (ct_np + 1) / 2  # Convert from [-1, 1] to [0, 1]
    ct_np = ct_np * 4000 - 1000  # Convert from [0, 1] to HU range [-1000, 3000]
    ct_np = np.clip(ct_np, -1000, 3000)  # Ensure valid HU range

    # Save output
    out_path = Path(args.output)
    save_mha(ct_np, out_path, spacing=spacing, origin=origin, direction=direction)
    print(f"Saved output to {out_path}")


if __name__ == "__main__":
    main()
