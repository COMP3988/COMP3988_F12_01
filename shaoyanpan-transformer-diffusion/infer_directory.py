"""
Batch inference script for MRI→CT conversion using transformer-diffusion model.
Processes all .npz/.mha/.mhd files in a directory using infer_single.py.

Usage:
  python infer_directory.py \
    --ckpt synthRAD_checkpoints/Synth_A_to_B.pth \
    --inputdir SynthRAD/imagesTs \
    --outputdir outputs_mha \
    --steps 20 --overlap 0.25 --sw-batch 16
"""

import argparse
import os
import glob
import subprocess
from pathlib import Path
from natsort import natsorted

def parse_args():
    p = argparse.ArgumentParser(description="Batch MRI→CT inference for .mha/.npz volumes")
    p.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pth)")
    p.add_argument("--inputdir", required=True, help="Directory containing .npz or .mha/.mhd files")
    p.add_argument("--outputdir", required=True, help="Output directory for .mha results")
    p.add_argument("--steps", type=int, default=2, help="Diffusion steps at inference (timestep_respacing)")
    p.add_argument("--overlap", type=float, default=0.50, help="Sliding-window overlap [0,1)")
    p.add_argument("--sw-batch", type=int, default=8, help="Sliding-window batch size")
    p.add_argument("--ref-mha", type=str, default=None, help="Optional reference .mha for geometry copying")
    p.add_argument("--fp16", action="store_true", help="Enable autocast float16")
    p.add_argument("--device", default="cuda:0", help="Device, e.g., cuda:0 or cpu")
    return p.parse_args()


def main():
    args = parse_args()

    inputdir = Path(args.inputdir)
    outputdir = Path(args.outputdir)
    outputdir.mkdir(parents=True, exist_ok=True)

    # collect inputs
    inputs = natsorted(
        [str(p) for p in inputdir.glob("*.npz")] +
        [str(p) for p in inputdir.glob("*.mha")] +
        [str(p) for p in inputdir.glob("*.mhd")]
    )

    if not inputs:
        raise FileNotFoundError(f"No .npz/.mha/.mhd files found in {inputdir}")

    for infile in inputs:
        print(f"Processing {infile}...")
        # Generate output filename
        infile_path = Path(infile)
        outfile = outputdir / f"{infile_path.stem}_ct_pred.mha"

        cmd = [
            "python", "infer_single.py",
            "--ckpt", args.ckpt,
            "--input", infile,
            "--output", str(outfile),
            "--steps", str(args.steps),
            "--overlap", str(args.overlap),
            "--sw-batch", str(args.sw_batch),
            "--device", args.device,
        ]
        if args.ref_mha:
            cmd += ["--ref-mha", args.ref_mha]
        if args.fp16:
            cmd += ["--fp16"]

        subprocess.run(cmd, check=True)

    print(f"All volumes processed → outputs saved under {outputdir}")


if __name__ == "__main__":
    main()
