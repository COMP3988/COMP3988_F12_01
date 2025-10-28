#!/usr/bin/env python3
"""
Match synthetic CT (sCT) intensity distribution to ground truth CT (gCT).
Usage:
  python match_sct_to_gt.py --gt path/to/ground_truth_ct.mha --sct path/to/synthetic_ct.mha --output matched_sct.mha
"""

import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from skimage.exposure import match_histograms
import napari


def read_mha(path):
    """Load .mha image as numpy array and metadata."""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr, img


def save_mha(array, ref_img, out_path):
    """Save numpy array as .mha, copying spatial metadata from reference image."""
    img = sitk.GetImageFromArray(array.astype(np.float32))
    img.CopyInformation(ref_img)
    sitk.WriteImage(img, str(out_path))
    print(f"Saved matched sCT to: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Histogram-match synthetic CT to ground truth CT")
    p.add_argument("--gt", required=True, help="Path to ground truth CT (.mha)")
    p.add_argument("--sct", required=True, help="Path to synthetic CT (.mha)")
    p.add_argument("--output", required=True, help="Output path for matched sCT (.mha)")
    p.add_argument("--window", type=float, nargs=2, default=(-200, 300),
                   help="Display window for Napari (min max HU)")
    args = p.parse_args()

    gt_path = Path(args.gt)
    sct_path = Path(args.sct)
    out_path = Path(args.output)

    print("\n===== Matching Configuration =====")
    print(f"Ground Truth CT: {gt_path}")
    print(f"Synthetic CT:    {sct_path}")
    print(f"Output file:     {out_path}")
    print(f"Window (HU):     {args.window}")
    print("=================================\n")

    # --- Load both volumes ---
    gt, gt_img = read_mha(gt_path)
    sct, sct_img = read_mha(sct_path)

    if gt.shape != sct.shape:
        raise ValueError(f"Shape mismatch: GT {gt.shape} vs SCT {sct.shape}")

    # --- Histogram matching ---
    print("Performing histogram matching (skimage.exposure.match_histograms)...")
    sct_matched = match_histograms(sct, gt, channel_axis=None)

    # --- Save result ---
    save_mha(sct_matched, sct_img, out_path)

    # --- Launch Napari viewer for visual comparison ---
    print("Launching Napari for visual inspection...")
    viewer = napari.Viewer()
    viewer.add_image(gt, name="Ground Truth CT", contrast_limits=args.window)
    viewer.add_image(sct, name="Original sCT", contrast_limits=args.window)
    viewer.add_image(sct_matched, name="Matched sCT", contrast_limits=args.window)
    napari.run()


if __name__ == "__main__":
    main()