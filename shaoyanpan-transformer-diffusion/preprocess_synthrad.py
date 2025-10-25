"""
Preprocess SynthRAD dataset into NPZ files
compatible with Shaoyanpan's transformer-diffusion model
"""

import random, os, glob, argparse
import SimpleITK as sitk
import numpy as np
from collections import defaultdict
from preprocess_utils import preprocess_mri, preprocess_ct

def preprocess_synthrad(synthrad_location, out_dir, seed, proportion, reverse_order=True):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)

    mri_files = sorted(glob.glob(os.path.join(synthrad_location, "**", "**", "mr.mha")))
    ct_files  = sorted(glob.glob(os.path.join(synthrad_location, "**", "**", "ct.mha")))

    if len(mri_files) != len(ct_files):
        raise RuntimeError(f"#MRI ({len(mri_files)}) != #CT ({len(ct_files)})")
    if not mri_files:
        raise RuntimeError("No MRI or CT files found")

    by_section = defaultdict(list)
    for mri_path in mri_files:
        patient_path = os.path.dirname(mri_path)
        section = os.path.basename(os.path.dirname(patient_path))
        by_section[section].append(patient_path)

    def split(lst, p):
        lst = sorted(set(lst))
        if p < 1: lst = lst[:int(round(len(lst) * p))]
        random.shuffle(lst); n = len(lst)
        return lst[:int(0.70*n)], lst[int(0.70*n):int(0.85*n)], lst[int(0.85*n):]

    train, val, test = [], [], []
    for _, patients in by_section.items():
        a,b,c = split(patients, proportion)
        train += a; val += b; test += c

    pat_i = 1
    processed_count = 0
    skipped_count = 0

    def handle_one(tag, lst):
        subdir = os.path.join(out_dir, f"images{tag}")
        os.makedirs(subdir, exist_ok=True)
        nonlocal pat_i, processed_count, skipped_count

        # Reverse order for parallel processing (if enabled)
        if reverse_order:
            lst = list(reversed(lst))

        for patient_path in lst:
            patient_code = os.path.basename(patient_path)
            out_path = os.path.join(subdir, f"{patient_code}.npz")

            # Skip if file already exists
            if os.path.exists(out_path):
                print(f"[{pat_i}/{len(mri_files)}] Skipping {patient_code} (already exists)"); pat_i += 1
                skipped_count += 1
                continue

            print(f"[{pat_i}/{len(mri_files)}] Processing {patient_code}"); pat_i += 1
            processed_count += 1

            mr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(patient_path, "mr.mha"))).astype(np.float32)
            ct = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(patient_path, "ct.mha"))).astype(np.float32)
            mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(patient_path, "mask.mha"))).astype(np.float32)

            # Use standardized preprocessing functions
            mr = preprocess_mri(mr)
            ct = preprocess_ct(ct)
            # Mask doesn't need preprocessing - keep as binary 0/1

            np.savez_compressed(out_path,
                              image=mr.astype(np.float32),
                              label=ct.astype(np.float32),
                              mask=mask.astype(np.float32))

    handle_one("Tr", train); handle_one("Val", val); handle_one("Ts", test)
    print(f"Finished preprocessing. Saved NPZ pairs to {out_dir}")
    print(f"Summary: {processed_count} files processed, {skipped_count} files skipped (already existed)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--synthrad_location", required=True, help="Path to SynthRAD root")
    p.add_argument("--out_dir", default="SynthRAD", help="Output directory for NPZ files")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--proportion", type=float, default=1.0)
    p.add_argument("--reverse_order", action="store_true", default=True, help="Process files in reverse order for parallelization")
    p.add_argument("--no_reverse_order", action="store_false", dest="reverse_order", help="Process files in normal order")
    args = p.parse_args()
    preprocess_synthrad(args.synthrad_location, args.out_dir, args.seed, args.proportion, args.reverse_order)
