"""
Preprocess SynthRAD dataset into NPZ files
compatible with Shaoyanpan's transformer-diffusion model
"""

import random, os, glob, argparse
import SimpleITK as sitk
import numpy as np
from collections import defaultdict

def preprocess_synthrad(synthrad_location, out_dir, seed, proportion):
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
    def handle_one(tag, lst):
        subdir = os.path.join(out_dir, f"images{tag}")
        os.makedirs(subdir, exist_ok=True)
        nonlocal pat_i
        for patient_path in lst:
            patient_code = os.path.basename(patient_path)
            print(f"[{pat_i}/{len(mri_files)}] Processing {patient_code}"); pat_i += 1

            mr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(patient_path, "mr.mha"))).astype(np.float32)
            ct = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(patient_path, "ct.mha"))).astype(np.float32)

            # MRI robust scale → [-1,1]
            p_lo, p_hi = np.percentile(mr, [0.5, 99.5])
            if p_hi <= p_lo:
                mr = np.zeros_like(mr, dtype=np.float32)
            else:
                mr = np.clip(mr, p_lo, p_hi)
                mr = 2.0 * (mr - p_lo) / (p_hi - p_lo) - 1.0

            # CT HU clip [-1000,3000] → [-1,1]
            ct = np.clip(ct, -1000.0, 3000.0)
            ct = (ct + 1000.0) / 4000.0 * 2.0 - 1.0

            out_path = os.path.join(subdir, f"{patient_code}.npz")
            np.savez_compressed(out_path, image=mr.astype(np.float32), label=ct.astype(np.float32))

    handle_one("Tr", train); handle_one("Val", val); handle_one("Ts", test)
    print(f"Finished preprocessing. Saved NPZ pairs to {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--synthrad_location", required=True, help="Path to SynthRAD root")
    p.add_argument("--out_dir", default="SynthRAD", help="Output directory for NPZ files")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--proportion", type=float, default=1.0)
    args = p.parse_args()
    preprocess_synthrad(args.synthrad_location, args.out_dir, args.seed, args.proportion)
