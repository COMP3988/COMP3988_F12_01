"""
Preprocess SynthRAD dataset into MATLAB .mat files
compatible with Shaoyanpan's transformer-diffusion model
"""

import random
import os
import glob
import argparse
import SimpleITK as sitk
import numpy as np
import scipy.io

from collections import defaultdict


def preprocess_synthrad(synthrad_location, out_dir, seed, proportion):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)

    # TODO can implement more fine-grained control for sections later on
    # TODO masks
    # <dataset>/<section>/<patient>/...
    mri_files = sorted(glob.glob(os.path.join(synthrad_location, "**", "**", "mr.mha")))
    ct_files  = sorted(glob.glob(os.path.join(synthrad_location, "**", "**", "ct.mha")))

    if len(mri_files) != len(ct_files):
        raise RuntimeError(f"Number of MRI files ({len(mri_files)}) "
                           f"!= number of CT files ({len(ct_files)})")

    if len(mri_files) == 0:
        raise RuntimeError(f"No MRI or CT files found!")

    by_section = defaultdict(list)

    for mri_path in mri_files:
        patient_path = os.path.dirname(mri_path)
        section_path = os.path.dirname(patient_path)
        section = os.path.basename(section_path)

        by_section[section].append(patient_path)

    train = []
    val = []
    test = []

    for section, patients in by_section.items():
        patients = sorted(set(patients))

        if proportion < 1:
            patients = patients[:int(round(len(patients) * proportion))]

        random.shuffle(patients)
        n = len(patients)

        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        train.extend(patients[:n_train])
        val.extend(patients[n_train:n_train + n_val])
        test.extend(patients[n_train + n_val:])

    pat_i = 1

    def handle_one(type, lst):

        subdir = f'images{type}'
        os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)

        nonlocal pat_i

        for patient_path in lst:

            patient_code = os.path.basename(patient_path)
            mr_path = os.path.join(patient_path, "mr.mha")
            ct_path = os.path.join(patient_path, "ct.mha")

            print(f"[{pat_i}/{len(mri_files)}] Processing {patient_code}")
            pat_i += 1

            mr_img = sitk.ReadImage(mr_path)
            ct_img = sitk.ReadImage(ct_path)

            mri_arr = sitk.GetArrayFromImage(mr_img).astype(np.float32)
            ct_arr  = sitk.GetArrayFromImage(ct_img).astype(np.float32)

            # TODO: add normalization / resampling here if needed
            # e.g., mri_arr = (mri_arr - np.mean(mri_arr)) / np.std(mri_arr)

            out_path = os.path.join(out_dir, subdir, f"{patient_code}.mat")

            scipy.io.savemat(out_path, {
                "image": mri_arr,
                "label": ct_arr
            })

    handle_one('Tr', train)
    handle_one('Val', val)
    handle_one('Ts', test)

    print(f"✅ Finished preprocessing. Saved {len(mri_files)} .mat files to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthrad_location", required=True,
                        help="Path to SynthRAD dataset root (with imagesTr/, labelsTr/)")
    parser.add_argument("--out_dir", default="SynthRAD",
                        help="Output directory for .mat files")
    parser.add_argument("--seed", default="42", type=int,
                        help="Seed for how patients are randomly divided into training, validation, and testing sets")
    parser.add_argument("--proportion", default="1", type=float,
                        help="What proportion of total SynthRAD set to preprocess")
    args = parser.parse_args()

    preprocess_synthrad(args.synthrad_location, args.out_dir, args.seed, args.proportion)
