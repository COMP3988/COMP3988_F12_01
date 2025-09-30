"""
Preprocess SynthRAD dataset into MATLAB .mat files
compatible with Shaoyanpan's transformer-diffusion model
"""

import os
import glob
import argparse
import SimpleITK as sitk
import numpy as np
import scipy.io

def preprocess_synthrad(synthrad_location, out_dir):
    os.makedirs(out_dir, exist_ok=True)

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

    id = 0
    last_section = ""

    for i, (mri_path, ct_path) in enumerate(zip(mri_files, ct_files), start=1):

        patient_path = os.path.dirname(mri_path)
        section_path = os.path.dirname(patient_path)
        section = os.path.basename(section_path)

        if section != last_section:
            last_section = section
            id = 0

        case_name = f"{section}_{id}"
        id += 1

        print(f"[{i}/{len(mri_files)}] Processing {case_name}")

        # Load MRI + CT
        mri_img = sitk.ReadImage(mri_path)
        ct_img = sitk.ReadImage(ct_path)

        mri_arr = sitk.GetArrayFromImage(mri_img).astype(np.float32)
        ct_arr  = sitk.GetArrayFromImage(ct_img).astype(np.float32)

        # TODO: add normalization / resampling here if needed
        # e.g., mri_arr = (mri_arr - np.mean(mri_arr)) / np.std(mri_arr)

        # Save as .mat in out_dir
        out_path = os.path.join(out_dir, f"{case_name}.mat")

        scipy.io.savemat(out_path, {
            "image": mri_arr,
            "label": ct_arr
        })

    print(f"✅ Finished preprocessing. Saved {len(mri_files)} .mat files to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthrad_location", required=True,
                        help="Path to SynthRAD dataset root (with imagesTr/, labelsTr/)")
    parser.add_argument("--out_dir", default="SynthRAD",
                        help="Output directory for .mat files")
    args = parser.parse_args()

    preprocess_synthrad(args.synthrad_location, args.out_dir)
