#!/usr/bin/env python3
import os
import glob
import shutil
import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def prepare_sample_set(task_root, out_dir, n=10):
    """Collect 10 MR/CT pairs from AB/HN/TH in reverse alphabetical order and copy to out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    for region in ["AB", "HN", "TH"]:
        region_dir = os.path.join(task_root, region)
        if not os.path.isdir(region_dir):
            continue

        # get patient dirs in reverse alphabetical order
        patients = sorted(os.listdir(region_dir), reverse=True)
        selected = patients[:n]

        for patient in selected:
            pdir = os.path.join(region_dir, patient)
            mr_path = os.path.join(pdir, "mr.mha")
            ct_path = os.path.join(pdir, "ct.mha")
            if os.path.exists(mr_path) and os.path.exists(ct_path):
                # copy into out_dir/images with paired names
                fake_name = f"{region}_{patient}_fake_B.mha"
                real_name = f"{region}_{patient}_real_A.mha"
                shutil.copy(mr_path, os.path.join(out_dir, fake_name))  # treat MRI as "generated"
                shutil.copy(ct_path, os.path.join(out_dir, real_name))  # treat CT as "real"

def calculate_volume_metrics(results_dir):
    """Calculate SSIM, PSNR, MAE between .mha fake_B and real_A volumes slice by slice."""
    ssim_scores, psnr_scores, mae_scores = [], [], []

    fake_vols = sorted(glob.glob(os.path.join(results_dir, "*_fake_B.mha")))
    real_vols = sorted(glob.glob(os.path.join(results_dir, "*_real_A.mha")))

    for fake_path, real_path in zip(fake_vols, real_vols):
        fake_img = sitk.ReadImage(fake_path)
        real_img = sitk.ReadImage(real_path)
        fake_vol = sitk.GetArrayFromImage(fake_img).astype(np.float32)
        real_vol = sitk.GetArrayFromImage(real_img).astype(np.float32)

        # normalize to 0–1
        fake_vol = (fake_vol - fake_vol.min()) / (fake_vol.max() - fake_vol.min() + 1e-8)
        real_vol = (real_vol - real_vol.min()) / (real_vol.max() - real_vol.min() + 1e-8)

        # per-slice metrics
        for f_slice, r_slice in zip(fake_vol, real_vol):
            ssim_scores.append(ssim(r_slice, f_slice, data_range=1.0))
            psnr_scores.append(psnr(r_slice, f_slice, data_range=1.0))
            mae_scores.append(np.mean(np.abs(r_slice - f_slice)))

    return ssim_scores, psnr_scores, mae_scores

def print_summary(ssim_scores, psnr_scores, mae_scores):
    print("="*60)
    print("MRI→CT SYNTHESIS RESULTS SUMMARY")
    print("="*60)
    print(f"Number of slices: {len(ssim_scores)}")
    print(f"SSIM: {np.mean(ssim_scores):.3f} ± {np.std(ssim_scores):.3f}")
    print(f"PSNR: {np.mean(psnr_scores):.2f} ± {np.std(psnr_scores):.2f} dB")
    print(f"MAE : {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")

def main():
    task_root = "./synthRAD2025_Task1_Train/Task1"  # ADJUST AS NEEDED
    results_dir = "./demo_results/images"           # ADJUST AS NEEDED
    prepare_sample_set(task_root, results_dir, n=10)
    print(f"Prepared sample set under {results_dir}")

    ssim_scores, psnr_scores, mae_scores = calculate_volume_metrics(results_dir)
    print_summary(ssim_scores, psnr_scores, mae_scores)

if __name__ == "__main__":
    main()

