# evaluate.py
"""
Performs quantitative evaluation of the trained diffusion model.

This script loads a trained model and calculates performance metrics (SSIM, PSNR, MAE)
by comparing generated CTs against ground truth CTs from a validation set.
The results are averaged over multiple samples for a reliable score.
"""

import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Local module imports
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from dataset import SynthRADDataset
from utils import generate_conditional_ct

# Set a fixed seed for reproducible evaluation metrics.
torch.manual_seed(42)

def calculate_metrics(generated, ground_truth):
    """Calculates SSIM, PSNR, and MAE for a single pair of images."""
    ssim_score = ssim(ground_truth, generated, data_range=1.0)
    psnr_score = psnr(ground_truth, generated, data_range=1.0)
    mae_score = np.mean(np.abs(ground_truth - generated))
    return {'ssim': ssim_score, 'psnr': psnr_score, 'mae': mae_score}


# --- SET UP ---

# Initialise model and diffusion framework.
model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=2)
diffusion = GaussianDiffusion(model, image_size=128, timesteps=1000)

# Load the trained model weights.
trained_model_path = 'diffusion_baseline_20_epochs.pt'
diffusion.load_state_dict(torch.load(trained_model_path))

# Load the dataset.
dataset = SynthRADDataset(root_dir='data/synthrad_train', image_size=128)

# --- EVALUATION LOOP ---

num_eval_samples = 3
all_metrics = {'ssim': [], 'psnr': [], 'mae': []}

print(f"Starting evaluation on {num_eval_samples} samples...")
for i in range(num_eval_samples):
    mri_tensor, ground_truth_ct = dataset[i]
    mri_tensor = mri_tensor.unsqueeze(0)

    # Generate the CT from the MRI.
    generated_ct_tensor = generate_conditional_ct(diffusion, mri_tensor)

    # Convert tensors to NumPy arrays for metric calculation.
    # Rescale from [-1, 1] (model output) to [0, 1] (metric input).
    generated_np = (generated_ct_tensor.squeeze().numpy() + 1) / 2
    ground_truth_np = (ground_truth_ct.squeeze().numpy() + 1) / 2

    # Calculate and store metrics for this sample.
    metrics = calculate_metrics(generated_np, ground_truth_np)
    for key in all_metrics:
        all_metrics[key].append(metrics[key])

# --- REPORT RESULTS ---

print("\n--- AVERAGE EVALUATION METRICS ---")
print(f"SSIM: {np.mean(all_metrics['ssim']):.4f} (+/- {np.std(all_metrics['ssim']):.4f})")
print(f"PSNR: {np.mean(all_metrics['psnr']):.2f} dB (+/- {np.std(all_metrics['psnr']):.2f})")
print(f"MAE:  {np.mean(all_metrics['mae']):.4f} (+/- {np.std(all_metrics['mae']):.4f})")
print("------------------------------------\n")