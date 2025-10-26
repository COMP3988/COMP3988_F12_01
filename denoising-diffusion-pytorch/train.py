"""
Training script for conditional diffusion on full dataset.
Includes:
Resume from checkpoints or old best model
Quick SSIM check after loading weights
Save every 500 batches
Full dataset training
Fast validation (50 steps) at epoch end
Early stopping
Logs everything to Google Drive
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split, Subset

from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from dataset import SynthRADDataset
from utils import generate_conditional_ct

# set-up logging
drive_project_dir = '/content/drive/MyDrive/denoising-diffusion-pytorch'
checkpoint_dir = os.path.join(drive_project_dir, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest.pt')
best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
log_file_path = os.path.join(drive_project_dir, 'train_log.txt')

sys.stdout = open(log_file_path, "a", buffering=1)
sys.stderr = sys.stdout
print("\n--- NEW TRAINING SESSION STARTED ---")

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# model set-up
model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=2)
diffusion = GaussianDiffusion(model, image_size=128, timesteps=1000)
diffusion.to(device)
print("Model and diffusion pipeline moved to:", device)

# data loading
full_dataset = SynthRADDataset(root_dir='synthRAD2025_Task1_Train/Task1', image_size=128)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

print(f"Total dataset size: {len(full_dataset)}")
print(f"Training on {len(train_data)}, validating on {len(val_data)}")

batch_size = 4
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# training configs
optimizer = Adam(diffusion.parameters(), lr=1e-5)
epochs = 50
best_ssim = 0.0
best_epoch = 0
patience = 5
patience_counter = 0
start_epoch = 0

# validation steps
num_val_samples = min(50, len(val_data))
val_subset = Subset(val_data, list(range(num_val_samples)))
val_loader = DataLoader(val_subset, batch_size=1)

def validate_model(model, loader, device):
    ssim_scores, psnr_scores, mae_scores = [], [], []
    model.eval()
    with torch.no_grad():
        for val_mri, val_gt_ct in tqdm(loader, desc="Validation (1000 steps)"):
            val_mri = val_mri.to(device)
            generated_ct = generate_conditional_ct(model, val_mri)
            gen_np = (generated_ct.cpu().numpy() + 1) / 2
            gt_np = (val_gt_ct.cpu().numpy() + 1) / 2
            for j in range(gen_np.shape[0]):
                ssim_scores.append(ssim(gt_np[j,0], gen_np[j,0], data_range=1.0))
                psnr_scores.append(psnr(gt_np[j,0], gen_np[j,0], data_range=1.0))
                mae_scores.append(np.mean(np.abs(gt_np[j,0]-gen_np[j,0])))
    return np.mean(ssim_scores), np.mean(psnr_scores), np.mean(mae_scores)

# load checkpoints
if os.path.exists(latest_checkpoint_path):
    print("Loading from latest checkpoint...")
    checkpoint = torch.load(latest_checkpoint_path, map_location=device)
    diffusion.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"]
    best_ssim = checkpoint.get("best_ssim", 0.0)
    best_epoch = checkpoint.get("best_epoch", 0)
    patience_counter = checkpoint.get("patience_counter", 0)
    print(f"Resumed from latest checkpoint: epoch {start_epoch}, best SSIM {best_ssim:.4f}")
    
elif os.path.exists(best_model_path):
    print("No latest checkpoint found. Loading old best model weights...")
    diffusion.load_state_dict(torch.load(best_model_path, map_location=device))
    print("Resumed from old best model weights. Optimizer restarted. Starting from epoch 0.")

else:
    print("No checkpoint found. Starting new training session.")

# quick validation after loading
avg_val_ssim, avg_val_psnr, avg_val_mae = validate_model(diffusion, val_loader, device)
print(f"Validation after loading — SSIM={avg_val_ssim:.4f} | PSNR={avg_val_psnr:.2f} | MAE={avg_val_mae:.4f}")

# training loop
for epoch in range(start_epoch, epochs):
    print(f"\n--- Starting Epoch {epoch+1}/{epochs} ---")
    diffusion.train()
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

    for batch_idx, (mri_tensor, ct_tensor) in enumerate(progress_bar):
        mri_tensor, ct_tensor = mri_tensor.to(device), ct_tensor.to(device)
        loss = diffusion(torch.cat([mri_tensor, ct_tensor], dim=1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # save checkpoint every 500 batches
        if batch_idx % 500 == 0:
            torch.save({
                "epoch": epoch,
                "batch": batch_idx,
                "model_state": diffusion.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_ssim": best_ssim,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter
            }, latest_checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}, batch {batch_idx}")

    # validation phase
    avg_ssim, avg_psnr, avg_mae = validate_model(diffusion, val_loader, device)
    print(f"Validation (50 samples, 1000 steps) — SSIM={avg_ssim:.4f} | PSNR={avg_psnr:.2f} | MAE={avg_mae:.4f}")

    # checkpoint best model
    if avg_ssim > best_ssim:
        best_ssim = avg_ssim
        best_epoch = epoch + 1
        torch.save(diffusion.state_dict(), best_model_path)
        print(f"New best model saved with SSIM={best_ssim:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"No improvement. Patience {patience_counter}/{patience}")

    # save the latest state after each epoch for resuming
    torch.save({
        "epoch": epoch + 1,
        "model_state": diffusion.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_ssim": best_ssim,
        "best_epoch": best_epoch,
        "patience_counter": patience_counter
    }, latest_checkpoint_path)
    print(f"Epoch {epoch+1} complete. Latest checkpoint saved.")

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

print(f"\nTraining complete. Best model from epoch {best_epoch} with SSIM={best_ssim:.4f} saved at {best_model_path}")
