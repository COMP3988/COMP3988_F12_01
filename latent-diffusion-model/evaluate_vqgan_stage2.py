"""
Evaluation script for Stage 2 VQ-GAN (MRI-to-CT Translation)
Generates CT images from MRI inputs and calculates SSIM scores
"""

import os
import sys
import torch
import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

# Add VQ-GAN path
sys.path.append('/Users/joshareshua/Documents/USYD CS/COMP3988_F12_01/latent-diffusion-model/ALDM/VQ-GAN')
from taming.models.vqgan import VQModel


def load_model(config_path, checkpoint_path):
    """Load the trained Stage 2 VQ-GAN model"""
    config = OmegaConf.load(config_path)
    model_config = config.model.params
    
    # Instantiate model
    model = VQModel(
        ddconfig=model_config.ddconfig,
        lossconfig=model_config.lossconfig,
        n_embed=model_config.n_embed,
        embed_dim=model_config.embed_dim,
        num_classes=model_config.num_classes,
        stage=model_config.stage
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    
    return model


def load_mri_ct_volume(mri_path, ct_path, image_size=64):
    """Load and preprocess MRI and CT volumes"""
    # Read volumes
    mr_vol = sitk.GetArrayFromImage(sitk.ReadImage(mri_path, sitk.sitkFloat32))
    ct_vol = sitk.GetArrayFromImage(sitk.ReadImage(ct_path, sitk.sitkFloat32))
    
    # Normalize
    mr_vol = (mr_vol - np.min(mr_vol)) / (np.max(mr_vol) - np.min(mr_vol) + 1e-8)
    ct_vol = (ct_vol - np.min(ct_vol)) / (np.max(ct_vol) - np.min(ct_vol) + 1e-8)
    
    # Convert to tensor
    mr_tensor = torch.from_numpy(mr_vol).unsqueeze(0).float()
    ct_tensor = torch.from_numpy(ct_vol).unsqueeze(0).float()
    
    # Resize
    mr_tensor = torch.nn.functional.interpolate(
        mr_tensor.unsqueeze(0), 
        size=(image_size, image_size, image_size), 
        mode='trilinear', 
        align_corners=False
    ).squeeze(0)
    
    ct_tensor = torch.nn.functional.interpolate(
        ct_tensor.unsqueeze(0), 
        size=(image_size, image_size, image_size), 
        mode='trilinear', 
        align_corners=False
    ).squeeze(0)
    
    # Normalize to [-1, 1]
    mr_tensor = 2.0 * mr_tensor - 1.0
    ct_tensor = 2.0 * ct_tensor - 1.0
    
    return mr_tensor, ct_tensor


def generate_ct_from_mri(model, mri_tensor, target_class=0):
    """Generate CT image from MRI using Stage 2 VQ-GAN with SPADE"""
    with torch.no_grad():
        mri_batch = mri_tensor.unsqueeze(0)  # Add batch dimension
        
        # Encode MRI to latent space
        xsrc, _, _ = model.encode(mri_batch)
        
        # Apply SPADE transformation with target class (0 = CT)
        y = torch.tensor([target_class], dtype=torch.long)
        xrec = model.spade(xsrc, y)
        
        # Decode back to image space
        generated_ct = model.decode(xrec)
        
    return generated_ct.squeeze(0).squeeze(0)  # Remove batch and channel dims


def calculate_metrics(img1, img2):
    """Calculate SSIM, PSNR, and MAE between two volumes"""
    # Convert to numpy and denormalize from [-1, 1] to [0, 1]
    img1_np = ((img1.cpu().numpy() + 1.0) / 2.0).clip(0, 1)
    img2_np = ((img2.cpu().numpy() + 1.0) / 2.0).clip(0, 1)
    
    # Calculate SSIM for the entire 3D volume
    ssim_score = ssim(img1_np, img2_np, data_range=1.0)
    
    # Calculate PSNR
    mse = np.mean((img1_np - img2_np) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # Calculate MAE (Mean Absolute Error)
    mae = np.mean(np.abs(img1_np - img2_np))
    
    return ssim_score, psnr, mae


def calculate_slice_ssim(img1, img2):
    """Calculate average SSIM across all slices"""
    img1_np = ((img1.cpu().numpy() + 1.0) / 2.0).clip(0, 1)
    img2_np = ((img2.cpu().numpy() + 1.0) / 2.0).clip(0, 1)
    
    slice_ssims = []
    for i in range(img1_np.shape[0]):
        slice_ssim = ssim(img1_np[i], img2_np[i], data_range=1.0)
        slice_ssims.append(slice_ssim)
    
    return np.mean(slice_ssims), slice_ssims


def save_slice_visualization(mri, ct_real, ct_generated, output_path, slice_idx=None):
    """Save a visualization comparing MRI input, real CT, and generated CT"""
    # Denormalize from [-1, 1] to [0, 1]
    mri_np = ((mri.cpu().numpy() + 1.0) / 2.0).clip(0, 1)
    ct_real_np = ((ct_real.cpu().numpy() + 1.0) / 2.0).clip(0, 1)
    ct_gen_np = ((ct_generated.cpu().numpy() + 1.0) / 2.0).clip(0, 1)
    
    # Get middle slice if not specified
    if slice_idx is None:
        slice_idx = mri_np.shape[0] // 2
    
    # Calculate SSIM for this slice
    slice_ssim = ssim(ct_real_np[slice_idx], ct_gen_np[slice_idx], data_range=1.0)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(mri_np[slice_idx], cmap='gray')
    axes[0].set_title('Input MRI', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(ct_real_np[slice_idx], cmap='gray')
    axes[1].set_title('Ground Truth CT', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(ct_gen_np[slice_idx], cmap='gray')
    axes[2].set_title(f'Generated CT\nSSIM: {slice_ssim:.4f}', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved visualization to {output_path}")


def main():
    print("="*60)
    print("VQ-GAN Stage 2: MRI-to-CT Translation Evaluation")
    print("="*60)
    
    # Paths
    config_path = '/Users/joshareshua/Documents/USYD CS/COMP3988_F12_01/latent-diffusion-model/ALDM/VQ-GAN/configs/brats_vqgan_stage2.yaml'
    checkpoint_path = '/Users/joshareshua/Documents/USYD CS/COMP3988_F12_01/latent-diffusion-model/ALDM/VQ-GAN/logs/stage2_training/version_2/checkpoints/epoch=29-step=89.ckpt'
    sample_dataset_path = '/Users/joshareshua/Documents/USYD CS/COMP3988_F12_01/latent-diffusion-model/sample-dataset'
    output_dir = '/Users/joshareshua/Documents/USYD CS/COMP3988_F12_01/latent-diffusion-model/results'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("\n[1/4] Loading trained Stage 2 VQ-GAN model...")
    model = load_model(config_path, checkpoint_path)
    print("✓ Model loaded successfully")
    
    # Get patient folders
    patient_folders = sorted([d for d in os.listdir(sample_dataset_path) 
                             if os.path.isdir(os.path.join(sample_dataset_path, d))])
    
    print(f"\n[2/4] Found {len(patient_folders)} patient samples: {patient_folders}")
    
    # Process each patient
    ssim_scores_3d = []
    ssim_scores_slicewise = []
    psnr_scores = []
    mae_scores = []
    
    print("\n[3/4] Generating CT images and calculating metrics...")
    for idx, patient_id in enumerate(patient_folders):
        print(f"\n  Patient {patient_id} ({idx+1}/{len(patient_folders)}):")
        
        mri_path = os.path.join(sample_dataset_path, patient_id, 'mr.mha')
        ct_path = os.path.join(sample_dataset_path, patient_id, 'ct.mha')
        
        # Load data
        print(f"    Loading MRI and CT volumes...")
        mri_tensor, ct_real_tensor = load_mri_ct_volume(mri_path, ct_path, image_size=64)
        
        # Generate CT from MRI
        print(f"    Generating CT from MRI using SPADE...")
        ct_generated_tensor = generate_ct_from_mri(model, mri_tensor, target_class=0)
        
        # Calculate all metrics
        ssim_3d, psnr, mae = calculate_metrics(ct_generated_tensor, ct_real_tensor.squeeze(0))
        ssim_scores_3d.append(ssim_3d)
        psnr_scores.append(psnr)
        mae_scores.append(mae)
        
        # Calculate slice-wise SSIM
        ssim_avg_slices, _ = calculate_slice_ssim(ct_generated_tensor, ct_real_tensor.squeeze(0))
        ssim_scores_slicewise.append(ssim_avg_slices)
        
        print(f"    ✓ SSIM:  {ssim_3d:.4f}")
        print(f"    ✓ PSNR:  {psnr:.2f} dB")
        print(f"    ✓ MAE:   {mae:.4f}")
        print(f"    ✓ Slice SSIM: {ssim_avg_slices:.4f}")
        
        # Save visualization
        output_path = os.path.join(output_dir, f'{patient_id}_mri_to_ct_comparison.png')
        save_slice_visualization(
            mri_tensor.squeeze(0), 
            ct_real_tensor.squeeze(0), 
            ct_generated_tensor,
            output_path
        )
    
    # Summary
    print("\n" + "="*60)
    print("[4/4] EVALUATION SUMMARY")
    print("="*60)
    print(f"\nTotal samples evaluated: {len(ssim_scores_3d)}")
    print(f"\nPer-Sample Results:")
    print(f"{'─'*60}")
    print(f"{'Patient':<12} {'SSIM':>8} {'PSNR (dB)':>12} {'MAE':>8}")
    print(f"{'─'*60}")
    for patient_id, s, p, m in zip(patient_folders, ssim_scores_3d, psnr_scores, mae_scores):
        print(f"{patient_id:<12} {s:>8.4f} {p:>12.2f} {m:>8.4f}")
    
    avg_ssim_3d = np.mean(ssim_scores_3d)
    std_ssim_3d = np.std(ssim_scores_3d)
    
    avg_ssim_slices = np.mean(ssim_scores_slicewise)
    std_ssim_slices = np.std(ssim_scores_slicewise)
    
    avg_psnr = np.mean(psnr_scores)
    std_psnr = np.std(psnr_scores)
    
    avg_mae = np.mean(mae_scores)
    std_mae = np.std(mae_scores)
    
    print(f"{'─'*60}")
    print(f"\nAVERAGE METRICS (across {len(ssim_scores_3d)} samples):")
    print(f"{'─'*60}")
    print(f"SSIM (3D):           {avg_ssim_3d:.4f} ± {std_ssim_3d:.4f}")
    print(f"SSIM (Slice-wise):   {avg_ssim_slices:.4f} ± {std_ssim_slices:.4f}")
    print(f"PSNR:                {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"MAE:                 {avg_mae:.4f} ± {std_mae:.4f}")
    print(f"{'─'*60}")
    
    # Analysis
    print(f"\n⚠️  PERFORMANCE ANALYSIS:")
    print(f"{'─'*60}")
    print(f"Resolution used:      64×64×64 (reduced from 192×192×192)")
    print(f"Training samples:     3 volumes only")
    print(f"Training epochs:      30 per stage (90 total)")
    print(f"Training device:      CPU (very slow)")
    print(f"\n💡 TO IMPROVE SSIM:")
    print(f"  1. Increase resolution to 128×128×128 or 192×192×192")
    print(f"  2. Train on more data (at least 50-100 samples)")
    print(f"  3. Train for more epochs (100-200)")
    print(f"  4. Use GPU for faster, more stable training")
    print(f"{'─'*60}")
    
    print(f"\n✓ Results saved to: {output_dir}/")
    print(f"  - Visual comparisons: {len(patient_folders)} PNG files")
    print(f"  - Input: MRI images")
    print(f"  - Output: Generated CT images")
    print(f"  - All {len(patient_folders)} samples processed successfully!")
    
    return {
        'ssim_3d': ssim_scores_3d,
        'ssim_slicewise': ssim_scores_slicewise,
        'psnr': psnr_scores,
        'mae': mae_scores,
        'avg_ssim': avg_ssim_3d,
        'avg_psnr': avg_psnr,
        'avg_mae': avg_mae
    }


if __name__ == "__main__":
    results = main()
    print(f"\n" + "="*60)
    print(f"✓ EVALUATION COMPLETE!")
    print(f"  SSIM: {results['avg_ssim']:.4f}")
    print(f"  PSNR: {results['avg_psnr']:.2f} dB")
    print(f"  MAE:  {results['avg_mae']:.4f}")
    print("="*60)

