import os
import SimpleITK as sitk
import numpy as np
import torch
from omegaconf import OmegaConf
# from main import instantiate_from_config  # Moved below
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from ldm.models.diffusion.ddim import DDIMSampler

# Ensure the path to the LDM directory is in sys.path
import sys
sys.path.append(os.path.abspath('./ALDM/LDM'))

# Import from LDM main
from ALDM.LDM.main import instantiate_from_config

def load_mri_ct_volume(mri_path, ct_path, image_size=64):
    """Load and preprocess MRI and CT volumes"""
    # Read volumes
    mri_vol = sitk.GetArrayFromImage(sitk.ReadImage(mri_path, sitk.sitkFloat32))
    ct_vol = sitk.GetArrayFromImage(sitk.ReadImage(ct_path, sitk.sitkFloat32))
    
    # Normalize volumes
    mri_vol = (mri_vol - np.min(mri_vol)) / (np.max(mri_vol) - np.min(mri_vol) + 1e-8)
    ct_vol = (ct_vol - np.min(ct_vol)) / (np.max(ct_vol) - np.min(ct_vol) + 1e-8)
    
    # Convert to tensor and add channel dimension: (D, H, W) -> (1, D, H, W)
    mri_tensor = torch.from_numpy(mri_vol).unsqueeze(0).float()
    ct_tensor = torch.from_numpy(ct_vol).unsqueeze(0).float()
    
    # Resize spatially using interpolation
    mri_tensor = torch.nn.functional.interpolate(
        mri_tensor.unsqueeze(0), 
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
    mri_tensor = 2.0 * mri_tensor - 1.0
    ct_tensor = 2.0 * ct_tensor - 1.0
    
    return mri_tensor, ct_tensor

def generate_ct_from_mri_ldm(model, sampler, mri_tensor, num_steps=50):
    """Generate CT from MRI using the complete LDM"""
    model.eval()
    
    with torch.no_grad():
        # Encode MRI to latent space
        mri_latent = model.get_first_stage_encoding(model.encode_first_stage(mri_tensor))
        
        # Generate CT in latent space using DDIM sampling
        # Start with noise
        shape = mri_latent.shape
        noise = torch.randn_like(mri_latent)
        
        # Use DDIM sampling
        samples, _ = sampler.sample(
            S=num_steps,
            conditioning=mri_latent,  # Use MRI as conditioning
            batch_size=1,
            shape=shape[1:],  # Remove batch dimension
            verbose=False
        )
        
        # Decode back to image space
        ct_generated = model.decode_first_stage(samples)
        
        # Denormalize from [-1, 1] to [0, 1]
        ct_generated = (ct_generated + 1.0) / 2.0
        ct_generated = torch.clamp(ct_generated, 0, 1)
        
        return ct_generated

def calculate_metrics(pred, target):
    """Calculate SSIM, PSNR, and MAE between predicted and target volumes"""
    # Convert to numpy and ensure same shape
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Calculate 3D SSIM
    ssim_3d = ssim(target_np, pred_np, data_range=1.0)
    
    # Calculate PSNR
    mse = np.mean((target_np - pred_np) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # Calculate MAE
    mae = np.mean(np.abs(target_np - pred_np))
    
    return ssim_3d, psnr, mae

def calculate_slice_ssim(pred, target):
    """Calculate average slice-wise SSIM"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    slice_ssims = []
    for i in range(pred_np.shape[1]):  # Iterate through depth dimension
        slice_ssim = ssim(
            target_np[0, i, :, :], 
            pred_np[0, i, :, :], 
            data_range=1.0
        )
        slice_ssims.append(slice_ssim)
    
    return np.mean(slice_ssims), slice_ssims

def save_slice_visualization(mri_vol, ct_real, ct_generated, output_path, num_slices=8):
    """Save visualization comparing MRI, real CT, and generated CT"""
    # Convert to numpy
    mri_np = mri_vol.detach().cpu().numpy()
    ct_real_np = ct_real.detach().cpu().numpy()
    ct_gen_np = ct_generated.detach().cpu().numpy()
    
    # Select slices to visualize
    depth = mri_np.shape[1]
    slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
    
    fig, axes = plt.subplots(3, num_slices, figsize=(num_slices*2, 6))
    
    for i, slice_idx in enumerate(slice_indices):
        # MRI slice
        axes[0, i].imshow(mri_np[0, slice_idx, :, :], cmap='gray')
        axes[0, i].set_title(f'MRI Slice {slice_idx}')
        axes[0, i].axis('off')
        
        # Real CT slice
        axes[1, i].imshow(ct_real_np[0, slice_idx, :, :], cmap='gray')
        axes[1, i].set_title(f'Real CT Slice {slice_idx}')
        axes[1, i].axis('off')
        
        # Generated CT slice
        axes[2, i].imshow(ct_gen_np[0, slice_idx, :, :], cmap='gray')
        axes[2, i].set_title(f'Generated CT Slice {slice_idx}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    ldm_config_path = os.path.join(project_root, 'ALDM', 'LDM', 'configs', 'latent-diffusion', 'brats-ldm-vq-4.yaml')
    ldm_checkpoint_path = os.path.join(project_root, 'ALDM', 'LDM', 'logs', 'checkpoints_ldm', 'epoch_epoch=29-v1.ckpt')
    sample_dataset_path = os.path.join(project_root, 'sample-dataset')
    output_dir = os.path.join(project_root, 'results_ldm')
    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("COMPLETE LDM: MRI-to-CT Translation Evaluation (All 3 Stages)")
    print("="*70)

    # Load LDM model
    print("\n[1/4] Loading trained LDM model...")
    config = OmegaConf.load(ldm_config_path)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ldm_checkpoint_path, map_location="cpu")["state_dict"])
    model.eval()
    model.freeze()
    
    # Create DDIM sampler
    sampler = DDIMSampler(model)
    
    print("✓ LDM model loaded successfully")
    print("✓ DDIM sampler initialized")

    # Get patient folders
    patient_folders = sorted([d for d in os.listdir(sample_dataset_path) 
                              if os.path.isdir(os.path.join(sample_dataset_path, d))])
    
    print(f"\n[2/4] Found {len(patient_folders)} patient samples: {patient_folders}")
    
    # Process each patient
    ssim_scores_3d = []
    ssim_scores_slicewise = []
    psnr_scores = []
    mae_scores = []
    
    print("\n[3/4] Generating CT images using complete LDM and calculating metrics...")
    for idx, patient_id in enumerate(patient_folders):
        print(f"\n  Patient {patient_id} ({idx+1}/{len(patient_folders)}):")
        
        mri_path = os.path.join(sample_dataset_path, patient_id, 'mr.mha')
        ct_path = os.path.join(sample_dataset_path, patient_id, 'ct.mha')
        
        # Load data
        print(f"Loading MRI and CT volumes...")
        mri_tensor, ct_real_tensor = load_mri_ct_volume(mri_path, ct_path, image_size=64)
        
        # Generate CT from MRI using complete LDM
        print(f"    Generating CT from MRI using LDM...")
        ct_generated_tensor = generate_ct_from_mri_ldm(model, sampler, mri_tensor, num_steps=50)
        
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
        output_path = os.path.join(output_dir, f'{patient_id}_ldm_mri_to_ct_comparison.png')
        save_slice_visualization(
            mri_tensor.squeeze(0), 
            ct_real_tensor.squeeze(0), 
            ct_generated_tensor,
            output_path
        )
    
    # Summary
    print("\n" + "="*70)
    print("[4/4] EVALUATION SUMMARY - COMPLETE LDM (3 STAGES)")
    print("="*70)
    print(f"\nTotal samples evaluated: {len(ssim_scores_3d)}")
    print(f"\nPer-Sample Results:")
    print(f"{'─'*70}")
    print(f"{'Patient':<12} {'SSIM':>8} {'PSNR (dB)':>12} {'MAE':>8}")
    print(f"{'─'*70}")
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
    
    print(f"{'─'*70}")
    print(f"\nAVERAGE METRICS (across {len(ssim_scores_3d)} samples):")
    print(f"{'─'*70}")
    print(f"SSIM (3D):           {avg_ssim_3d:.4f} ± {std_ssim_3d:.4f}")
    print(f"SSIM (Slice-wise):   {avg_ssim_slices:.4f} ± {std_ssim_slices:.4f}")
    print(f"PSNR:                {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"MAE:                 {avg_mae:.4f} ± {std_mae:.4f}")
    print(f"{'─'*70}")
    
    # Comparison with Stage 2 only
    print(f"\nCOMPARISON: Stage 2 VQ-GAN vs Complete LDM (Stage 3)")
    print(f"{'─'*70}")
    print(f"Stage 2 VQ-GAN only:     SSIM ≈ 0.37, PSNR ≈ 20 dB")
    print(f"Complete LDM (3 stages): SSIM = {avg_ssim_3d:.4f}, PSNR = {avg_psnr:.2f} dB")
    
    improvement_ssim = ((avg_ssim_3d - 0.37) / 0.37) * 100
    improvement_psnr = avg_psnr - 20
    
    print(f"SSIM Improvement:       {improvement_ssim:+.1f}%")
    print(f"PSNR Improvement:       {improvement_psnr:+.1f} dB")
    print(f"{'─'*70}")
    
    # Analysis
    print(f"\nPERFORMANCE ANALYSIS:")
    print(f"{'─'*70}")
    print(f"Model: Complete LDM (VQ-GAN + SPADE + Diffusion)")
    print(f"Resolution: 64×64×64 (reduced from 192×192×192)")
    print(f"Training samples: 3 volumes only")
    print(f"Training epochs: 30 per stage (90 total)")
    print(f"Training device: CPU (very slow)")
    print(f"DDIM sampling steps: 50")
    print(f"\nTO IMPROVE FURTHER:")
    print(f"  1. Increase resolution to 128×128×128 or 192×192×192")
    print(f"  2. Train on more data (at least 50-100 samples)")
    print(f"  3. Train for more epochs (100-200 per stage)")
    print(f"  4. Use GPU for faster, more stable training")
    print(f"  5. Fine-tune DDIM sampling parameters")
    print(f"{'─'*70}")
    
    print(f"\n✓ Results saved to: {output_dir}/")
    print(f"  - Visual comparisons: {len(patient_folders)} PNG files")
    print(f"  - Input: MRI images")
    print(f"  - Output: Generated CT images using complete LDM")
    print(f"  - All {len(patient_folders)} samples processed successfully!")
    
    return {
        'ssim_3d': ssim_scores_3d,
        'ssim_slicewise': ssim_scores_slicewise,
        'psnr': psnr_scores,
        'mae': mae_scores,
        'avg_ssim': avg_ssim_3d,
        'avg_psnr': avg_psnr,
        'avg_mae': avg_mae,
        'improvement_ssim': improvement_ssim,
        'improvement_psnr': improvement_psnr
    }

if __name__ == "__main__":
    results = main()
    print(f"\n" + "="*70)
    print(f"✓ COMPLETE LDM EVALUATION FINISHED:")
    print(f"  Final SSIM: {results['avg_ssim']:.4f}")
    print(f"  Final PSNR: {results['avg_psnr']:.2f} dB")
    print(f"  SSIM Improvement: {results['improvement_ssim']:+.1f}%")
    print("="*70)
