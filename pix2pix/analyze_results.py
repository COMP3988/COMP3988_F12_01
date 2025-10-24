#!/usr/bin/env python3
"""
Analyze Pix2Pix training results and generate metrics for demo
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import glob

def extract_training_metrics():
    """Extract ACTUAL training metrics from your training output"""
    # Real values from your training output - only the epochs we have data for
    epochs_with_data = [5, 10, 15, 20, 25, 30]
    
    # ACTUAL values from your training output
    g_gan = [4.189, 0.161, 2.792, 0.684, 1.711, 3.008]
    g_l1 = [34.335, 21.417, 33.252, 19.310, 30.575, 31.518]
    d_real = [0.025, 0.850, 0.157, 0.938, 0.600, 0.025]
    d_fake = [0.018, 0.016, 0.064, 0.033, 0.062, 0.188]
    
    print("Using ACTUAL training metrics from your 30-epoch run:")
    print(f"Epochs with data: {epochs_with_data}")
    print(f"G_GAN values: {g_gan}")
    print(f"G_L1 values: {g_l1}")
    print(f"D_real values: {d_real}")
    print(f"D_fake values: {d_fake}")
    
    return epochs_with_data, g_gan, g_l1, d_real, d_fake

def calculate_image_metrics(results_dir):
    """Calculate SSIM, PSNR, and MAE for generated images"""
    images_dir = os.path.join(results_dir, "images")
    
    ssim_scores = []
    psnr_scores = []
    mae_scores = []
    
    # Get all generated images
    fake_images = sorted(glob.glob(os.path.join(images_dir, "*_fake_B.png")))
    real_images = sorted(glob.glob(os.path.join(images_dir, "*_real_A.png")))
    
    print(f"Found {len(fake_images)} generated images to analyze")
    
    for fake_path, real_path in zip(fake_images, real_images):
        try:
            # Load images
            fake_img = np.array(Image.open(fake_path).convert('L'))  # Convert to grayscale
            real_img = np.array(Image.open(real_path).convert('L'))
            
            # Normalize to 0-1 range
            fake_img = fake_img.astype(np.float32) / 255.0
            real_img = real_img.astype(np.float32) / 255.0
            
            # Calculate metrics
            ssim_score = ssim(real_img, fake_img, data_range=1.0)
            psnr_score = psnr(real_img, fake_img, data_range=1.0)
            mae_score = np.mean(np.abs(real_img - fake_img))
            
            ssim_scores.append(ssim_score)
            psnr_scores.append(psnr_score)
            mae_scores.append(mae_score)
            
        except Exception as e:
            print(f"Error processing {fake_path}: {e}")
    
    return ssim_scores, psnr_scores, mae_scores

def plot_training_curves(epochs, g_gan, g_l1, d_real, d_fake):
    """Plot training loss curves using ACTUAL data points"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Generator GAN Loss
    ax1.plot(epochs, g_gan, 'bo-', linewidth=2, markersize=8, label='Generator GAN Loss')
    ax1.set_title('Generator Adversarial Loss (Real Data Points)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Generator L1 Loss
    ax2.plot(epochs, g_l1, 'ro-', linewidth=2, markersize=8, label='Generator L1 Loss')
    ax2.set_title('Generator L1 Loss - Pixel Accuracy (Real Data Points)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Discriminator Real Loss
    ax3.plot(epochs, d_real, 'go-', linewidth=2, markersize=8, label='Discriminator Real Loss')
    ax3.set_title('Discriminator Real Loss (Real Data Points)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Discriminator Fake Loss
    ax4.plot(epochs, d_fake, 'mo-', linewidth=2, markersize=8, label='Discriminator Fake Loss')
    ax4.set_title('Discriminator Fake Loss (Real Data Points)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved as 'training_curves.png'")
    plt.close()  # Close to avoid display issues

def plot_image_metrics(ssim_scores, psnr_scores, mae_scores):
    """Plot image quality metrics"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # SSIM scores
    ax1.hist(ssim_scores, bins=10, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title(f'SSIM Distribution\nMean: {np.mean(ssim_scores):.3f}', fontweight='bold')
    ax1.set_xlabel('SSIM Score')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # PSNR scores
    ax2.hist(psnr_scores, bins=10, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title(f'PSNR Distribution\nMean: {np.mean(psnr_scores):.2f} dB', fontweight='bold')
    ax2.set_xlabel('PSNR (dB)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # MAE scores
    ax3.hist(mae_scores, bins=10, alpha=0.7, color='red', edgecolor='black')
    ax3.set_title(f'MAE Distribution\nMean: {np.mean(mae_scores):.3f}', fontweight='bold')
    ax3.set_xlabel('MAE')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('image_quality_metrics.png', dpi=300, bbox_inches='tight')
    print("Image quality metrics saved as 'image_quality_metrics.png'")
    plt.close()  # Close to avoid display issues

def print_summary(ssim_scores, psnr_scores, mae_scores):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("PIX2PIX MRI-TO-CT CONVERSION RESULTS SUMMARY")
    print("="*60)
    print(f"Number of test images: {len(ssim_scores)}")
    print(f"\nImage Quality Metrics:")
    print(f"  SSIM (Structural Similarity): {np.mean(ssim_scores):.3f} ± {np.std(ssim_scores):.3f}")
    print(f"  PSNR (Peak Signal-to-Noise): {np.mean(psnr_scores):.2f} ± {np.std(psnr_scores):.2f} dB")
    print(f"  MAE (Mean Absolute Error): {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
    
    print(f"\nInterpretation:")
    print(f"  • SSIM > 0.7: Good structural similarity")
    print(f"  • PSNR > 20 dB: Acceptable image quality")
    print(f"  • MAE < 0.1: Low pixel-wise error")
    
    # Overall assessment
    avg_ssim = np.mean(ssim_scores)
    avg_psnr = np.mean(psnr_scores)
    avg_mae = np.mean(mae_scores)
    
    print(f"\nOverall Assessment:")
    if avg_ssim > 0.7 and avg_psnr > 20 and avg_mae < 0.1:
        print("  EXCELLENT: Model shows high quality CT generation")
    elif avg_ssim > 0.5 and avg_psnr > 15 and avg_mae < 0.2:
        print("  GOOD: Model shows acceptable CT generation")
    else:
        print(" NEEDS IMPROVEMENT: Consider more training or data")

def main():
    print("Analyzing Pix2Pix MRI-to-CT conversion results...")
    
    # Extract training metrics
    epochs, g_gan, g_l1, d_real, d_fake = extract_training_metrics()
    
    # Plot training curves
    print("Generating training curves...")
    plot_training_curves(epochs, g_gan, g_l1, d_real, d_fake)
    
    # Calculate image quality metrics
    results_dir = "results/mri_to_ct_30epochs/test_latest"
    if os.path.exists(results_dir):
        print("Calculating image quality metrics...")
        ssim_scores, psnr_scores, mae_scores = calculate_image_metrics(results_dir)
        
        # Plot image quality metrics
        plot_image_metrics(ssim_scores, psnr_scores, mae_scores)
        
        # Print summary
        print_summary(ssim_scores, psnr_scores, mae_scores)
    else:
        print(f"Results directory not found: {results_dir}")

if __name__ == "__main__":
    main()
