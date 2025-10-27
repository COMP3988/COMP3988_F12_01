from typing import Dict
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def calculate_metrics(generated_image, ground_truth_image):
    """
    Calculates all performance metrics for a single pair of 2D images.

    Args:
    Assumes a NumPy array with pixel values. Automatically detects data range.
        generated_image (np.ndarray):   The model's output image.
        ground_truth_image (np.ndarray):    The real target image.

    Returns:
        Dict[str, float]: A dictionary containing the calculated scores for 'ssim', 'psnr', 'mae'.
    """
    # Detect data range: check if values are in [-1, 1] or [0, 1]
    img_min = min(generated_image.min(), ground_truth_image.min())
    img_max = max(generated_image.max(), ground_truth_image.max())

    if img_min >= -1.1 and img_max <= 1.1:  # [-1, 1] range
        data_range = 2.0
    elif img_min >= -0.1 and img_max <= 1.1:  # [0, 1] range
        data_range = 1.0
    else:
        # Auto-detect from actual range
        data_range = max(img_max - img_min, 1.0)

    ssim_score = ssim(ground_truth_image, generated_image, data_range=data_range)
    psnr_score = psnr(ground_truth_image, generated_image, data_range=data_range)
    mae_score = np.mean(np.abs(ground_truth_image - generated_image))

    return {'ssim': ssim_score, 'psnr': psnr_score, 'mae': mae_score}