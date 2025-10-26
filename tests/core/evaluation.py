from typing import Dict
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def calculate_metrics(generated_image, ground_truth_image):
    """ 
    Calculates all performance metrics for a single pair of 2D images.
    
    Args: 
    Assumes a 2D NumPy array with pixel values normalised to the [0, 1] range.
        generated_image (np.ndarray):   The model's output image.
        ground_truth_image (np.ndarray):    The real target image.
        
    Returns:
        Dict[str, float]: A dictionary containing the calculated scores for 'ssim', 'psnr', 'mae'.
    """

    ssim_score = ssim(ground_truth_image, generated_image, data_range=1.0)
    psnr_score = psnr(ground_truth_image, generated_image, data_range=1.0)
    mae_score = np.mean(np.abs(ground_truth_image - generated_image))
    
    return {'ssim': ssim_score, 'psnr': psnr_score, 'mae': mae_score}