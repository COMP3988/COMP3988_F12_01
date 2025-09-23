#tests/test_framework.py
""" 
Main evaluation framework for testing MRI-to-CT synthesis models.

This script serves as the primary entry point for running quatitative evaluation.
It takes a model name, a training weights fle, and a test dataset as input, 
then runs inference and calculates performance metrics (SSIM, PSNR, MAE).
"""

import sys
import os
# Add the parent directory to the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from typing import List, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.adapters import load_model
from evaluation import calculate_metrics
from tests.simple_loader import SimpleTestDataset #TEMP: using this for now

def setup_arg_parser() -> argparse.Namespace:
    """ 
    Sets up and parses the command-line args for the script."""
    parser = argparse.ArgumentParser(description='Run model evaluation framework.')
    parser.add_argument('--model_name', type=str, required=True, choices=['unet', 'pix2pix'],
                        help='Name of the model to text (e.g., "unet").')
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to the trained model weights file (.pth).')
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Path to the folder containing the test dataset.')
    return parser.parse_args()

def run_evaluation_loop(model, data_loader: DataLoader) -> List[Dict[str, float]]:
    """ 
    Runs the inference and evaluation loop for all images in the data loader.
    
    Args:
        model: The initialised and loaded ModelAdapter object.
        data_loader (DataLoader): The PyTorch DataLoader for the test set. 
        
    Returns:
        List[Dict[str, float]]: A list of dict where each dict contains the 
            SSIM, PSNR, and MAE for one test image. 
    """
    results = []
    print(f"Running interface on {len(data_loader)} test images...")
    
    for mri_image, ground_truth_ct in data_loader:
        # Run model's predict method to get the generated CT scan.
        generated_ct = model.predict(mri_image)
        
        # Convert tensors to NumPy arrays for eval.
        # .squeeze() to remove extra dimensions. 
        generated_np = generated_ct.squeeze().numpy()
        ground_truth_np = ground_truth_ct.squeeze().numpy()
        
        # Use the function from evaluation.py to calculate the metrics. 
        metrics = calculate_metrics(generated_np, ground_truth_np)
        results.append(metrics)
    
    print("Evaluation complete.")
    return results

def print_summary(model_name: str, results: List[Dict[str, float]]): 
    """ Calculates and prints the final performance summary. """
    ssim_scores = [r['ssim'] for r in results]
    psnr_scores = [r['psnr'] for r in results]
    mae_scores = [r['mae'] for r in results]
    
    print("\n" + "="*40)
    print("   PERFORMANCE SUMMARY")
    print("="*40)
    print(f"Model: {model_name.upper()}")
    print(f"Number of test images: {len(results)}")
    print("-" * 40)
    print(f"SSIM: {np.mean(ssim_scores):.3f} ± {np.std(ssim_scores):.3f}")
    print(f"PSNR: {np.mean(psnr_scores):.2f} ± {np.std(psnr_scores):.2f} dB")
    print(f"MAE:  {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
    print("="*40 + "\n")
    
    
def main():
    """ 
    The main function.
    """
    args = setup_arg_parser()
    
    print(f"-- Starting evaluation for {args.model_name.upper()} model --")
    
    # Load the model.
    print(f"Loading model from: {args.weights_path}")
    model = load_model(args.model_name, args.weights_path)
    
    # Load the test dataset.
    print(f"Loading data from: {args.data_folder}")
    test_dataset = SimpleTestDataset(data_folder=args.data_folder)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    
    # Run the main loop to get results.
    results = run_evaluation_loop(model, test_loader)
    
    # Print summary.
    print_summary(args.model_name, results)
    
    
if __name__ == '__main__':
    main()