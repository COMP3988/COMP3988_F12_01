#tests/test_framework.py
"""
Enhanced evaluation framework for testing MRI-to-CT synthesis models.

This script serves as the primary entry point for running comprehensive evaluation.
It supports both basic evaluation and robust testing with edge case simulation.

Features:
- Basic evaluation with SSIM, PSNR, MAE metrics
- Robust testing with edge case simulation (resampling, noise, misalignment, intensity shifts)
- HU (Hounsfield Unit) analysis for CT images
- Pass/fail criteria evaluation
- Comprehensive reporting and visualization

Usage:
    python test_framework.py --model_name unet --weights_path model.pth --data_folder ./test_data
    python test_framework.py --model_name unet --weights_path model.pth --data_folder ./test_data --robust_testing
"""

import sys
import os
# Add the parent directory to the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from typing import List, Dict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.adapters import load_model
from tests.core.evaluation import calculate_metrics
from tests.core.simple_loader import SimpleTestDataset

# Optional imports for enhanced functionality
try:
    from tests.core.robust_testing_pipeline import RobustTestingPipeline
    ROBUST_TESTING_AVAILABLE = True
except ImportError:
    ROBUST_TESTING_AVAILABLE = False

try:
    from tests.core.enhanced_evaluation import calculate_enhanced_metrics
    ENHANCED_METRICS_AVAILABLE = True
except ImportError:
    ENHANCED_METRICS_AVAILABLE = False

def setup_arg_parser() -> argparse.Namespace:
    """
    Sets up and parses the command-line args for the script."""
    parser = argparse.ArgumentParser(description='Enhanced model evaluation framework.')
    parser.add_argument('--model_name', type=str, required=True, choices=['unet', 'pix2pix'],
                        help='Name of the model to test (e.g., "unet").')
    parser.add_argument('--weights_path', type=str, required=True,
                        help='Path to the trained model weights file (.pth).')
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Path to the folder containing the test dataset.')
    parser.add_argument('--robust_testing', action='store_true',
                        help='Enable robust testing with edge case simulation.')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Output directory for results and reports.')
    parser.add_argument('--edge_case_types', nargs='+',
                        default=['resampling', 'noise', 'misalignment', 'intensity', 'all'],
                        help='Edge case types to test (only used with --robust_testing).')
    parser.add_argument('--enhanced_metrics', action='store_true',
                        help='Use enhanced metrics including HU analysis.')
    parser.add_argument('--custom_thresholds', type=str, default=None,
                        help='Path to JSON file with custom pass/fail thresholds.')
    return parser.parse_args()

def run_evaluation_loop(model, data_loader: DataLoader, enhanced_metrics: bool = False) -> List[Dict[str, float]]:
    """
    Runs the inference and evaluation loop for all images in the data loader.

    Args:
        model: The initialised and loaded ModelAdapter object.
        data_loader (DataLoader): The PyTorch DataLoader for the test set.
        enhanced_metrics: Whether to use enhanced metrics including HU analysis.

    Returns:
        List[Dict[str, float]]: A list of dict where each dict contains the
            metrics for one test image.
    """
    results = []
    print(f"Running inference on {len(data_loader)} test images...")

    for mri_image, ground_truth_ct in data_loader:
        # Run model's predict method to get the generated CT scan.
        generated_ct = model.predict(mri_image)

        # Convert tensors to NumPy arrays for eval.
        # .squeeze() to remove extra dimensions.
        generated_np = generated_ct.squeeze().numpy()
        ground_truth_np = ground_truth_ct.squeeze().numpy()

        # Use appropriate metrics function
        if enhanced_metrics and ENHANCED_METRICS_AVAILABLE:
            metrics = calculate_enhanced_metrics(generated_np, ground_truth_np)
        else:
            metrics = calculate_metrics(generated_np, ground_truth_np)

        results.append(metrics)

    print("Evaluation complete.")
    return results

def print_summary(model_name: str, results: List[Dict[str, float]], enhanced_metrics: bool = False):
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

    if enhanced_metrics and ENHANCED_METRICS_AVAILABLE and 'hu_mae' in results[0]:
        hu_mae_scores = [r['hu_mae'] for r in results]
        hu_correlation_scores = [r['hu_correlation'] for r in results]
        print(f"HU MAE: {np.mean(hu_mae_scores):.1f} ± {np.std(hu_mae_scores):.1f}")
        print(f"HU Correlation: {np.mean(hu_correlation_scores):.3f} ± {np.std(hu_correlation_scores):.3f}")

    print("="*40 + "\n")


def main():
    """
    The main function.
    """
    args = setup_arg_parser()

    print(f"-- Starting evaluation for {args.model_name.upper()} model --")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.robust_testing:
        if not ROBUST_TESTING_AVAILABLE:
            print("ERROR: Robust testing requires additional dependencies.")
            print("Please install: seaborn, matplotlib, scipy")
            print("Falling back to basic evaluation...")
            args.robust_testing = False
        else:
            # Use robust testing pipeline
            print("Using robust testing pipeline with edge case simulation...")

            # Load custom thresholds if provided
            thresholds = None
            if args.custom_thresholds:
                with open(args.custom_thresholds, 'r') as f:
                    thresholds = json.load(f)

            # Initialize robust testing pipeline
            pipeline = RobustTestingPipeline(
                model_name=args.model_name,
                weights_path=args.weights_path,
                thresholds=thresholds
            )

            # Load model
            pipeline.load_model()

            # Run comprehensive evaluation
            results = pipeline.run_comprehensive_evaluation(
                args.data_folder, args.edge_case_types)

            # Save results and create visualizations
            pipeline.save_results(results, args.output_dir)
            pipeline.create_visualizations(results, args.output_dir)

            print(f"\nRobust testing completed. Results saved to {args.output_dir}")
            return

    else:
        # Use basic evaluation
        print("Using basic evaluation framework...")

        # Load the model.
        print(f"Loading model from: {args.weights_path}")
        model = load_model(args.model_name, args.weights_path)

        # Load the test dataset.
        print(f"Loading data from: {args.data_folder}")
        test_dataset = SimpleTestDataset(data_folder=args.data_folder)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

        # Run the main loop to get results.
        results = run_evaluation_loop(model, test_loader, args.enhanced_metrics)

        # Print summary.
        print_summary(args.model_name, results, args.enhanced_metrics)

        # Save basic results
        results_dict = {
            'model_name': args.model_name,
            'weights_path': args.weights_path,
            'data_folder': args.data_folder,
            'enhanced_metrics': args.enhanced_metrics,
            'results': results,
            'summary': {
                'ssim_mean': np.mean([r['ssim'] for r in results]),
                'ssim_std': np.std([r['ssim'] for r in results]),
                'psnr_mean': np.mean([r['psnr'] for r in results]),
                'psnr_std': np.std([r['psnr'] for r in results]),
                'mae_mean': np.mean([r['mae'] for r in results]),
                'mae_std': np.std([r['mae'] for r in results])
            }
        }

        if args.enhanced_metrics and 'hu_mae' in results[0]:
            results_dict['summary'].update({
                'hu_mae_mean': np.mean([r['hu_mae'] for r in results]),
                'hu_mae_std': np.std([r['hu_mae'] for r in results]),
                'hu_correlation_mean': np.mean([r['hu_correlation'] for r in results]),
                'hu_correlation_std': np.std([r['hu_correlation'] for r in results])
            })

        # Save results to JSON
        results_file = output_path / f"{args.model_name}_basic_evaluation.json"
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)

        print(f"Basic evaluation completed. Results saved to {results_file}")


if __name__ == '__main__':
    main()