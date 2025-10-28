# tests/test_framework.py
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
    python test_framework.py --model_name transformer_diffusion --weights_path model.pth --data_folder ./test_data
    python test_framework.py --model_name unet --weights_path model.pth --data_folder ./test_data --robust_testing
    python test_framework.py --model_name transformer_diffusion --weights_path model.pth --data_folder ./test_data --max 5
    python test_framework.py --model_name transformer_diffusion --weights_path model.pth --data_folder ./test_data --diffusion_steps 20
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
from tests.core.simple_loader import SimpleTestDataset, SynthRADTestDataset, NPZTestDataset, SynthRAD2025Dataset, SynthRADTask1Dataset

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

def create_dataset(data_folder: str):
    """
    Automatically detects the dataset type and creates the appropriate dataset.

    Args:
        data_folder (str): Path to the data folder

    Returns:
        Dataset: The appropriate dataset class
    """
    import os
    import glob
    from pathlib import Path

    # Check for SynthRAD2025 Task1 format first (Task1/AB, Task1/HN, Task1/TH)
    if _is_synthrad_task1_format(data_folder):
        print(f"Detected SynthRAD2025 Task1 format - using SynthRADTask1Dataset")
        return SynthRADTask1Dataset(data_folder)

    # Check for SynthRAD2025 section format (patient directories with mr.mha/ct.mha)
    elif _is_synthrad2025_format(data_folder):
        print(f"Detected SynthRAD2025 section format - using SynthRAD2025Dataset")
        return SynthRAD2025Dataset(data_folder)

    # Check for different file types
    png_files = glob.glob(os.path.join(data_folder, '*.png'))
    mha_files = glob.glob(os.path.join(data_folder, '*.mha'))
    npz_files = glob.glob(os.path.join(data_folder, '*.npz'))

    if npz_files:
        print(f"Detected .npz files - using NPZTestDataset")
        return NPZTestDataset(data_folder)
    elif mha_files:
        print(f"Detected .mha files - using SynthRADTestDataset")
        return SynthRADTestDataset(data_folder)
    elif png_files:
        print(f"Detected .png files - using SimpleTestDataset")
        return SimpleTestDataset(data_folder)
    else:
        raise FileNotFoundError(f"No supported files (.png, .mha, .npz) or SynthRAD2025 format found in {data_folder}")


def _is_synthrad_task1_format(data_folder: str) -> bool:
    """
    Check if the data folder contains SynthRAD2025 Task1 format.

    SynthRAD2025 Task1 format:
    - Contains Task1 subdirectory OR is already the Task1 directory
    - Task1 contains AB, HN, TH section directories
    - Each section contains patient directories with mr.mha and ct.mha files
    """
    try:
        data_path = Path(data_folder)
        if not data_path.exists() or not data_path.is_dir():
            return False

        # Check if we're already in Task1 directory or need to look for Task1 subdirectory
        task1_path = data_path
        if data_path.name != "Task1":
            task1_path = data_path / "Task1"
            if not task1_path.exists() or not task1_path.is_dir():
                return False

        # Check for section directories (AB, HN, TH)
        sections = ['AB', 'HN', 'TH']
        found_sections = 0

        for section in sections:
            section_path = task1_path / section
            if section_path.exists() and section_path.is_dir():
                # Check if section has at least one patient with required files
                patient_dirs = [d for d in section_path.iterdir() if d.is_dir()]
                for patient_dir in patient_dirs[:2]:  # Check first 2 patients
                    mr_file = patient_dir / "mr.mha"
                    ct_file = patient_dir / "ct.mha"
                    if mr_file.exists() and ct_file.exists():
                        found_sections += 1
                        break

        # Need at least one valid section
        return found_sections > 0

    except Exception:
        return False


def _is_synthrad2025_format(data_folder: str) -> bool:
    """
    Check if the data folder contains SynthRAD2025 format.

    SynthRAD2025 format:
    - Contains subdirectories (patient folders)
    - Each subdirectory contains mr.mha and ct.mha files
    """
    try:
        data_path = Path(data_folder)
        if not data_path.exists() or not data_path.is_dir():
            return False

        # Look for patient directories
        patient_dirs = [d for d in data_path.iterdir() if d.is_dir()]

        if not patient_dirs:
            return False

        # Check if at least one patient directory has the required files
        for patient_dir in patient_dirs[:3]:  # Check first 3 directories
            mr_file = patient_dir / "mr.mha"
            ct_file = patient_dir / "ct.mha"
            if mr_file.exists() and ct_file.exists():
                return True

        return False
    except Exception:
        return False


def setup_arg_parser() -> argparse.Namespace:
    """
    Sets up and parses the command-line args for the script."""
    parser = argparse.ArgumentParser(description='Enhanced model evaluation framework.')
    parser.add_argument('--model_name', type=str, required=True, choices=['unet', 'pix2pix', 'transformer_diffusion', 'shaoyanpan'],
                        help='Name of the model to test (e.g., "unet", "transformer_diffusion").')
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
    parser.add_argument('--max', type=int, default=None,
                        help='Maximum number of images/volumes to process (in natural order). If not specified, processes all images.')
    parser.add_argument('--diffusion_steps', type=int, default=2,
                        help='Number of diffusion steps for inference (default: 2, same as infer_single.py).')
    return parser.parse_args()

def run_evaluation_loop(model, data_loader: DataLoader, enhanced_metrics: bool = False, max: int = None) -> List[Dict[str, float]]:
    """
    Runs the inference and evaluation loop for all images in the data loader.

    Args:
        model: The initialised and loaded ModelAdapter object.
        data_loader (DataLoader): The PyTorch DataLoader for the test set.
        enhanced_metrics: Whether to use enhanced metrics including HU analysis.
        max: Maximum number of images to process. If None, processes all images.

    Returns:
        List[Dict[str, float]]: A list of dict where each dict contains the
            metrics for one test image.
    """
    results = []

    # Determine how many images to process
    total_images = len(data_loader)
    images_to_process = min(max, total_images) if max is not None else total_images

    print(f"Running inference on {images_to_process} test images{' (limited from ' + str(total_images) + ')' if max is not None and max < total_images else ''}...")

    processed_count = 0
    for mri_image, ground_truth_ct in data_loader:
        # Check if we've reached the limit
        if max is not None and processed_count >= max:
            break

        # Run model's predict method to get the generated CT scan.
        generated_ct = model.predict(mri_image)

        # Convert tensors to NumPy arrays for eval.
        # .squeeze() to remove extra dimensions.
        # Move to CPU first in case tensors are on GPU
        generated_np = generated_ct.squeeze().cpu().numpy()
        ground_truth_np = ground_truth_ct.squeeze().cpu().numpy()

        # Handle shape mismatches by resizing generated output to match ground truth
        if generated_np.shape != ground_truth_np.shape:
            print(f"Shape mismatch: generated {generated_np.shape} vs ground truth {ground_truth_np.shape}")
            print("Resizing generated output to match ground truth...")

            # Convert to tensors for interpolation
            generated_tensor = torch.from_numpy(generated_np).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            ground_truth_tensor = torch.from_numpy(ground_truth_np).unsqueeze(0).unsqueeze(0)

            # Resize generated to match ground truth
            generated_resized = torch.nn.functional.interpolate(
                generated_tensor,
                size=ground_truth_tensor.shape[2:],  # Match spatial dimensions
                mode='trilinear',
                align_corners=False
            )

            # Convert back to numpy
            generated_np = generated_resized.squeeze().numpy()
            print(f"Resized generated output shape: {generated_np.shape}")

        # Use appropriate metrics function
        if enhanced_metrics and ENHANCED_METRICS_AVAILABLE:
            metrics = calculate_enhanced_metrics(generated_np, ground_truth_np)
        else:
            metrics = calculate_metrics(generated_np, ground_truth_np)

        results.append(metrics)
        processed_count += 1

    print("Evaluation complete.")
    return results

def print_summary(model_name: str, results: List[Dict[str, float]], enhanced_metrics: bool = False):
    """ Calculates and prints the final performance summary with pass/fail criteria. """
    import json
    import os

    # Load evaluation thresholds
    thresholds_path = os.path.join(os.path.dirname(__file__), 'default_thresholds.json')
    try:
        with open(thresholds_path, 'r') as f:
            thresholds = json.load(f)
    except FileNotFoundError:
        # Default thresholds if file not found
        thresholds = {
            "ssim_min": 0.85,
            "psnr_min": 20.0,
            "mae_max": 0.1,
            "hu_mae_max": 60.0,
            "hu_correlation_min": 0.8,
            "hu_range_valid_min": 0.95
        }

    ssim_scores = [r['ssim'] for r in results]
    psnr_scores = [r['psnr'] for r in results]
    mae_scores = [r['mae'] for r in results]

    # Calculate mean values
    mean_ssim = np.mean(ssim_scores)
    mean_psnr = np.mean(psnr_scores)
    mean_mae = np.mean(mae_scores)

    # Determine pass/fail status
    ssim_pass = mean_ssim >= thresholds["ssim_min"]
    psnr_pass = mean_psnr >= thresholds["psnr_min"]
    mae_pass = mean_mae <= thresholds["mae_max"]

    # Overall pass/fail
    overall_pass = ssim_pass and psnr_pass and mae_pass

    print("\n" + "="*50)
    print("   PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Model: {model_name.upper()}")
    print(f"Number of test images: {len(results)}")
    print("-" * 50)

    # SSIM
    ssim_status = "✅ PASS" if ssim_pass else "❌ FAIL"
    print(f"SSIM: {mean_ssim:.3f} ± {np.std(ssim_scores):.3f} | Criteria: ≥{thresholds['ssim_min']:.2f} | {ssim_status}")

    # PSNR
    psnr_status = "✅ PASS" if psnr_pass else "❌ FAIL"
    print(f"PSNR: {mean_psnr:.2f} ± {np.std(psnr_scores):.2f} dB | Criteria: ≥{thresholds['psnr_min']:.1f} dB | {psnr_status}")

    # MAE
    mae_status = "✅ PASS" if mae_pass else "❌ FAIL"
    print(f"MAE:  {mean_mae:.3f} ± {np.std(mae_scores):.3f} | Criteria: ≤{thresholds['mae_max']:.2f} | {mae_status}")

    # Enhanced metrics if available
    if enhanced_metrics and ENHANCED_METRICS_AVAILABLE and 'hu_mae' in results[0]:
        hu_mae_scores = [r['hu_mae'] for r in results]
        hu_correlation_scores = [r['hu_correlation'] for r in results]
        mean_hu_mae = np.mean(hu_mae_scores)
        mean_hu_correlation = np.mean(hu_correlation_scores)

        hu_mae_pass = mean_hu_mae <= thresholds["hu_mae_max"]
        hu_correlation_pass = mean_hu_correlation >= thresholds["hu_correlation_min"]

        hu_mae_status = "✅ PASS" if hu_mae_pass else "❌ FAIL"
        hu_correlation_status = "✅ PASS" if hu_correlation_pass else "❌ FAIL"

        print(f"HU MAE: {mean_hu_mae:.1f} ± {np.std(hu_mae_scores):.1f} | Criteria: ≤{thresholds['hu_mae_max']:.1f} | {hu_mae_status}")
        print(f"HU Correlation: {mean_hu_correlation:.3f} ± {np.std(hu_correlation_scores):.3f} | Criteria: ≥{thresholds['hu_correlation_min']:.2f} | {hu_correlation_status}")

        # Update overall pass status
        overall_pass = overall_pass and hu_mae_pass and hu_correlation_pass

    print("-" * 50)
    overall_status = "✅ OVERALL PASS" if overall_pass else "❌ OVERALL FAIL"
    print(f"Overall Result: {overall_status}")
    print("="*50 + "\n")


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
        model = load_model(args.model_name, args.weights_path, args.diffusion_steps)

        # Load the test dataset.
        print(f"Loading data from: {args.data_folder}")
        test_dataset = create_dataset(args.data_folder)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

        # Run the main loop to get results.
        results = run_evaluation_loop(model, test_loader, args.enhanced_metrics, args.max)

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