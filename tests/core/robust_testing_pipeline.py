"""
Robust testing pipeline for MRI-to-CT synthesis models.

This module provides a comprehensive testing pipeline that:
- Simulates edge cases (resampling, noise, misalignment, intensity shifts)
- Evaluates models with enhanced metrics including HU analysis
- Implements pass/fail criteria for quality assessment
- Generates comprehensive reports and visualizations

API:
- Input: directory of MRI-CT pairs
- Output: MRI predictions + metrics (MAE, SSIM, PSNR, HU)
- Flag whether each sample passes criteria
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tests.core.edge_case_simulator import EdgeCaseSimulator, create_edge_case_test_suite
from tests.core.enhanced_evaluation import RobustEvaluator, calculate_enhanced_metrics
from tests.core.simple_loader import SimpleTestDataset
from models.adapters import load_model


class RobustTestDataset(SimpleTestDataset):
    """
    Extended dataset class that supports edge case simulation.
    """

    def __init__(self, data_folder: str, edge_case_simulator: Optional[EdgeCaseSimulator] = None,
                 apply_edge_cases: bool = False, edge_case_type: str = "all"):
        """
        Initialize robust test dataset.

        Args:
            data_folder: Path to directory containing test images
            edge_case_simulator: Simulator for edge cases
            apply_edge_cases: Whether to apply edge cases to MRI images
            edge_case_type: Type of edge case to apply
        """
        super().__init__(data_folder)
        self.edge_case_simulator = edge_case_simulator or EdgeCaseSimulator()
        self.apply_edge_cases = apply_edge_cases
        self.edge_case_type = edge_case_type

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get item with optional edge case simulation.

        Returns:
            Tuple of (mri_tensor, ct_tensor, edge_case_label)
        """
        mri_tensor, ct_tensor = super().__getitem__(idx)

        edge_case_label = "none"
        if self.apply_edge_cases:
            # Convert tensors to numpy for edge case simulation
            mri_np = mri_tensor.squeeze().numpy()
            ct_np = ct_tensor.squeeze().numpy()

            # Apply edge case simulation
            modified_mri_np, modified_ct_np = self.edge_case_simulator.simulate_edge_case(
                mri_np, ct_np, self.edge_case_type)

            # Convert back to tensors
            mri_tensor = torch.from_numpy(modified_mri_np).unsqueeze(0)
            ct_tensor = torch.from_numpy(modified_ct_np).unsqueeze(0)
            edge_case_label = self.edge_case_type

        return mri_tensor, ct_tensor, edge_case_label


class RobustTestingPipeline:
    """
    Main testing pipeline for comprehensive model evaluation.
    """

    def __init__(self, model_name: str, weights_path: str,
                 thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the testing pipeline.

        Args:
            model_name: Name of the model to test
            weights_path: Path to model weights
            thresholds: Custom thresholds for pass/fail criteria
        """
        self.model_name = model_name
        self.weights_path = weights_path
        self.model = None
        self.evaluator = RobustEvaluator()

        if thresholds:
            self.evaluator.thresholds.update(thresholds)

    def load_model(self):
        """Load the model and weights."""
        print(f"Loading {self.model_name} model from {self.weights_path}")
        self.model = load_model(self.model_name, self.weights_path)
        print("Model loaded successfully.")

    def run_baseline_evaluation(self, data_folder: str) -> Dict:
        """
        Run baseline evaluation without edge cases.

        Args:
            data_folder: Path to test data

        Returns:
            Evaluation results
        """
        print("\n" + "="*50)
        print("RUNNING BASELINE EVALUATION")
        print("="*50)

        # Load dataset
        dataset = RobustTestDataset(data_folder, apply_edge_cases=False)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        all_metrics = []
        edge_case_labels = []

        print(f"Evaluating {len(dataset)} baseline samples...")

        for mri_tensor, ct_tensor, edge_case_label in tqdm(dataloader):
            # Run inference
            predicted_ct = self.model.predict(mri_tensor)

            # Convert to numpy
            predicted_np = predicted_ct.squeeze().numpy()
            ground_truth_np = ct_tensor.squeeze().numpy()

            # Calculate metrics
            metrics = calculate_enhanced_metrics(predicted_np, ground_truth_np)
            all_metrics.append(metrics)
            edge_case_labels.append(edge_case_label)

        # Analyze results
        analysis = self.evaluator.analyze_dataset_performance(all_metrics, edge_case_labels)

        print("Baseline evaluation completed.")
        return analysis

    def run_edge_case_evaluation(self, data_folder: str,
                                edge_case_types: List[str] = None) -> Dict:
        """
        Run evaluation with edge case simulation.

        Args:
            data_folder: Path to test data
            edge_case_types: List of edge case types to test

        Returns:
            Evaluation results
        """
        if edge_case_types is None:
            edge_case_types = ["resampling", "noise", "misalignment", "intensity", "all"]

        print("\n" + "="*50)
        print("RUNNING EDGE CASE EVALUATION")
        print("="*50)

        all_metrics = []
        edge_case_labels = []

        for edge_case_type in edge_case_types:
            print(f"\nTesting edge case: {edge_case_type}")

            # Load dataset with edge case simulation
            dataset = RobustTestDataset(
                data_folder,
                apply_edge_cases=True,
                edge_case_type=edge_case_type
            )
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            case_metrics = []

            for mri_tensor, ct_tensor, edge_case_label in tqdm(dataloader,
                                                              desc=f"{edge_case_type}"):
                # Run inference
                predicted_ct = self.model.predict(mri_tensor)

                # Convert to numpy
                predicted_np = predicted_ct.squeeze().numpy()
                ground_truth_np = ct_tensor.squeeze().numpy()

                # Calculate metrics
                metrics = calculate_enhanced_metrics(predicted_np, ground_truth_np)
                case_metrics.append(metrics)
                edge_case_labels.append(edge_case_label)

            all_metrics.extend(case_metrics)
            print(f"Completed {edge_case_type}: {len(case_metrics)} samples")

        # Analyze results
        analysis = self.evaluator.analyze_dataset_performance(all_metrics, edge_case_labels)

        print("Edge case evaluation completed.")
        return analysis

    def run_comprehensive_evaluation(self, data_folder: str,
                                   edge_case_types: List[str] = None) -> Dict:
        """
        Run comprehensive evaluation including baseline and edge cases.

        Args:
            data_folder: Path to test data
            edge_case_types: List of edge case types to test

        Returns:
            Comprehensive evaluation results
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*60)

        # Run baseline evaluation
        baseline_results = self.run_baseline_evaluation(data_folder)

        # Run edge case evaluation
        edge_case_results = self.run_edge_case_evaluation(data_folder, edge_case_types)

        # Combine results
        comprehensive_results = {
            'model_name': self.model_name,
            'weights_path': self.weights_path,
            'baseline': baseline_results,
            'edge_cases': edge_case_results,
            'summary': self._generate_summary(baseline_results, edge_case_results)
        }

        return comprehensive_results

    def _generate_summary(self, baseline_results: Dict, edge_case_results: Dict) -> Dict:
        """Generate summary comparison between baseline and edge case performance."""
        summary = {}

        # Compare overall pass rates
        baseline_pass_rate = baseline_results['pass_fail_stats']['overall_pass']['pass_rate']
        edge_case_pass_rate = edge_case_results['pass_fail_stats']['overall_pass']['pass_rate']

        summary['pass_rate_comparison'] = {
            'baseline': baseline_pass_rate,
            'edge_cases': edge_case_pass_rate,
            'degradation': baseline_pass_rate - edge_case_pass_rate
        }

        # Compare key metrics
        key_metrics = ['ssim', 'psnr', 'mae', 'hu_mae']
        summary['metric_comparison'] = {}

        for metric in key_metrics:
            baseline_mean = baseline_results['overall_stats'][metric]['mean']
            edge_case_mean = edge_case_results['overall_stats'][metric]['mean']

            summary['metric_comparison'][metric] = {
                'baseline': baseline_mean,
                'edge_cases': edge_case_mean,
                'change': edge_case_mean - baseline_mean,
                'change_percent': ((edge_case_mean - baseline_mean) / baseline_mean) * 100
            }

        return summary

    def save_results(self, results: Dict, output_dir: str):
        """
        Save evaluation results to files.

        Args:
            results: Evaluation results
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        json_path = output_path / f"{self.model_name}_evaluation_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Generate and save report
        report_path = output_path / f"{self.model_name}_evaluation_report.txt"

        # Baseline report
        baseline_report = self.evaluator.generate_report(
            results['baseline'], f"{self.model_name}_baseline")

        # Edge case report
        edge_case_report = self.evaluator.generate_report(
            results['edge_cases'], f"{self.model_name}_edge_cases")

        # Combined report
        combined_report = f"{baseline_report}\n\n{edge_case_report}\n\n"
        combined_report += "SUMMARY COMPARISON:\n"
        combined_report += "-" * 30 + "\n"

        summary = results['summary']
        combined_report += f"Overall Pass Rate - Baseline: {summary['pass_rate_comparison']['baseline']:.1%}\n"
        combined_report += f"Overall Pass Rate - Edge Cases: {summary['pass_rate_comparison']['edge_cases']:.1%}\n"
        combined_report += f"Performance Degradation: {summary['pass_rate_comparison']['degradation']:.1%}\n\n"

        for metric, comparison in summary['metric_comparison'].items():
            combined_report += f"{metric.upper()}: {comparison['change_percent']:+.1f}% change\n"

        with open(report_path, 'w') as f:
            f.write(combined_report)

        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  Report: {report_path}")

    def create_visualizations(self, results: Dict, output_dir: str):
        """
        Create visualization plots for the results.

        Args:
            results: Evaluation results
            output_dir: Output directory
        """
        output_path = Path(output_dir)

        # Set style
        plt.style.use('seaborn-v0_8')

        # Create comparison plots
        self._plot_metric_comparison(results, output_path)
        self._plot_pass_rate_comparison(results, output_path)
        self._plot_edge_case_analysis(results, output_path)

        print(f"Visualizations saved to {output_path}")

    def _plot_metric_comparison(self, results: Dict, output_path: Path):
        """Plot metric comparison between baseline and edge cases."""
        summary = results['summary']
        metrics = list(summary['metric_comparison'].keys())

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            comparison = summary['metric_comparison'][metric]

            categories = ['Baseline', 'Edge Cases']
            values = [comparison['baseline'], comparison['edge_cases']]

            bars = axes[i].bar(categories, values, color=['skyblue', 'lightcoral'])
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())

            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                           f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_path / f"{self.model_name}_metric_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pass_rate_comparison(self, results: Dict, output_path: Path):
        """Plot pass rate comparison."""
        summary = results['summary']

        fig, ax = plt.subplots(figsize=(8, 6))

        categories = ['Baseline', 'Edge Cases']
        pass_rates = [
            summary['pass_rate_comparison']['baseline'],
            summary['pass_rate_comparison']['edge_cases']
        ]

        bars = ax.bar(categories, pass_rates, color=['skyblue', 'lightcoral'])
        ax.set_title('Overall Pass Rate Comparison')
        ax.set_ylabel('Pass Rate')
        ax.set_ylim(0, 1)

        # Add percentage labels
        for bar, rate in zip(bars, pass_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{rate:.1%}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_path / f"{self.model_name}_pass_rate_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_edge_case_analysis(self, results: Dict, output_path: Path):
        """Plot detailed edge case analysis."""
        edge_case_results = results['edge_cases']

        if 'edge_case_analysis' not in edge_case_results:
            return

        edge_case_data = edge_case_results['edge_case_analysis']

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Pass rates by edge case
        case_types = list(edge_case_data.keys())
        pass_rates = [edge_case_data[case]['pass_rates']['overall_pass'] for case in case_types]

        bars = axes[0, 0].bar(case_types, pass_rates, color='lightcoral')
        axes[0, 0].set_title('Pass Rate by Edge Case Type')
        axes[0, 0].set_ylabel('Pass Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)

        for bar, rate in zip(bars, pass_rates):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{rate:.1%}', ha='center', va='bottom')

        # SSIM by edge case
        ssim_values = [edge_case_data[case]['metrics']['ssim']['mean'] for case in case_types]
        axes[0, 1].bar(case_types, ssim_values, color='skyblue')
        axes[0, 1].set_title('SSIM by Edge Case Type')
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # PSNR by edge case
        psnr_values = [edge_case_data[case]['metrics']['psnr']['mean'] for case in case_types]
        axes[1, 0].bar(case_types, psnr_values, color='lightgreen')
        axes[1, 0].set_title('PSNR by Edge Case Type')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # HU MAE by edge case
        hu_mae_values = [edge_case_data[case]['metrics']['hu_mae']['mean'] for case in case_types]
        axes[1, 1].bar(case_types, hu_mae_values, color='orange')
        axes[1, 1].set_title('HU MAE by Edge Case Type')
        axes[1, 1].set_ylabel('HU MAE')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path / f"{self.model_name}_edge_case_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function for running the robust testing pipeline."""
    parser = argparse.ArgumentParser(description='Robust Testing Pipeline for MRI-to-CT Models')

    parser.add_argument('--model_name', type=str, required=True,
                       choices=['unet', 'pix2pix'],
                       help='Name of the model to test')
    parser.add_argument('--weights_path', type=str, required=True,
                       help='Path to the trained model weights file (.pth)')
    parser.add_argument('--data_folder', type=str, required=True,
                       help='Path to the folder containing the test dataset')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='Output directory for results and reports')
    parser.add_argument('--edge_case_types', nargs='+',
                       default=['resampling', 'noise', 'misalignment', 'intensity', 'all'],
                       help='Edge case types to test')
    parser.add_argument('--baseline_only', action='store_true',
                       help='Run only baseline evaluation (no edge cases)')
    parser.add_argument('--custom_thresholds', type=str, default=None,
                       help='Path to JSON file with custom thresholds')

    args = parser.parse_args()

    # Load custom thresholds if provided
    thresholds = None
    if args.custom_thresholds:
        with open(args.custom_thresholds, 'r') as f:
            thresholds = json.load(f)

    # Initialize pipeline
    pipeline = RobustTestingPipeline(
        model_name=args.model_name,
        weights_path=args.weights_path,
        thresholds=thresholds
    )

    # Load model
    pipeline.load_model()

    # Run evaluation
    if args.baseline_only:
        results = pipeline.run_baseline_evaluation(args.data_folder)
        results = {
            'model_name': args.model_name,
            'weights_path': args.weights_path,
            'baseline': results
        }
    else:
        results = pipeline.run_comprehensive_evaluation(
            args.data_folder, args.edge_case_types)

    # Save results and create visualizations
    pipeline.save_results(results, args.output_dir)
    pipeline.create_visualizations(results, args.output_dir)

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
