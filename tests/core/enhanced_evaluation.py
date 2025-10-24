"""
Enhanced evaluation metrics for MRI-to-CT synthesis models.

This module extends the basic evaluation metrics to include:
- HU (Hounsfield Unit) analysis for CT images
- Robust pass/fail criteria
- Statistical analysis of model performance
- Comprehensive reporting capabilities
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class HUAnalyzer:
    """
    Analyzes Hounsfield Unit (HU) values in CT images.
    """

    def __init__(self, hu_range: Tuple[float, float] = (-1000, 3000)):
        """
        Initialize HU analyzer.

        Args:
            hu_range: Expected HU range for CT images
        """
        self.hu_range = hu_range
        self.tissue_hu_ranges = {
            'air': (-1000, -500),
            'lung': (-500, -100),
            'fat': (-100, -50),
            'water': (-10, 10),
            'soft_tissue': (10, 50),
            'muscle': (50, 100),
            'bone': (100, 3000)
        }

    def denormalize_to_hu(self, normalized_image: np.ndarray,
                         normalization_range: Tuple[float, float] = (-1, 1)) -> np.ndarray:
        """
        Convert normalized image back to HU values.

        Args:
            normalized_image: Image normalized to [-1, 1] or [0, 1]
            normalization_range: Range used for normalization

        Returns:
            Image in HU units
        """
        if normalization_range == (-1, 1):
            # Convert from [-1, 1] to [0, 1] first
            normalized_image = (normalized_image + 1) / 2

        # Convert to HU range
        hu_min, hu_max = self.hu_range
        hu_image = normalized_image * (hu_max - hu_min) + hu_min

        return hu_image

    def analyze_hu_distribution(self, hu_image: np.ndarray) -> Dict[str, float]:
        """
        Analyze HU distribution in the image.

        Args:
            hu_image: Image in HU units

        Returns:
            Dictionary with HU statistics
        """
        stats_dict = {
            'mean_hu': np.mean(hu_image),
            'std_hu': np.std(hu_image),
            'min_hu': np.min(hu_image),
            'max_hu': np.max(hu_image),
            'median_hu': np.median(hu_image),
            'percentile_25': np.percentile(hu_image, 25),
            'percentile_75': np.percentile(hu_image, 75)
        }

        # Analyze tissue composition
        for tissue, (hu_min, hu_max) in self.tissue_hu_ranges.items():
            tissue_mask = (hu_image >= hu_min) & (hu_image <= hu_max)
            stats_dict[f'{tissue}_percentage'] = np.sum(tissue_mask) / hu_image.size * 100

        return stats_dict

    def calculate_hu_range_validity(self, predicted_hu: np.ndarray,
                                  ground_truth_hu: np.ndarray) -> Dict[str, float]:
        """
        Calculate HU range validity metrics.

        Args:
            predicted_hu: Predicted CT image in HU
            ground_truth_hu: Ground truth CT image in HU

        Returns:
            Dictionary with HU range validity metrics
        """
        # Define valid HU range (typical CT range)
        valid_hu_min, valid_hu_max = -1000, 3000

        # Check predicted HU range validity
        pred_valid_mask = (predicted_hu >= valid_hu_min) & (predicted_hu <= valid_hu_max)
        pred_valid_percentage = np.sum(pred_valid_mask) / predicted_hu.size

        # Check ground truth HU range validity
        gt_valid_mask = (ground_truth_hu >= valid_hu_min) & (ground_truth_hu <= valid_hu_max)
        gt_valid_percentage = np.sum(gt_valid_mask) / ground_truth_hu.size

        # Check where both are valid
        both_valid_mask = pred_valid_mask & gt_valid_mask
        both_valid_percentage = np.sum(both_valid_mask) / predicted_hu.size

        # Calculate HU range accuracy (only for valid voxels)
        if np.sum(both_valid_mask) > 0:
            valid_pred = predicted_hu[both_valid_mask]
            valid_gt = ground_truth_hu[both_valid_mask]
            hu_range_mae = np.mean(np.abs(valid_pred - valid_gt))
        else:
            hu_range_mae = float('inf')

        return {
            'pred_hu_range_valid': pred_valid_percentage,
            'gt_hu_range_valid': gt_valid_percentage,
            'both_hu_range_valid': both_valid_percentage,
            'hu_range_mae': hu_range_mae
        }

    def calculate_hu_accuracy(self, predicted_hu: np.ndarray,
                             ground_truth_hu: np.ndarray) -> Dict[str, float]:
        """
        Calculate HU-specific accuracy metrics.

        Args:
            predicted_hu: Predicted CT image in HU
            ground_truth_hu: Ground truth CT image in HU

        Returns:
            Dictionary with HU accuracy metrics
        """
        hu_diff = predicted_hu - ground_truth_hu

        metrics = {
            'hu_mae': np.mean(np.abs(hu_diff)),
            'hu_rmse': np.sqrt(np.mean(hu_diff**2)),
            'hu_mape': np.mean(np.abs(hu_diff / (ground_truth_hu + 1e-8))) * 100,
            'hu_correlation': np.corrcoef(predicted_hu.flatten(), ground_truth_hu.flatten())[0, 1]
        }

        # Tissue-specific accuracy
        for tissue, (hu_min, hu_max) in self.tissue_hu_ranges.items():
            tissue_mask = (ground_truth_hu >= hu_min) & (ground_truth_hu <= hu_max)
            if np.sum(tissue_mask) > 0:
                tissue_pred = predicted_hu[tissue_mask]
                tissue_gt = ground_truth_hu[tissue_mask]
                tissue_diff = tissue_pred - tissue_gt

                metrics[f'{tissue}_hu_mae'] = np.mean(np.abs(tissue_diff))
                metrics[f'{tissue}_hu_rmse'] = np.sqrt(np.mean(tissue_diff**2))

        return metrics


class RobustEvaluator:
    """
    Robust evaluation with pass/fail criteria and statistical analysis.
    """

    def __init__(self, hu_analyzer: Optional[HUAnalyzer] = None):
        """
        Initialize robust evaluator.

        Args:
            hu_analyzer: HU analyzer instance
        """
        self.hu_analyzer = hu_analyzer or HUAnalyzer()

        # Pass/fail criteria thresholds
        self.thresholds = {
            'ssim_min': 0.85,
            'psnr_min': 20.0,
            'mae_max': 0.1,
            'hu_mae_max': 60.0,
            'hu_correlation_min': 0.8,
            'hu_range_valid_min': 0.95
        }

    def calculate_comprehensive_metrics(self, predicted_image: np.ndarray,
                                     ground_truth_image: np.ndarray,
                                     normalization_range: Tuple[float, float] = (-1, 1)) -> Dict[str, float]:
        """
        Calculate comprehensive metrics including HU analysis.

        Args:
            predicted_image: Predicted image (normalized)
            ground_truth_image: Ground truth image (normalized)
            normalization_range: Range used for normalization

        Returns:
            Dictionary with all metrics
        """
        # Basic metrics
        metrics = {
            'ssim': ssim(ground_truth_image, predicted_image, data_range=1.0),
            'psnr': psnr(ground_truth_image, predicted_image, data_range=1.0),
            'mae': np.mean(np.abs(predicted_image - ground_truth_image))
        }

        # Convert to HU for analysis
        predicted_hu = self.hu_analyzer.denormalize_to_hu(predicted_image, normalization_range)
        ground_truth_hu = self.hu_analyzer.denormalize_to_hu(ground_truth_image, normalization_range)

        # HU-specific metrics
        hu_metrics = self.hu_analyzer.calculate_hu_accuracy(predicted_hu, ground_truth_hu)
        metrics.update(hu_metrics)

        # HU range validity metrics
        hu_range_metrics = self.hu_analyzer.calculate_hu_range_validity(predicted_hu, ground_truth_hu)
        metrics.update(hu_range_metrics)

        # HU distribution analysis
        hu_dist_pred = self.hu_analyzer.analyze_hu_distribution(predicted_hu)
        hu_dist_gt = self.hu_analyzer.analyze_hu_distribution(ground_truth_hu)

        # Add distribution metrics with prefixes
        for key, value in hu_dist_pred.items():
            metrics[f'pred_{key}'] = value
        for key, value in hu_dist_gt.items():
            metrics[f'gt_{key}'] = value

        return metrics

    def evaluate_pass_fail(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """
        Evaluate pass/fail criteria for a single sample.

        Args:
            metrics: Dictionary of calculated metrics

        Returns:
            Dictionary with pass/fail results for each criterion
        """
        results = {}

        # SSIM criterion
        results['ssim_pass'] = metrics['ssim'] >= self.thresholds['ssim_min']

        # PSNR criterion
        results['psnr_pass'] = metrics['psnr'] >= self.thresholds['psnr_min']

        # MAE criterion
        results['mae_pass'] = metrics['mae'] <= self.thresholds['mae_max']

        # HU MAE criterion
        results['hu_mae_pass'] = metrics['hu_mae'] <= self.thresholds['hu_mae_max']

        # HU correlation criterion
        results['hu_correlation_pass'] = metrics['hu_correlation'] >= self.thresholds['hu_correlation_min']

        # HU range validity criterion
        results['hu_range_valid_pass'] = metrics['both_hu_range_valid'] >= self.thresholds['hu_range_valid_min']

        # Overall pass (all criteria must pass)
        results['overall_pass'] = all([
            results['ssim_pass'],
            results['psnr_pass'],
            results['mae_pass'],
            results['hu_mae_pass'],
            results['hu_correlation_pass'],
            results['hu_range_valid_pass']
        ])

        return results

    def analyze_dataset_performance(self, all_metrics: List[Dict[str, float]],
                                  edge_case_labels: List[str] = None) -> Dict:
        """
        Analyze performance across the entire dataset.

        Args:
            all_metrics: List of metrics dictionaries for all samples
            edge_case_labels: List of edge case types for each sample

        Returns:
            Comprehensive analysis results
        """
        analysis = {}

        # Extract metric arrays
        metric_names = ['ssim', 'psnr', 'mae', 'hu_mae', 'hu_correlation']
        metric_arrays = {name: [m[name] for m in all_metrics] for name in metric_names}

        # Overall statistics
        analysis['overall_stats'] = {}
        for name, values in metric_arrays.items():
            analysis['overall_stats'][name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }

        # Pass/fail analysis
        pass_fail_results = [self.evaluate_pass_fail(metrics) for metrics in all_metrics]

        analysis['pass_fail_stats'] = {}
        for criterion in ['ssim_pass', 'psnr_pass', 'mae_pass', 'hu_mae_pass',
                         'hu_correlation_pass', 'overall_pass']:
            pass_rate = np.mean([r[criterion] for r in pass_fail_results])
            analysis['pass_fail_stats'][criterion] = {
                'pass_rate': pass_rate,
                'total_samples': len(pass_fail_results),
                'passed_samples': sum([r[criterion] for r in pass_fail_results])
            }

        # Edge case analysis
        if edge_case_labels:
            analysis['edge_case_analysis'] = {}
            unique_cases = list(set(edge_case_labels))

            for case_type in unique_cases:
                case_indices = [i for i, label in enumerate(edge_case_labels) if label == case_type]
                case_metrics = [all_metrics[i] for i in case_indices]
                case_pass_fail = [pass_fail_results[i] for i in case_indices]

                analysis['edge_case_analysis'][case_type] = {
                    'sample_count': len(case_indices),
                    'metrics': {name: {
                        'mean': np.mean([m[name] for m in case_metrics]),
                        'std': np.std([m[name] for m in case_metrics])
                    } for name in metric_names},
                    'pass_rates': {criterion: np.mean([r[criterion] for r in case_pass_fail])
                                  for criterion in ['ssim_pass', 'psnr_pass', 'mae_pass',
                                                   'hu_mae_pass', 'hu_correlation_pass', 'overall_pass']}
                }

        return analysis

    def generate_report(self, analysis: Dict, model_name: str,
                       output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report.

        Args:
            analysis: Analysis results from analyze_dataset_performance
            model_name: Name of the model being evaluated
            output_path: Optional path to save the report

        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append(f"COMPREHENSIVE MODEL EVALUATION REPORT")
        report_lines.append(f"Model: {model_name.upper()}")
        report_lines.append("=" * 60)

        # Overall performance
        report_lines.append("\nOVERALL PERFORMANCE:")
        report_lines.append("-" * 30)
        overall_stats = analysis['overall_stats']
        for metric, stats in overall_stats.items():
            report_lines.append(f"{metric.upper():15}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        # Pass/fail summary
        report_lines.append("\nPASS/FAIL SUMMARY:")
        report_lines.append("-" * 30)
        pass_fail_stats = analysis['pass_fail_stats']
        for criterion, stats in pass_fail_stats.items():
            criterion_name = criterion.replace('_pass', '').upper()
            report_lines.append(f"{criterion_name:15}: {stats['pass_rate']:.1%} ({stats['passed_samples']}/{stats['total_samples']})")

        # Edge case analysis
        if 'edge_case_analysis' in analysis:
            report_lines.append("\nEDGE CASE ANALYSIS:")
            report_lines.append("-" * 30)
            edge_case_analysis = analysis['edge_case_analysis']

            for case_type, case_data in edge_case_analysis.items():
                report_lines.append(f"\n{case_type.upper()} ({case_data['sample_count']} samples):")
                report_lines.append(f"  Overall Pass Rate: {case_data['pass_rates']['overall_pass']:.1%}")
                report_lines.append(f"  SSIM: {case_data['metrics']['ssim']['mean']:.3f} ± {case_data['metrics']['ssim']['std']:.3f}")
                report_lines.append(f"  PSNR: {case_data['metrics']['psnr']['mean']:.1f} ± {case_data['metrics']['psnr']['std']:.1f} dB")
                report_lines.append(f"  MAE:  {case_data['metrics']['mae']['mean']:.4f} ± {case_data['metrics']['mae']['std']:.4f}")
                report_lines.append(f"  HU MAE: {case_data['metrics']['hu_mae']['mean']:.1f} ± {case_data['metrics']['hu_mae']['std']:.1f}")

        report_lines.append("\n" + "=" * 60)

        report_text = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)

        return report_text


def calculate_enhanced_metrics(predicted_image: np.ndarray,
                            ground_truth_image: np.ndarray,
                            normalization_range: Tuple[float, float] = (-1, 1)) -> Dict[str, float]:
    """
    Calculate enhanced metrics including HU analysis.

    Args:
        predicted_image: Predicted image (normalized)
        ground_truth_image: Ground truth image (normalized)
        normalization_range: Range used for normalization

    Returns:
        Dictionary with comprehensive metrics
    """
    evaluator = RobustEvaluator()
    return evaluator.calculate_comprehensive_metrics(
        predicted_image, ground_truth_image, normalization_range)
