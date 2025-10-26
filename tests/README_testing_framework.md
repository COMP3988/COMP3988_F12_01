# Testing Framework Usage Examples

This document provides examples of how to use the comprehensive testing framework for MRI-to-CT synthesis models.

## File Structure

```
tests/
├── test_framework.py         # Main entry point for testing
├── test_adapters.py         # Unit tests for model adapters
├── core/                    # Core testing framework modules
│   ├── __init__.py         # Package initialization
│   ├── edge_case_simulator.py    # Edge case simulation
│   ├── enhanced_evaluation.py    # Advanced metrics & HU analysis
│   ├── robust_testing_pipeline.py # Comprehensive testing pipeline
│   ├── simple_loader.py    # Data loading utilities
│   └── evaluation.py       # Basic evaluation metrics
├── default_thresholds.json # Default pass/fail thresholds
├── README_testing_framework.md # Usage documentation
└── README_input_output.md  # Input/output documentation
```

## Basic Evaluation

Run basic evaluation with standard metrics (SSIM, PSNR, MAE):

```bash
python tests/test_framework.py \
  --model_name unet \
  --weights_path ./checkpoints/unet_model.pth \
  --data_folder ./test_data \
  --output_dir ./evaluation_results
```

## Enhanced Evaluation with HU Analysis

Run evaluation with enhanced metrics including HU (Hounsfield Unit) analysis:

```bash
python tests/test_framework.py \
  --model_name unet \
  --weights_path ./checkpoints/unet_model.pth \
  --data_folder ./test_data \
  --enhanced_metrics \
  --output_dir ./evaluation_results
```

## Robust Testing with Edge Case Simulation

Run comprehensive testing with edge case simulation:

```bash
python tests/test_framework.py \
  --model_name unet \
  --weights_path ./checkpoints/unet_model.pth \
  --data_folder ./test_data \
  --robust_testing \
  --output_dir ./evaluation_results
```

## Custom Edge Case Types

Test specific edge case types only:

```bash
python tests/test_framework.py \
  --model_name unet \
  --weights_path ./checkpoints/unet_model.pth \
  --data_folder ./test_data \
  --robust_testing \
  --edge_case_types resampling noise misalignment \
  --output_dir ./evaluation_results
```

## Custom Pass/Fail Thresholds

Use custom thresholds for pass/fail criteria:

```bash
python tests/test_framework.py \
  --model_name unet \
  --weights_path ./checkpoints/unet_model.pth \
  --data_folder ./test_data \
  --robust_testing \
  --custom_thresholds ./tests/custom_thresholds.json \
  --output_dir ./evaluation_results
```

## Direct Robust Testing Pipeline

Use the robust testing pipeline directly:

```bash
python tests/core/robust_testing_pipeline.py \
  --model_name unet \
  --weights_path ./checkpoints/unet_model.pth \
  --data_folder ./test_data \
  --output_dir ./test_results \
  --edge_case_types resampling noise misalignment intensity all
```

## Expected Outputs

### Basic Evaluation
- JSON file with metrics for each sample
- Console output with summary statistics

### Robust Testing
- Comprehensive JSON results file
- Detailed text report
- Visualization plots (metric comparisons, pass rates, edge case analysis)
- Pass/fail analysis for each sample and edge case type

### File Structure
```
evaluation_results/
├── unet_evaluation_results.json          # Complete results
├── unet_evaluation_report.txt            # Human-readable report
├── unet_metric_comparison.png            # Metric comparison plots
├── unet_pass_rate_comparison.png         # Pass rate visualization
└── unet_edge_case_analysis.png          # Edge case analysis plots
```

## Edge Case Types

1. **resampling**: 20% downsampling and upsampling artifacts
2. **noise**: Gaussian noise, salt-and-pepper noise, motion artifacts
3. **misalignment**: Translation and rotation between MRI and CT
4. **intensity**: Global and local intensity variations
5. **all**: Combination of all edge case types

## Metrics

### Basic Metrics
- **SSIM**: Structural Similarity Index
- **PSNR**: Peak Signal-to-Noise Ratio
- **MAE**: Mean Absolute Error

### Enhanced Metrics (with --enhanced_metrics)
- **HU MAE**: Mean Absolute Error in Hounsfield Units
- **HU RMSE**: Root Mean Square Error in HU
- **HU Correlation**: Correlation between predicted and ground truth HU values
- **Tissue-specific metrics**: Accuracy for different tissue types (air, lung, fat, water, soft tissue, muscle, bone)

## Pass/Fail Criteria

Default thresholds:
- SSIM ≥ 0.85
- PSNR ≥ 20.0 dB
- MAE ≤ 0.1
- HU MAE ≤ 60.0
- HU Correlation ≥ 0.8
- HU Range Validity ≥ 95% (voxels within valid HU range)

A sample passes if it meets ALL criteria.
