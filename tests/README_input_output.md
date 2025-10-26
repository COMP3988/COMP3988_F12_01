# Testing Framework - Input/Output Documentation

This document describes the expected input format, data structure, and output formats for the comprehensive MRI-to-CT synthesis testing framework.

## 📁 Input Requirements

### Data Directory Structure

The testing framework expects a specific directory structure containing paired MRI-CT images:

```
test_data/
├── patient001_real_A.png    # MRI image
├── patient001_real_B.png    # Corresponding CT image
├── patient002_real_A.png    # MRI image
├── patient002_real_B.png    # Corresponding CT image
├── patient003_real_A.png    # MRI image
├── patient003_real_B.png    # Corresponding CT image
└── ...
```

### File Naming Convention

- **MRI files**: `{patient_id}_real_A.png`
- **CT files**: `{patient_id}_real_B.png`
- **Patient ID**: Can be any string (numbers, letters, underscores)
- **File format**: PNG images (grayscale)

### Image Requirements

#### Format Specifications
- **File format**: PNG (grayscale)
- **Color channels**: Single channel (grayscale)
- **Data type**: 8-bit or 16-bit unsigned integer
- **Pixel values**: Typically 0-255 (will be normalized to [-1, 1] internally)

#### Size Requirements
- **Minimum size**: 64x64 pixels
- **Recommended size**: 256x256 pixels or larger
- **Aspect ratio**: Square images preferred (width = height)
- **Consistency**: All images should have the same dimensions

#### Content Requirements
- **MRI images**: Should contain anatomical structures
- **CT images**: Should be corresponding CT scans of the same anatomy
- **Alignment**: MRI and CT pairs should be reasonably aligned
- **Quality**: Images should be free of severe artifacts for baseline testing

### Example Input Validation

```python
# Valid file pairs
patient001_real_A.png  ←→  patient001_real_B.png
patient002_real_A.png  ←→  patient002_real_B.png
patient003_real_A.png  ←→  patient003_real_B.png

# Invalid examples
patient001_real_A.png  ←→  patient002_real_B.png  # Mismatched patient IDs
patient001_mri.png     ←→  patient001_ct.png      # Wrong naming convention
patient001_real_A.jpg  ←→  patient001_real_B.png  # Mixed file formats
```

## 🔧 Model Requirements

### Supported Models
- **UNet**: U-Net architecture for image-to-image translation
- **Pix2Pix**: Conditional GAN for paired image translation

### Model Weight Files
- **Format**: `.pth` files (PyTorch state dictionaries)
- **Content**: Trained model weights
- **Compatibility**: Must be compatible with the model adapter system

### Model Input/Output Specifications
- **Input**: Single-channel MRI image tensor `(1, 1, H, W)`
- **Output**: Single-channel CT image tensor `(1, 1, H, W)`
- **Normalization**: Images normalized to [-1, 1] range
- **Device**: Supports both CPU and GPU inference

## 📊 Output Formats

### 1. JSON Results File

**File**: `{model_name}_evaluation_results.json`

```json
{
  "model_name": "unet",
  "weights_path": "./checkpoints/unet_model.pth",
  "baseline": {
    "overall_stats": {
      "ssim": {
        "mean": 0.850,
        "std": 0.050,
        "min": 0.720,
        "max": 0.920,
        "median": 0.855
      },
      "psnr": {
        "mean": 25.3,
        "std": 2.1,
        "min": 20.1,
        "max": 28.9,
        "median": 25.5
      },
      "mae": {
        "mean": 0.080,
        "std": 0.020,
        "min": 0.045,
        "max": 0.125,
        "median": 0.078
      },
      "hu_mae": {
        "mean": 45.2,
        "std": 12.1,
        "min": 28.5,
        "max": 67.8,
        "median": 43.1
      },
      "hu_correlation": {
        "mean": 0.892,
        "std": 0.045,
        "min": 0.801,
        "max": 0.945,
        "median": 0.895
      }
    },
    "pass_fail_stats": {
      "ssim_pass": {
        "pass_rate": 0.92,
        "total_samples": 25,
        "passed_samples": 23
      },
      "psnr_pass": {
        "pass_rate": 0.88,
        "total_samples": 25,
        "passed_samples": 22
      },
      "mae_pass": {
        "pass_rate": 0.96,
        "total_samples": 25,
        "passed_samples": 24
      },
      "hu_mae_pass": {
        "pass_rate": 0.84,
        "total_samples": 25,
        "passed_samples": 21
      },
      "hu_correlation_pass": {
        "pass_rate": 0.88,
        "total_samples": 25,
        "passed_samples": 22
      },
      "hu_range_valid_pass": {
        "pass_rate": 0.92,
        "total_samples": 25,
        "passed_samples": 23
      },
      "overall_pass": {
        "pass_rate": 0.76,
        "total_samples": 25,
        "passed_samples": 19
      }
    }
  },
  "edge_cases": {
    "overall_stats": {
      "ssim": {
        "mean": 0.720,
        "std": 0.080,
        "min": 0.580,
        "max": 0.850,
        "median": 0.725
      }
    },
    "edge_case_analysis": {
      "resampling": {
        "sample_count": 5,
        "metrics": {
          "ssim": {
            "mean": 0.780,
            "std": 0.060
          },
          "psnr": {
            "mean": 22.1,
            "std": 2.3
          },
          "mae": {
            "mean": 0.095,
            "std": 0.025
          },
          "hu_mae": {
            "mean": 52.3,
            "std": 15.2
          },
          "hu_correlation": {
            "mean": 0.845,
            "std": 0.055
          }
        },
        "pass_rates": {
          "ssim_pass": 0.80,
          "psnr_pass": 0.60,
          "mae_pass": 0.80,
          "hu_mae_pass": 0.60,
          "hu_correlation_pass": 0.60,
          "hu_range_valid_pass": 0.80,
          "overall_pass": 0.40
        }
      },
      "noise": {
        "sample_count": 5,
        "metrics": {
          "ssim": {
            "mean": 0.650,
            "std": 0.080
          }
        },
        "pass_rates": {
          "overall_pass": 0.20
        }
      },
      "misalignment": {
        "sample_count": 5,
        "metrics": {
          "ssim": {
            "mean": 0.720,
            "std": 0.070
          }
        },
        "pass_rates": {
          "overall_pass": 0.40
        }
      },
      "intensity": {
        "sample_count": 5,
        "metrics": {
          "ssim": {
            "mean": 0.750,
            "std": 0.065
          }
        },
        "pass_rates": {
          "overall_pass": 0.60
        }
      },
      "all": {
        "sample_count": 5,
        "metrics": {
          "ssim": {
            "mean": 0.580,
            "std": 0.090
          }
        },
        "pass_rates": {
          "overall_pass": 0.00
        }
      }
    }
  },
  "summary": {
    "pass_rate_comparison": {
      "baseline": 0.80,
      "edge_cases": 0.32,
      "degradation": 0.48
    },
    "metric_comparison": {
      "ssim": {
        "baseline": 0.850,
        "edge_cases": 0.720,
        "change": -0.130,
        "change_percent": -15.3
      },
      "psnr": {
        "baseline": 25.3,
        "edge_cases": 21.8,
        "change": -3.5,
        "change_percent": -13.8
      },
      "mae": {
        "baseline": 0.080,
        "edge_cases": 0.105,
        "change": 0.025,
        "change_percent": 31.3
      },
      "hu_mae": {
        "baseline": 45.2,
        "edge_cases": 58.7,
        "change": 13.5,
        "change_percent": 29.9
      }
    }
  }
}
```

### 2. Text Report File

**File**: `{model_name}_evaluation_report.txt`

```
================================================================================
COMPREHENSIVE MODEL EVALUATION REPORT
Model: UNET
================================================================================

OVERALL PERFORMANCE:
------------------------------
SSIM: 0.850 ± 0.050
PSNR: 25.3 ± 2.1 dB
MAE: 0.080 ± 0.020
HU MAE: 45.2 ± 12.1
HU CORRELATION: 0.892 ± 0.045

PASS/FAIL SUMMARY:
------------------------------
SSIM: 92% (23/25)
PSNR: 88% (22/25)
MAE: 96% (24/25)
HU MAE: 84% (21/25)
HU CORRELATION: 88% (22/25)
HU RANGE VALID: 92% (23/25)
OVERALL: 76% (19/25)

EDGE CASE ANALYSIS:
------------------------------

RESAMPLING (5 samples):
  Overall Pass Rate: 40%
  SSIM: 0.780 ± 0.060
  PSNR: 22.1 ± 2.3 dB
  MAE: 0.095 ± 0.025
  HU MAE: 52.3 ± 15.2
  HU CORRELATION: 0.845 ± 0.055

NOISE (5 samples):
  Overall Pass Rate: 20%
  SSIM: 0.650 ± 0.080
  PSNR: 19.8 ± 2.8 dB
  MAE: 0.125 ± 0.035
  HU MAE: 68.2 ± 18.5
  HU CORRELATION: 0.720 ± 0.085

MISALIGNMENT (5 samples):
  Overall Pass Rate: 40%
  SSIM: 0.720 ± 0.070
  PSNR: 21.2 ± 2.5 dB
  MAE: 0.110 ± 0.030
  HU MAE: 61.5 ± 16.8
  HU CORRELATION: 0.780 ± 0.070

INTENSITY (5 samples):
  Overall Pass Rate: 60%
  SSIM: 0.750 ± 0.065
  PSNR: 22.8 ± 2.2 dB
  MAE: 0.095 ± 0.025
  HU MAE: 55.1 ± 14.3
  HU CORRELATION: 0.820 ± 0.060

ALL (5 samples):
  Overall Pass Rate: 0%
  SSIM: 0.580 ± 0.090
  PSNR: 17.5 ± 3.2 dB
  MAE: 0.155 ± 0.045
  HU MAE: 78.9 ± 22.1
  HU CORRELATION: 0.650 ± 0.095

SUMMARY COMPARISON:
------------------------------
Overall Pass Rate - Baseline: 76.0%
Overall Pass Rate - Edge Cases: 32.0%
Performance Degradation: 44.0%

SSIM: -15.3% change
PSNR: -13.8% change
MAE: +31.3% change
HU MAE: +29.9% change

================================================================================
```

### 3. Visualization Files

#### Metric Comparison Plot
**File**: `{model_name}_metric_comparison.png`
- **Content**: Bar charts comparing baseline vs edge case performance
- **Metrics**: SSIM, PSNR, MAE, HU MAE
- **Format**: High-resolution PNG (300 DPI)

#### Pass Rate Comparison Plot
**File**: `{model_name}_pass_rate_comparison.png`
- **Content**: Bar chart showing overall pass rates
- **Comparison**: Baseline vs Edge Cases
- **Format**: High-resolution PNG (300 DPI)

#### Edge Case Analysis Plot
**File**: `{model_name}_edge_case_analysis.png`
- **Content**: Multiple subplots analyzing each edge case type
- **Metrics**: Pass rates, SSIM, PSNR, HU MAE by edge case
- **Format**: High-resolution PNG (300 DPI)

### 4. Basic Evaluation Output (Non-Robust Mode)

**File**: `{model_name}_basic_evaluation.json`

```json
{
  "model_name": "unet",
  "weights_path": "./checkpoints/unet_model.pth",
  "data_folder": "./test_data",
  "enhanced_metrics": true,
  "results": [
    {
      "ssim": 0.850,
      "psnr": 25.3,
      "mae": 0.080,
      "hu_mae": 45.2,
      "hu_correlation": 0.892,
      "pred_hu_range_valid": 0.95,
      "gt_hu_range_valid": 0.98,
      "both_hu_range_valid": 0.94,
      "pred_mean_hu": 125.3,
      "gt_mean_hu": 128.7
    }
  ],
  "summary": {
    "ssim_mean": 0.850,
    "ssim_std": 0.050,
    "psnr_mean": 25.3,
    "psnr_std": 2.1,
    "mae_mean": 0.080,
    "mae_std": 0.020,
    "hu_mae_mean": 45.2,
    "hu_mae_std": 12.1,
    "hu_correlation_mean": 0.892,
    "hu_correlation_std": 0.045,
    "both_hu_range_valid_mean": 0.94,
    "both_hu_range_valid_std": 0.03
  }
}
```

## 📋 Output Directory Structure

```
evaluation_results/
├── unet_evaluation_results.json          # Complete results (robust mode)
├── unet_evaluation_report.txt            # Human-readable report
├── unet_metric_comparison.png            # Metric comparison plots
├── unet_pass_rate_comparison.png         # Pass rate visualization
├── unet_edge_case_analysis.png          # Edge case analysis plots
└── unet_basic_evaluation.json           # Basic results (non-robust mode)
```

## 📁 Testing Framework Structure

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

## 🔍 Metric Interpretations

### Basic Metrics
- **SSIM (0-1)**: Higher is better. Values >0.7 considered good
- **PSNR (dB)**: Higher is better. Values >20 dB considered acceptable
- **MAE (0-1)**: Lower is better. Values <0.1 considered good

### Enhanced Metrics (HU Analysis)
- **HU MAE**: Mean absolute error in Hounsfield Units. Lower is better
- **HU Correlation**: Correlation between predicted and ground truth HU values (0-1)
- **Tissue-specific metrics**: Accuracy for different tissue types

### Pass/Fail Criteria
- **Overall Pass**: Sample must pass ALL individual criteria
- **Pass Rate**: Percentage of samples that pass overall criteria
- **Performance Degradation**: Difference between baseline and edge case pass rates

### Updated Pass/Fail Criteria (New Requirements)
- **SSIM ≥ 0.85**: Higher structural similarity requirement
- **PSNR ≥ 20.0 dB**: Maintained signal quality requirement
- **MAE ≤ 0.1**: Maintained pixel accuracy requirement
- **HU MAE ≤ 60.0**: Increased HU accuracy tolerance (was 50.0)
- **HU Range Validity ≥ 95%**: New requirement for voxels within valid HU range (-1000 to 3000 HU)

## ⚠️ Common Issues and Solutions

### Input Issues
1. **Missing file pairs**: Ensure every MRI has a corresponding CT
2. **Wrong naming**: Use `_real_A.png` for MRI, `_real_B.png` for CT
3. **Size mismatch**: All images should have consistent dimensions
4. **Format issues**: Use PNG format for all images

### Output Issues
1. **Empty results**: Check that model weights are compatible
2. **Low pass rates**: Consider adjusting thresholds in `default_thresholds.json`
3. **Missing visualizations**: Ensure matplotlib/seaborn are installed

### Performance Issues
1. **Slow inference**: Use GPU if available (`--device cuda:0`)
2. **Memory errors**: Reduce batch size or image resolution
3. **Import errors**: Ensure all dependencies are installed
