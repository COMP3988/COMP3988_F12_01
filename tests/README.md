# MRI-to-CT Synthesis Testing Framework

A comprehensive testing framework for evaluating MRI-to-CT synthesis models with support for multiple data formats, robust testing, and detailed analysis.

## 🚀 Quick Start

### Basic Evaluation
```bash
python tests/test_framework.py \
  --model_name transformer_diffusion \
  --weights_path ./synthRAD_checkpoints/Synth_A_to_B.pth \
  --data_folder ./test_data \
  --output_dir ./evaluation_results
```

### Enhanced Evaluation with HU Analysis
```bash
python tests/test_framework.py \
  --model_name transformer_diffusion \
  --weights_path ./synthRAD_checkpoints/Synth_A_to_B.pth \
  --data_folder ./test_data \
  --enhanced_metrics \
  --output_dir ./evaluation_results
```

### Robust Testing with Edge Cases
```bash
python tests/test_framework.py \
  --model_name transformer_diffusion \
  --weights_path ./synthRAD_checkpoints/Synth_A_to_B.pth \
  --data_folder ./test_data \
  --robust_testing \
  --output_dir ./evaluation_results
```

## 📁 Supported Data Formats

The framework automatically detects and handles multiple data formats:

### 1. SynthRAD2025 Task1 Format (Recommended)
```
data_folder/Task1/
├── AB/                    # Abdomen section
│   ├── 1ABA005/
│   │   ├── mr.mha
│   │   ├── ct.mha
│   │   └── mask.mha
│   └── ...
├── HN/                    # Head & Neck section
│   ├── 1HNA001/
│   │   ├── mr.mha
│   │   ├── ct.mha
│   │   └── mask.mha
│   └── ...
└── TH/                    # Thorax section
    ├── 1THA001/
    │   ├── mr.mha
    │   ├── ct.mha
    │   └── mask.mha
    └── ...
```

### 2. SynthRAD2025 Section Format
```
data_folder/
├── 1ABA005/
│   ├── mr.mha    # MRI volume
│   ├── ct.mha    # CT volume
│   └── mask.mha  # Mask (optional)
├── 1ABA009/
│   ├── mr.mha
│   ├── ct.mha
│   └── mask.mha
└── ...
```

### 3. NPZ Format (Training Compatible)
```
data_folder/
├── patient001.npz    # Contains 'image' (MRI) and 'label' (CT) keys
├── patient002.npz
└── ...
```

### 4. PNG Format (Simple Testing)
```
data_folder/
├── patient001_real_A.png    # MRI image
├── patient001_real_B.png    # CT image
├── patient002_real_A.png
├── patient002_real_B.png
└── ...
```

### 5. MHA Format (Legacy)
```
data_folder/
├── patient001.mha    # Combined MRI+CT volume
├── patient002.mha
└── ...
```

## 🔧 Model Support

### Supported Models
- **transformer_diffusion**: Transformer-based diffusion model
- **unet**: U-Net architecture
- **pix2pix**: Conditional GAN
- **shaoyanpan**: Custom model variant

### Model Requirements
- **Format**: `.pth` files (PyTorch state dictionaries)
- **Input**: MRI tensor `(1, 1, D, H, W)` normalized to [-1, 1]
- **Output**: CT tensor `(1, 1, D, H, W)` normalized to [-1, 1]
- **Device**: Supports CPU and GPU inference

## 📊 Evaluation Modes

### Basic Evaluation
- **Metrics**: SSIM, PSNR, MAE
- **Output**: JSON results + console summary with pass/fail criteria
- **Use case**: Quick performance assessment
- **Pass/Fail**: Shows criteria thresholds and overall result status

### Enhanced Evaluation (`--enhanced_metrics`)
- **Additional Metrics**: HU MAE, HU RMSE, HU Correlation
- **Tissue Analysis**: Air, lung, fat, water, soft tissue, muscle, bone
- **Use case**: Clinical validation

### Robust Testing (`--robust_testing`)
- **Edge Cases**: Resampling, noise, misalignment, intensity variations
- **Output**: Comprehensive analysis + visualizations
- **Use case**: Model robustness assessment

## 🎯 Edge Case Types

1. **resampling**: 20% downsampling/upsampling artifacts
2. **noise**: Gaussian, salt-and-pepper, motion artifacts
3. **misalignment**: Translation and rotation between MRI/CT
4. **intensity**: Global and local intensity variations
5. **all**: Combination of all edge case types

## 📈 Metrics & Thresholds

### Basic Metrics
- **SSIM (0-1)**: Structural similarity (≥0.85)
- **PSNR (dB)**: Peak signal-to-noise ratio (≥20.0)
- **MAE (0-1)**: Mean absolute error (≤0.1)

### Enhanced Metrics
- **HU MAE**: Mean absolute error in Hounsfield Units (≤60.0)
- **HU Correlation**: Correlation between predicted/ground truth HU (≥0.8)
- **HU Range Validity**: Voxels within valid HU range -1000 to 3000 (≥95%)

### Pass/Fail Criteria
A sample passes if it meets **ALL** criteria. Overall pass rate is the percentage of samples that pass.

## 📋 Output Files

### Basic Evaluation
```
evaluation_results/
└── {model_name}_basic_evaluation.json
```

**Console Output Example:**
```
==================================================
   PERFORMANCE SUMMARY
==================================================
Model: TRANSFORMER_DIFFUSION
Number of test images: 1
--------------------------------------------------
SSIM: 0.753 ± 0.000 | Criteria: ≥0.85 | ❌ FAIL
PSNR: 18.15 ± 0.00 dB | Criteria: ≥20.0 dB | ❌ FAIL
MAE:  0.060 ± 0.000 | Criteria: ≤0.10 | ✅ PASS
--------------------------------------------------
Overall Result: ❌ OVERALL FAIL
==================================================
```

### Robust Testing
```
evaluation_results/
├── {model_name}_evaluation_results.json    # Complete results
├── {model_name}_evaluation_report.txt       # Human-readable report
├── {model_name}_metric_comparison.png       # Metric comparison plots
├── {model_name}_pass_rate_comparison.png    # Pass rate visualization
└── {model_name}_edge_case_analysis.png     # Edge case analysis plots
```

## ⚙️ Advanced Options

### Custom Edge Cases
```bash
--edge_case_types resampling noise misalignment intensity all
```

### Custom Thresholds
```bash
--custom_thresholds ./tests/custom_thresholds.json
```

### Limit Processing
```bash
--max 10  # Process only first 10 samples
```

### Diffusion Steps (Transformer-Diffusion)
```bash
--diffusion_steps 20  # Use 20 diffusion steps instead of default 2
```

## 🔍 Example Outputs

### JSON Results Structure
```json
{
  "model_name": "transformer_diffusion",
  "baseline": {
    "overall_stats": {
      "ssim": {"mean": 0.850, "std": 0.050},
      "psnr": {"mean": 25.3, "std": 2.1},
      "mae": {"mean": 0.080, "std": 0.020}
    },
    "pass_fail_stats": {
      "overall_pass": {"pass_rate": 0.76, "total_samples": 25}
    }
  },
  "edge_cases": {
    "edge_case_analysis": {
      "resampling": {"overall_pass": 0.40},
      "noise": {"overall_pass": 0.20},
      "misalignment": {"overall_pass": 0.40}
    }
  }
}
```

### Text Report Sample
```
================================================================================
COMPREHENSIVE MODEL EVALUATION REPORT
Model: TRANSFORMER_DIFFUSION
================================================================================

OVERALL PERFORMANCE:
------------------------------
SSIM: 0.850 ± 0.050
PSNR: 25.3 ± 2.1 dB
MAE: 0.080 ± 0.020
HU MAE: 45.2 ± 12.1

PASS/FAIL SUMMARY:
------------------------------
OVERALL: 76% (19/25)

EDGE CASE ANALYSIS:
------------------------------
RESAMPLING: Overall Pass Rate: 40%
NOISE: Overall Pass Rate: 20%
MISALIGNMENT: Overall Pass Rate: 40%
```

## 🛠️ Framework Structure

```
tests/
├── test_framework.py              # Main entry point
├── test_adapters.py              # Unit tests
├── core/                         # Core modules
│   ├── simple_loader.py         # Data loading (supports all formats)
│   ├── evaluation.py            # Basic metrics
│   ├── enhanced_evaluation.py   # HU analysis
│   ├── robust_testing_pipeline.py # Edge case testing
│   └── edge_case_simulator.py  # Edge case generation
├── default_thresholds.json      # Pass/fail criteria
├── dummy_test_data/             # Sample test data
└── README.md                    # This file
```

## ⚠️ Troubleshooting

### Common Issues
1. **No supported files found**: Check data folder contains supported formats
2. **Model loading errors**: Verify `.pth` file compatibility
3. **Memory errors**: Use `--max` to limit samples or reduce batch size
4. **Import errors**: Ensure all dependencies installed (`torch`, `SimpleITK`, `matplotlib`)

### Performance Tips
1. **GPU acceleration**: Use CUDA if available
2. **Batch processing**: Adjust batch size for memory constraints
3. **Sample limiting**: Use `--max` for quick testing
4. **Format selection**: SynthRAD2025 format is most efficient

## 📚 Usage Examples

### SynthRAD2025 Task1 Data Testing (All Sections)
```bash
python tests/test_framework.py \
  --model_name transformer_diffusion \
  --weights_path ./synthRAD_checkpoints/Synth_A_to_B.pth \
  --data_folder /path/to/synthRAD2025_Task1_Train \
  --enhanced_metrics \
  --output_dir ./results
```

### SynthRAD2025 Section Data Testing (Single Section)
```bash
python tests/test_framework.py \
  --model_name transformer_diffusion \
  --weights_path ./synthRAD_checkpoints/Synth_A_to_B.pth \
  --data_folder /path/to/synthRAD2025_Task1_Train/Task1/AB \
  --enhanced_metrics \
  --output_dir ./results
```

### Quick Sanity Check
```bash
python tests/test_framework.py \
  --model_name transformer_diffusion \
  --weights_path ./model.pth \
  --data_folder ./test_data \
  --max 5 \
  --output_dir ./quick_test
```

### Comprehensive Robust Testing
```bash
python tests/test_framework.py \
  --model_name transformer_diffusion \
  --weights_path ./model.pth \
  --data_folder ./test_data \
  --robust_testing \
  --enhanced_metrics \
  --edge_case_types resampling noise misalignment intensity \
  --output_dir ./robust_test
```

---

**Note**: The framework automatically detects data format and selects appropriate dataset class. No manual format conversion required!
