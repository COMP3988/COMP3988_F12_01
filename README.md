# Medical Image Synthesis for Radiotherapy Planning

## Project Overview

This project implements a **Pix2Pix conditional GAN** for converting MRI brain images to CT images for radiotherapy planning. The system uses deep learning to generate synthetic CT images from MRI inputs, reducing the need for additional radiation exposure in medical imaging.

**Key Features:**
- MRI-to-CT image conversion using Pix2Pix GAN
- Automated data preprocessing and pairing
- Comprehensive quality assessment with medical imaging metrics
- Fast inference (< 1 second per image)
- Reproducible training pipeline

## Quick Start Guide

### Prerequisites
- Python 3.9+
- PyTorch 2.4+
- MacBook Air M1 (or compatible system)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/COMP3988/COMP3988_F12_01
cd COMP3988_F12_01/pytorch-CycleGAN-and-pix2pix
```

2. **Install dependencies:**
```bash
pip install torch torchvision
pip install scikit-image pillow matplotlib numpy
```

### Running the Code

#### Step 1: Prepare Your Data
```bash
# Create paired MRI-CT dataset (20 samples)
python create_paired_sample.py
```

#### Step 2: Resize Images (if needed)
```bash
# Resize all images to 256x256
python resize_sample.py
```

#### Step 3: Train the Model
```bash
# Train Pix2Pix model for 30 epochs
python train.py --dataroot ./paired_sample --name mri_to_ct_30epochs --model pix2pix --direction AtoB --n_epochs 30 --batch_size 1
```

#### Step 4: Test the Model
```bash
# Generate CT images from MRI inputs
python test.py --dataroot ./paired_sample --name mri_to_ct_30epochs --model pix2pix --direction AtoB --epoch 30
```

#### Step 5: Analyze Results
```bash
# Calculate quality metrics and generate visualizations
python analyze_results.py
```

## Expected Results

After running the complete pipeline, you should see:

- **Training Progress**: Loss curves showing generator and discriminator training
- **Generated Images**: CT images created from MRI inputs in `results/mri_to_ct_30epochs/test_latest/images/`
- **Quality Metrics**: 
  - SSIM: ~0.301 ± 0.136
  - PSNR: ~10.38 ± 1.17 dB  
  - MAE: ~0.191 ± 0.050
- **Visualizations**: Training curves and quality metric distributions

## File Structure

```
pytorch-CycleGAN-and-pix2pix/
├── create_paired_sample.py    # Data preprocessing and pairing
├── resize_sample.py           # Image resizing utility
├── analyze_results.py         # Quality assessment and visualization
├── train.py                   # Model training script
├── test.py                    # Model testing script
├── paired_sample/             # Processed dataset
│   ├── trainA/               # CT images (training)
│   ├── trainB/               # MRI images (training)
│   ├── testA/                # CT images (testing)
│   └── testB/                # MRI images (testing)
└── results/                   # Generated outputs
    └── mri_to_ct_30epochs/
        └── test_latest/
            └── images/       # Generated CT images
```

## Model Architecture

- **Generator**: U-Net with 54.414M parameters
- **Discriminator**: PatchGAN with 2.769M parameters  
- **Input/Output**: 256×256×3 RGB images
- **Training**: 30 epochs, Adam optimizer (lr=0.0002)

## Troubleshooting

**Common Issues:**

1. **Memory Error**: Reduce batch size to 1 (already set as default)
2. **Dataset Not Found**: Ensure `paired_sample` folder exists with proper structure
3. **CUDA Error**: Code runs on CPU by default (MacBook Air M1 compatible)

**Getting Help:**
- Check the training output for error messages
- Ensure all dependencies are installed correctly
- Verify dataset structure matches expected format

## Project Context

This project was developed for **COMP3988 - Software Development Project** at the University of Sydney, in collaboration with PhD researcher Chengbo Wang. The system addresses the critical need for synthetic CT generation in radiotherapy planning while reducing patient radiation exposure.

## Technical Details

- **Framework**: PyTorch with pytorch-CycleGAN-and-pix2pix
- **Dataset**: MRI-CT Brain Dataset (20 paired samples)
- **Evaluation**: SSIM, PSNR, MAE metrics
- **Performance**: Sub-second inference time
- **Reproducibility**: Fixed random seeds (seed=42)

---

*For detailed technical documentation, see the group report and individual reports in the project repository.*

### Viewing .mha files

There are many tools to view .mha files

A simple lightweight one is `napari`

Install the cli

```bash
pip install "napari[all]"
```

Then to view

```bash
napari <mha-filename>
```
