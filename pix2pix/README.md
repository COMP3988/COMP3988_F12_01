# Pix2Pix Medical Image Synthesis for Radiotherapy Planning

## Project Overview

This project implements Pix2Pix models for converting MRI scans to synthetic CT images for radiotherapy planning. The goal is to reduce patient radiation exposure while maintaining the accuracy needed for clinical treatment planning.

**Dataset**: Brain CT-MRI pairs from Kaggle
**Application**: Radiotherapy planning and dose calculation
**Target**: High-quality synthetic CT images suitable for clinical use

## What We're Building

### The Problem
- Radiotherapy planning requires CT scans for dose calculation
- CT scans expose patients to ionizing radiation
- MRI provides better soft tissue contrast without radiation
- Solution: Use AI to generate synthetic CT from MRI

### The Solution
- Input: MRI brain scan (no radiation)
- Output: Synthetic CT brain scan (for radiotherapy planning)
- Method: Pix2Pix (proven image-to-image translation approach)

## How It Works

### Pix2Pix Architecture
The model uses two competing neural networks:

1. **Generator**: Creates synthetic CT from MRI
   - U-Net architecture with encoder-decoder structure
   - Learns to map MRI features to CT features
   - Preserves anatomical structures through skip connections

2. **Discriminator**: Judges if CT looks real or fake
   - CNN that analyzes MRI-CT pairs
   - Forces generator to create realistic images
   - Improves through adversarial training

### Training Process
1. **Forward Pass**: MRI → Generator → Synthetic CT → Discriminator
2. **Loss Calculation**: L1 loss (pixel accuracy) + Adversarial loss (realism)
3. **Backward Pass**: Update both networks to minimize errors
4. **Repeat**: Thousands of iterations until convergence

### Loss Functions
- **L1 Loss**: Measures pixel-by-pixel difference between real and synthetic CT
- **Adversarial Loss**: Measures how realistic the synthetic CT looks
- **Total Loss**: L1 + λ × Adversarial (λ = 100 for balance)

## Project Status

### Completed
- Development environment setup (Python, PyTorch, medical imaging libraries)
- Working Pix2Pix implementation for MRI to CT conversion
- Training pipeline with proper loss functions
- Results generation and visualization

### In Progress
- Model optimization and performance improvement
- Advanced loss functions implementation
- Medical-specific validation metrics

### Planned
- Attention mechanisms for better anatomical focus
- Perceptual loss for improved image quality
- Clinical validation with dosimetric accuracy
- Comparison with state-of-the-art methods

## Getting Started

### Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Train the model
python train.py
```

### Results
- Training progress saved in results/ folder
- Model weights saved as .pth files
- Visual comparisons generated each epoch

## Technical Details

### Model Architecture
- **Generator**: U-Net with 4 encoder and 4 decoder layers
- **Discriminator**: CNN with 5 convolutional layers
- **Input/Output**: 256x256 RGB images
- **Training**: Adam optimizer, batch size 2, 5 epochs for demo

### Performance Metrics
- **Image Quality**: PSNR, SSIM for pixel-level accuracy
- **Anatomical Accuracy**: Dice coefficient for structure preservation
- **Clinical Relevance**: Dosimetric accuracy for radiotherapy planning

## Development Plan

### Phase 1: Foundation (Completed)
- Set up working baseline model
- Establish training pipeline
- Generate initial results

### Phase 2: Advanced Improvements (Current)
- Implement attention mechanisms
- Add perceptual loss functions
- Optimize training strategies

### Phase 3: Medical Optimization (Next)
- Anatomical structure preservation
- Dosimetric accuracy validation
- Clinical relevance assessment

### Phase 4: Evaluation (Final)
- Comprehensive performance evaluation
- Comparison with existing methods
- Clinical validation results

## Research Goals

1. **Technical Innovation**: Implement novel improvements over existing methods
2. **Clinical Relevance**: Focus on radiotherapy planning applications
3. **Performance**: Achieve state-of-the-art results on medical image synthesis
4. **Documentation**: Comprehensive evaluation and comparison

## Key Features

- Medical image support for CT and MRI formats
- U-Net generator with skip connections
- Adversarial training for realistic results
- Multiple loss functions for balanced training
- Visual results generation for evaluation
- Modular code structure for easy modification

## Next Steps

1. Implement attention mechanisms for better anatomical focus
2. Add perceptual loss using pre-trained networks
3. Optimize for different anatomical regions
4. Validate with clinical dosimetric accuracy
5. Compare with state-of-the-art methods

## Notes

This project builds on proven Pix2Pix architecture while focusing on medical imaging applications. The goal is to create a system that can generate high-quality synthetic CT images from MRI scans, reducing patient radiation exposure while maintaining clinical accuracy for radiotherapy planning.

The implementation uses PyTorch and follows best practices for medical AI development, including proper data handling, model architecture, and evaluation metrics.