#!/usr/bin/env python3
"""
Test script to verify SynthRAD setup for conditional DDPM
"""
import os
import sys
import torch
from datasets import ImageDataset

def test_dataset_structure():
    """Test if the dataset structure is correct"""
    print("Testing SynthRAD dataset setup...")

    # Check if synthrad_data directory exists
    dataset_path = "./synthrad_data"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset directory {dataset_path} not found!")
        print("You need to create the following structure:")
        print("synthrad_data/")
        print("├── train/")
        print("│   ├── a/  (place ct.mha files here)")
        print("│   └── b/  (place cbct.mha files here)")
        return False

    # Check for train/a and train/b directories
    train_a = os.path.join(dataset_path, "train", "a")
    train_b = os.path.join(dataset_path, "train", "b")

    if not os.path.exists(train_a):
        print(f"❌ Directory {train_a} not found!")
        return False

    if not os.path.exists(train_b):
        print(f"❌ Directory {train_b} not found!")
        return False

    # Check for .mha files
    mha_files_a = [f for f in os.listdir(train_a) if f.endswith('.mha')]
    mha_files_b = [f for f in os.listdir(train_b) if f.endswith('.mha')]

    if len(mha_files_a) == 0:
        print(f"❌ No .mha files found in {train_a}")
        return False

    if len(mha_files_b) == 0:
        print(f"❌ No .mha files found in {train_b}")
        return False

    print(f"✅ Found {len(mha_files_a)} files in train/a")
    print(f"✅ Found {len(mha_files_b)} files in train/b")

    return True

def test_dataset_loading():
    """Test if the dataset can be loaded"""
    try:
        print("\nTesting dataset loading...")
        dataset = ImageDataset("./synthrad_data", transforms_=False, unaligned=True)
        print(f"✅ Dataset loaded successfully with {len(dataset)} samples")

        # Test loading one sample
        sample = dataset[0]
        print(f"✅ Sample keys: {list(sample.keys())}")
        print(f"✅ Sample 'a' shape: {sample['a'].shape}")
        print(f"✅ Sample 'b' shape: {sample['b'].shape}")

        return True
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_training_imports():
    """Test if training script imports work"""
    try:
        print("\nTesting training script imports...")
        from Diffusion_condition import GaussianDiffusionTrainer_cond
        from Model_condition import UNet
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    print("SynthRAD Conditional DDPM Setup Test")
    print("=" * 40)

    success = True
    success &= test_dataset_structure()
    success &= test_training_imports()

    if success:
        success &= test_dataset_loading()

    print("\n" + "=" * 40)
    if success:
        print("✅ All tests passed! Ready to train.")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
