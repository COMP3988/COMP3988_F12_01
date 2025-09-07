#!/usr/bin/env python3
"""
Create a small balanced sample dataset for quick testing
"""
import os
import shutil
import random

def create_small_sample(source_dir="MRI-CT-Dataset/images", target_dir="small_sample", sample_size=10):
    """Create a small balanced sample dataset"""
    
    # Create target directories
    os.makedirs(f"{target_dir}/trainA", exist_ok=True)
    os.makedirs(f"{target_dir}/trainB", exist_ok=True)
    os.makedirs(f"{target_dir}/testA", exist_ok=True)
    os.makedirs(f"{target_dir}/testB", exist_ok=True)
    
    # Get all files
    trainA_files = [f for f in os.listdir(f"{source_dir}/trainA") if f.endswith(('.jpg', '.png'))]
    trainB_files = [f for f in os.listdir(f"{source_dir}/trainB") if f.endswith(('.jpg', '.png'))]
    testA_files = [f for f in os.listdir(f"{source_dir}/testA") if f.endswith(('.jpg', '.png'))]
    testB_files = [f for f in os.listdir(f"{source_dir}/testB") if f.endswith(('.jpg', '.png'))]
    
    print(f"Original dataset sizes:")
    print(f"  trainA: {len(trainA_files)}")
    print(f"  trainB: {len(trainB_files)}")
    print(f"  testA: {len(testA_files)}")
    print(f"  testB: {len(testB_files)}")
    
    # Use the minimum size to ensure balance
    min_train = min(len(trainA_files), len(trainB_files))
    min_test = min(len(testA_files), len(testB_files))
    
    # Sample files
    train_sample_size = min(sample_size, min_train)
    test_sample_size = min(sample_size, min_test)
    
    # Ensure we don't exceed available files
    train_sample_size = min(train_sample_size, len(trainA_files), len(trainB_files))
    test_sample_size = min(test_sample_size, len(testA_files), len(testB_files))
    
    print(f"\nCreating sample with {train_sample_size} training pairs and {test_sample_size} test pairs")
    
    # Randomly sample files
    random.seed(42)  # For reproducibility
    trainA_sample = random.sample(trainA_files, train_sample_size)
    trainB_sample = random.sample(trainB_files, train_sample_size)
    testA_sample = random.sample(testA_files, test_sample_size)
    testB_sample = random.sample(testB_files, test_sample_size)
    
    # Copy files
    for file in trainA_sample:
        shutil.copy2(f"{source_dir}/trainA/{file}", f"{target_dir}/trainA/{file}")
    for file in trainB_sample:
        shutil.copy2(f"{source_dir}/trainB/{file}", f"{target_dir}/trainB/{file}")
    for file in testA_sample:
        shutil.copy2(f"{source_dir}/testA/{file}", f"{target_dir}/testA/{file}")
    for file in testB_sample:
        shutil.copy2(f"{source_dir}/testB/{file}", f"{target_dir}/testB/{file}")
    
    print(f"\nSample dataset created in '{target_dir}' folder")
    print(f"  trainA: {len(os.listdir(f'{target_dir}/trainA'))} files")
    print(f"  trainB: {len(os.listdir(f'{target_dir}/trainB'))} files")
    print(f"  testA: {len(os.listdir(f'{target_dir}/testA'))} files")
    print(f"  testB: {len(os.listdir(f'{target_dir}/testB'))} files")

if __name__ == "__main__":
    create_small_sample(sample_size=20)
