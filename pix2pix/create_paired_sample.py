#!/usr/bin/env python3
"""
Create a properly paired sample dataset for Pix2Pix training
"""
import os
import shutil
import random

def create_paired_sample(source_dir="MRI-CT-Dataset/images", target_dir="paired_sample", sample_size=20):
    """Create a properly paired sample dataset"""
    
    # Create target directories
    os.makedirs(f"{target_dir}/trainA", exist_ok=True)
    os.makedirs(f"{target_dir}/trainB", exist_ok=True)
    os.makedirs(f"{target_dir}/testA", exist_ok=True)
    os.makedirs(f"{target_dir}/testB", exist_ok=True)
    
    # Get all files and extract numbers
    trainA_files = [f for f in os.listdir(f"{source_dir}/trainA") if f.endswith(('.jpg', '.png'))]
    trainB_files = [f for f in os.listdir(f"{source_dir}/trainB") if f.endswith(('.jpg', '.png'))]
    testA_files = [f for f in os.listdir(f"{source_dir}/testA") if f.endswith(('.jpg', '.png'))]
    testB_files = [f for f in os.listdir(f"{source_dir}/testB") if f.endswith(('.jpg', '.png'))]
    
    print(f"Original dataset sizes:")
    print(f"  trainA: {len(trainA_files)}")
    print(f"  trainB: {len(trainB_files)}")
    print(f"  testA: {len(testA_files)}")
    print(f"  testB: {len(testB_files)}")
    
    def extract_number(filename):
        """Extract number from filename (e.g., 'ct123.png' -> 123)"""
        if filename.startswith('ct'):
            return int(filename[2:].split('.')[0])
        elif filename.startswith('mri'):
            return int(filename[3:].split('.')[0])
        return 0
    
    # Find paired files
    def find_pairs(files_a, files_b, prefix_a, prefix_b):
        pairs = []
        for file_a in files_a:
            num = extract_number(file_a)
            file_b = f"{prefix_b}{num}.jpg" if prefix_b == "mri" else f"{prefix_b}{num}.png"
            if file_b in files_b:
                pairs.append((file_a, file_b))
        return pairs
    
    # Find training pairs
    train_pairs = find_pairs(trainA_files, trainB_files, "ct", "mri")
    test_pairs = find_pairs(testA_files, testB_files, "ct", "mri")
    
    print(f"\nFound {len(train_pairs)} training pairs and {len(test_pairs)} test pairs")
    
    # Sample pairs
    random.seed(42)  # For reproducibility
    train_sample_size = min(sample_size, len(train_pairs))
    test_sample_size = min(sample_size, len(test_pairs))
    
    train_sample_pairs = random.sample(train_pairs, train_sample_size)
    test_sample_pairs = random.sample(test_pairs, test_sample_size)
    
    print(f"\nCreating sample with {train_sample_size} training pairs and {test_sample_size} test pairs")
    
    # Copy paired files
    for ct_file, mri_file in train_sample_pairs:
        shutil.copy2(f"{source_dir}/trainA/{ct_file}", f"{target_dir}/trainA/{ct_file}")
        shutil.copy2(f"{source_dir}/trainB/{mri_file}", f"{target_dir}/trainB/{mri_file}")
        print(f"  Paired: {ct_file} ↔ {mri_file}")
    
    for ct_file, mri_file in test_sample_pairs:
        shutil.copy2(f"{source_dir}/testA/{ct_file}", f"{target_dir}/testA/{ct_file}")
        shutil.copy2(f"{source_dir}/testB/{mri_file}", f"{target_dir}/testB/{mri_file}")
        print(f"  Paired: {ct_file} ↔ {mri_file}")
    
    print(f"\nPaired sample dataset created in '{target_dir}' folder")
    print(f"  trainA: {len(os.listdir(f'{target_dir}/trainA'))} files")
    print(f"  trainB: {len(os.listdir(f'{target_dir}/trainB'))} files")
    print(f"  testA: {len(os.listdir(f'{target_dir}/testA'))} files")
    print(f"  testB: {len(os.listdir(f'{target_dir}/testB'))} files")

if __name__ == "__main__":
    create_paired_sample(sample_size=20)
