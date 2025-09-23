#!/usr/bin/env python3
"""
Resize all images in the sample dataset to 256x256
"""
import os
from PIL import Image

def resize_images_in_folder(folder_path, target_size=(256, 256)):
    """Resize all images in a folder to target size"""
    print(f"Resizing images in {folder_path} to {target_size}")
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(folder_path, filename)
            try:
                with Image.open(filepath) as img:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Resize
                    img = img.resize(target_size, Image.LANCZOS)
                    # Save
                    img.save(filepath)
                    print(f"  Resized {filename}")
            except Exception as e:
                print(f"  Error resizing {filename}: {e}")

def resize_sample_dataset(sample_dir="paired_sample", target_size=(256, 256)):
    """Resize all images in the sample dataset"""
    print(f"Resizing sample dataset to {target_size[0]}x{target_size[1]}")
    
    for subfolder in ['trainA', 'trainB', 'testA', 'testB']:
        folder_path = os.path.join(sample_dir, subfolder)
        if os.path.exists(folder_path):
            resize_images_in_folder(folder_path, target_size)
    
    print("All images resized successfully!")

if __name__ == "__main__":
    resize_sample_dataset()
