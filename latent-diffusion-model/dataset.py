#dataset.py

import os
import glob
from typing import Tuple

import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SynthRADDataset(Dataset):
    
    def __init__(self, root_dir: str, image_size: int = 128):
        self.root_dir = root_dir
        self.image_size = image_size
        
        patient_folders = sorted(glob.glob(os.path.join(self.root_dir, '*')))
        
        self.file_pairs = []
        for patient_folder in patient_folders:
            mr_path = os.path.join(patient_folder, 'mr.mha')
            ct_path = os.path.join(patient_folder, 'ct.mha')
            
            if os.path.exists(mr_path) and os.path.exists(ct_path):
                reader = sitk.ImageFileReader()
                reader.SetFileName(mr_path)
                reader.ReadImageInformation()
                num_slices = reader.GetSize()[2]
                
                for i in range(num_slices):
                    self.file_pairs.append((mr_path, ct_path, i))
                    
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
    def __len__(self) -> int:
        return len(self.file_pairs)
    
    def __getitem__ (self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mr_path, ct_path, slice_idx = self.file_pairs[idx]
        
        mr_vol = sitk.GetArrayFromImage(sitk.ReadImage(mr_path, sitk.sitkFloat32))
        ct_vol = sitk.GetArrayFromImage(sitk.ReadImage(ct_path, sitk.sitkFloat32))
        
        mr_slice = mr_vol[slice_idx, :, :]
        ct_slice = ct_vol[slice_idx, :, :]
        
        mr_slice = (mr_slice - np.min(mr_slice)) / (np.max(mr_slice) - np.min(mr_slice) + 1e-8)
        ct_slice = (ct_slice - np.min(ct_slice)) / (np.max(ct_slice) - np.min(ct_slice) + 1e-8)
        
        mr_tensor = self.transform(mr_slice)
        ct_tensor = self.transform(ct_slice)
        
        return mr_tensor, ct_tensor