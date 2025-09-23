import os
import glob
from typing import Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SimpleTestDataset(Dataset):
    """ 
    A simple PyTorch Dataset for loading paired MRI and CT images for testing.
    
    Args:
        data_folder (str): The path to the directory containing the test images.
    """
    def __init__(self, data_folder: str):
        self.mri_paths: List[str] = sorted(glob.glob(os.path.join(data_folder, '*_real_A.png')))
        self.ct_paths: List[str] = sorted(glob.glob(os.path.join(data_folder, '*_real_B.png')))
        
        # Define a standard transformation pipeline for the images.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __len__(self):
        """Returns the total number of image pairs in the dataset."""
        return len(self.mri_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ 
        Retrieves one pair of MRI and CT images from the dataset.
        
        Args:
            idx(int): This index of the image pair to retrieve. 
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the 
            transformed MRI tensor and the transformed CT tensor.
        """
        # Get file paths for the image pair at the given idx.
        mri_path = self.mri_paths[idx]
        ct_path = self.ct_paths[idx]
        
        #Open the images using the PIL library and convert them to grayscale.
        mri_image = Image.open(mri_path).convert('L')
        ct_image = Image.open(ct_path).convert('L')
        
        #Apply the defined transformati9on pipeline to both images.
        mri_tensor = self.transform(mri_image)
        ct_tensor = self.transform(ct_image)
        
        return mri_tensor, ct_tensor