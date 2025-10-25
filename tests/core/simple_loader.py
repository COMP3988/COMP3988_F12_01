import os
import glob
from typing import Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shaoyanpan-transformer-diffusion'))
from preprocess_utils import preprocess_mri, preprocess_ct

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


class SynthRAD2025Dataset(Dataset):
    """
    A PyTorch Dataset for loading SynthRAD2025 Task1 data structure.

    Expected structure:
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

    Args:
        data_folder (str): The path to the directory containing SynthRAD2025 patient folders.
    """
    def __init__(self, data_folder: str):
        self.data_folder = Path(data_folder)

        # Find all patient directories
        self.patient_dirs = sorted([d for d in self.data_folder.iterdir()
                                   if d.is_dir() and self._has_required_files(d)])

        if not self.patient_dirs:
            raise FileNotFoundError(f"No valid SynthRAD2025 patient directories found in {data_folder}")

        print(f"Found {len(self.patient_dirs)} SynthRAD2025 patient directories in {data_folder}")

        # Verify all patients have required files
        for patient_dir in self.patient_dirs:
            if not self._has_required_files(patient_dir):
                print(f"Warning: {patient_dir.name} missing required files (mr.mha, ct.mha)")

    def _has_required_files(self, patient_dir: Path) -> bool:
        """Check if patient directory has required mr.mha and ct.mha files."""
        mr_file = patient_dir / "mr.mha"
        ct_file = patient_dir / "ct.mha"
        return mr_file.exists() and ct_file.exists()

    def __len__(self):
        """Returns the total number of patient directories."""
        return len(self.patient_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves MRI and CT volumes for a patient.

        Args:
            idx (int): The index of the patient to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the
            MRI tensor and CT tensor.
        """
        patient_dir = self.patient_dirs[idx]

        # Load MRI volume
        mr_path = patient_dir / "mr.mha"
        mr_image = sitk.ReadImage(str(mr_path))
        mr_array = sitk.GetArrayFromImage(mr_image).astype(np.float32)

        # Load CT volume
        ct_path = patient_dir / "ct.mha"
        ct_image = sitk.ReadImage(str(ct_path))
        ct_array = sitk.GetArrayFromImage(ct_image).astype(np.float32)

        # Convert to tensors
        mri_tensor = torch.from_numpy(mr_array)
        ct_tensor = torch.from_numpy(ct_array)

        # Normalize to [-1, 1] range
        mri_tensor = self._normalize_tensor(mri_tensor, is_ct=False)
        ct_tensor = self._normalize_tensor(ct_tensor, is_ct=True)

        # Add batch and channel dimensions if needed
        if len(mri_tensor.shape) == 3:  # [D, H, W]
            mri_tensor = mri_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
            ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)     # [1, 1, D, H, W]

        return mri_tensor, ct_tensor

    def _normalize_tensor(self, tensor: torch.Tensor, is_ct: bool = False) -> torch.Tensor:
        """Normalize tensor to [-1, 1] range using standardized preprocessing."""
        tensor_np = tensor.numpy()

        if is_ct:
            # Use CT preprocessing (HU range clipping)
            tensor_preprocessed = preprocess_ct(tensor_np)
        else:
            # Use MRI preprocessing (percentile-based normalization)
            tensor_preprocessed = preprocess_mri(tensor_np)

        return torch.from_numpy(tensor_preprocessed)


class SynthRADTask1Dataset(Dataset):
    """
    A PyTorch Dataset for loading SynthRAD2025 Task1 structure directly.

    Expected structure:
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

    Args:
        data_folder (str): The path to the Task1 directory containing section folders (AB, HN, TH).
    """
    def __init__(self, data_folder: str):
        self.data_folder = Path(data_folder)

        # Determine if we're pointing to Task1 directory or parent directory
        if self.data_folder.name == "Task1":
            task1_path = self.data_folder
        else:
            task1_path = self.data_folder / "Task1"
            if not task1_path.exists():
                raise FileNotFoundError(f"Task1 directory not found in {data_folder}")

        # Expected section folders
        self.sections = ['AB', 'HN', 'TH']
        self.patient_dirs = []

        # Collect all patient directories from all sections
        for section in self.sections:
            section_path = task1_path / section
            if section_path.exists() and section_path.is_dir():
                section_patients = sorted([d for d in section_path.iterdir()
                                         if d.is_dir() and self._has_required_files(d)])
                self.patient_dirs.extend(section_patients)
                print(f"Found {len(section_patients)} patients in {section} section")

        if not self.patient_dirs:
            raise FileNotFoundError(f"No valid SynthRAD2025 Task1 patient directories found in {data_folder}")

        print(f"Total: {len(self.patient_dirs)} SynthRAD2025 Task1 patients across all sections")

        # Verify all patients have required files
        for patient_dir in self.patient_dirs:
            if not self._has_required_files(patient_dir):
                print(f"Warning: {patient_dir.name} missing required files (mr.mha, ct.mha)")

    def _has_required_files(self, patient_dir: Path) -> bool:
        """Check if patient directory has required mr.mha and ct.mha files."""
        mr_file = patient_dir / "mr.mha"
        ct_file = patient_dir / "ct.mha"
        return mr_file.exists() and ct_file.exists()

    def __len__(self):
        """Returns the total number of patient directories across all sections."""
        return len(self.patient_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves MRI and CT volumes for a patient.

        Args:
            idx (int): The index of the patient to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the
            MRI tensor and CT tensor.
        """
        patient_dir = self.patient_dirs[idx]

        # Load MRI volume
        mr_path = patient_dir / "mr.mha"
        mr_image = sitk.ReadImage(str(mr_path))
        mr_array = sitk.GetArrayFromImage(mr_image).astype(np.float32)

        # Load CT volume
        ct_path = patient_dir / "ct.mha"
        ct_image = sitk.ReadImage(str(ct_path))
        ct_array = sitk.GetArrayFromImage(ct_image).astype(np.float32)

        # Convert to tensors
        mri_tensor = torch.from_numpy(mr_array)
        ct_tensor = torch.from_numpy(ct_array)

        # Normalize to [-1, 1] range
        mri_tensor = self._normalize_tensor(mri_tensor, is_ct=False)
        ct_tensor = self._normalize_tensor(ct_tensor, is_ct=True)

        # Add batch and channel dimensions if needed
        if len(mri_tensor.shape) == 3:  # [D, H, W]
            mri_tensor = mri_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
            ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)     # [1, 1, D, H, W]

        return mri_tensor, ct_tensor

    def _normalize_tensor(self, tensor: torch.Tensor, is_ct: bool = False) -> torch.Tensor:
        """Normalize tensor to [-1, 1] range using standardized preprocessing."""
        tensor_np = tensor.numpy()

        if is_ct:
            # Use CT preprocessing (HU range clipping)
            tensor_preprocessed = preprocess_ct(tensor_np)
        else:
            # Use MRI preprocessing (percentile-based normalization)
            tensor_preprocessed = preprocess_mri(tensor_np)

        return torch.from_numpy(tensor_preprocessed)


class SynthRADTestDataset(Dataset):
    """
    A PyTorch Dataset for loading SynthRAD .mha files for testing.

    Args:
        data_folder (str): The path to the directory containing the SynthRAD .mha files.
    """
    def __init__(self, data_folder: str):
        # Look for .mha files in the directory
        self.mha_files: List[str] = sorted(glob.glob(os.path.join(data_folder, '*.mha')))

        if not self.mha_files:
            raise FileNotFoundError(f"No .mha files found in {data_folder}")

        print(f"Found {len(self.mha_files)} .mha files in {data_folder}")

    def __len__(self):
        """Returns the total number of .mha files in the dataset."""
        return len(self.mha_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves one .mha file and extracts MRI and CT volumes.

        Args:
            idx (int): The index of the .mha file to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the
            MRI tensor and CT tensor.
        """
        mha_path = self.mha_files[idx]

        # Read the .mha file
        image = sitk.ReadImage(mha_path)
        image_array = sitk.GetArrayFromImage(image)

        # SynthRAD .mha files typically contain both MRI and CT in the same file
        # We need to split them - this assumes they are concatenated along one dimension
        # or stored as separate channels

        # For now, let's assume the file contains both volumes concatenated
        # This might need adjustment based on the actual SynthRAD format
        if len(image_array.shape) == 4:  # [C, D, H, W] or [D, H, W, C]
            if image_array.shape[0] == 2:  # Two channels
                mri_volume = image_array[0]  # First channel is MRI
                ct_volume = image_array[1]   # Second channel is CT
            else:
                # Assume concatenated along depth dimension
                mid_point = image_array.shape[0] // 2
                mri_volume = image_array[:mid_point]
                ct_volume = image_array[mid_point:]
        elif len(image_array.shape) == 3:  # [D, H, W]
            # Assume concatenated along depth dimension
            mid_point = image_array.shape[0] // 2
            mri_volume = image_array[:mid_point]
            ct_volume = image_array[mid_point:]
        else:
            raise ValueError(f"Unexpected image shape: {image_array.shape}")

        # Convert to tensors and normalize to [-1, 1]
        mri_tensor = torch.from_numpy(mri_volume.astype(np.float32))
        ct_tensor = torch.from_numpy(ct_volume.astype(np.float32))

        # Normalize to [-1, 1] range
        mri_tensor = (mri_tensor - mri_tensor.min()) / (mri_tensor.max() - mri_tensor.min()) * 2 - 1
        ct_tensor = (ct_tensor - ct_tensor.min()) / (ct_tensor.max() - ct_tensor.min()) * 2 - 1

        # Add channel dimension if needed
        if len(mri_tensor.shape) == 3:
            mri_tensor = mri_tensor.unsqueeze(0)  # [1, D, H, W]
            ct_tensor = ct_tensor.unsqueeze(0)    # [1, D, H, W]

        return mri_tensor, ct_tensor




class NPZTestDataset(Dataset):
    """
    A PyTorch Dataset for loading SynthRAD .npz files for testing.

    Args:
        data_folder (str): The path to the directory containing the SynthRAD .npz files.
    """
    def __init__(self, data_folder: str):
        # Look for .npz files in the directory
        self.npz_files: List[str] = sorted(glob.glob(os.path.join(data_folder, '*.npz')))

        if not self.npz_files:
            raise FileNotFoundError(f"No .npz files found in {data_folder}")

        print(f"Found {len(self.npz_files)} .npz files in {data_folder}")

    def __len__(self):
        """Returns the total number of .npz files in the dataset."""
        return len(self.npz_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves one .npz file and extracts MRI and CT volumes.

        Args:
            idx (int): The index of the .npz file to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the
            MRI tensor and CT tensor.
        """
        npz_path = self.npz_files[idx]

        # Load the .npz file
        data = np.load(npz_path)

        # Extract MRI and CT volumes
        mri_volume = data['image'].astype(np.float32)  # MRI volume
        ct_volume = data['label'].astype(np.float32)    # CT volume

        # Convert to tensors
        mri_tensor = torch.from_numpy(mri_volume)
        ct_tensor = torch.from_numpy(ct_volume)

        # Add batch and channel dimensions if needed
        if len(mri_tensor.shape) == 3:  # [D, H, W]
            mri_tensor = mri_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
            ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)    # [1, 1, D, H, W]
        elif len(mri_tensor.shape) == 4:  # [B, D, H, W] or [C, D, H, W]
            if mri_tensor.shape[0] == 1:  # Already has batch dimension
                mri_tensor = mri_tensor.unsqueeze(1)  # [1, 1, D, H, W]
                ct_tensor = ct_tensor.unsqueeze(1)    # [1, 1, D, H, W]
            else:  # Has channel dimension
                mri_tensor = mri_tensor.unsqueeze(0)  # [1, C, D, H, W]
                ct_tensor = ct_tensor.unsqueeze(0)    # [1, C, D, H, W]

        return mri_tensor, ct_tensor