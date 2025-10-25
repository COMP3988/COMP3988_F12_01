"""
Standardized preprocessing functions for SynthRAD data.
This module ensures consistent normalization across training, inference, and testing.
"""

import numpy as np
import torch
import SimpleITK as sitk
from typing import Tuple, Union


def preprocess_mri(mr_array: np.ndarray) -> np.ndarray:
    """
    Preprocess MRI using percentile-based normalization to [-1, 1] range.
    This matches the exact preprocessing used in training.

    Args:
        mr_array: Raw MRI array (any dtype)

    Returns:
        Preprocessed MRI array in [-1, 1] range (float32)
    """
    mr = mr_array.astype(np.float32)

    # MRI robust scale → [-1,1] (matches preprocess_synthrad.py)
    p_lo, p_hi = np.percentile(mr, [0.5, 99.5])
    if p_hi <= p_lo:
        mr = np.zeros_like(mr, dtype=np.float32)
    else:
        mr = np.clip(mr, p_lo, p_hi)
        mr = 2.0 * (mr - p_lo) / (p_hi - p_lo) - 1.0

    return mr.astype(np.float32)


def preprocess_ct(ct_array: np.ndarray) -> np.ndarray:
    """
    Preprocess CT using HU range clipping to [-1, 1] range.
    This matches the exact preprocessing used in training.

    Args:
        ct_array: Raw CT array in HU units (any dtype)

    Returns:
        Preprocessed CT array in [-1, 1] range (float32)
    """
    ct = ct_array.astype(np.float32)

    # CT HU clip [-1000,3000] → [-1,1] (matches preprocess_synthrad.py)
    ct = np.clip(ct, -1000.0, 3000.0)
    ct = (ct + 1000.0) / 4000.0 * 2.0 - 1.0

    return ct.astype(np.float32)


def preprocess_pair(mr_array: np.ndarray, ct_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess MRI-CT pair using standardized normalization.

    Args:
        mr_array: Raw MRI array
        ct_array: Raw CT array in HU units

    Returns:
        Tuple of (preprocessed_mri, preprocessed_ct) both in [-1, 1] range
    """
    mr_preprocessed = preprocess_mri(mr_array)
    ct_preprocessed = preprocess_ct(ct_array)

    return mr_preprocessed, ct_preprocessed


def load_and_preprocess_mha(mr_path: str, ct_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load MRI and CT from MHA files and preprocess them.

    Args:
        mr_path: Path to MRI MHA file
        ct_path: Path to CT MHA file

    Returns:
        Tuple of (preprocessed_mri, preprocessed_ct, metadata)
        metadata contains spacing, origin, direction from the CT image
    """
    # Load images
    mr_img = sitk.ReadImage(mr_path)
    ct_img = sitk.ReadImage(ct_path)

    # Get arrays
    mr_array = sitk.GetArrayFromImage(mr_img)
    ct_array = sitk.GetArrayFromImage(ct_img)

    # Preprocess
    mr_preprocessed, ct_preprocessed = preprocess_pair(mr_array, ct_array)

    # Extract metadata from CT image (used for saving output)
    metadata = {
        'spacing': ct_img.GetSpacing(),
        'origin': ct_img.GetOrigin(),
        'direction': ct_img.GetDirection()
    }

    return mr_preprocessed, ct_preprocessed, metadata


def preprocess_tensor(tensor: torch.Tensor, is_ct: bool = False) -> torch.Tensor:
    """
    Preprocess PyTorch tensor using standardized normalization.

    Args:
        tensor: Input tensor
        is_ct: Whether this is CT data (uses HU normalization) or MRI data

    Returns:
        Preprocessed tensor in [-1, 1] range
    """
    tensor_np = tensor.numpy()

    if is_ct:
        tensor_preprocessed = preprocess_ct(tensor_np)
    else:
        tensor_preprocessed = preprocess_mri(tensor_np)

    return torch.from_numpy(tensor_preprocessed)


def postprocess_ct_output(ct_output: np.ndarray) -> np.ndarray:
    """
    Postprocess CT output from [-1, 1] back to HU units.
    This is the reverse of preprocess_ct.

    Args:
        ct_output: CT array in [-1, 1] range

    Returns:
        CT array in HU units (float32)
    """
    # Reverse: [-1, 1] -> [0, 1] -> [0, 4000] -> [-1000, 3000]
    ct_hu = (ct_output + 1.0) / 2.0 * 4000.0 - 1000.0

    # Clip to valid HU range
    ct_hu = np.clip(ct_hu, -1000.0, 3000.0)

    return ct_hu.astype(np.float32)


def postprocess_mri_output(mr_output: np.ndarray, original_range: Tuple[float, float]) -> np.ndarray:
    """
    Postprocess MRI output from [-1, 1] back to original scanner units.
    This is the reverse of preprocess_mri.

    Args:
        mr_output: MRI array in [-1, 1] range
        original_range: Original MRI range (min, max) from percentile normalization

    Returns:
        MRI array in original scanner units
    """
    mr_min, mr_max = original_range

    # Reverse: [-1, 1] -> [0, 1] -> [mr_min, mr_max]
    mr_original = (mr_output + 1.0) / 2.0 * (mr_max - mr_min) + mr_min

    return mr_original
