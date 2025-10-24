"""
Edge case simulation module for testing MRI-to-CT synthesis models.

This module provides functionality to simulate various edge cases and artifacts
that can occur in real-world medical imaging scenarios, including:
- Resampling artifacts (20% downsampling)
- Noise and artifacts
- Misalignment between MRI and CT
- Intensity shifts and variations

Used for robust testing of model performance under challenging conditions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, rotate, zoom
from scipy.ndimage import uniform_filter
from typing import Tuple, Dict, Optional
import random


class EdgeCaseSimulator:
    """
    Simulates various edge cases and artifacts for robust model testing.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the edge case simulator.

        Args:
            seed: Random seed for reproducible results
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)

    def simulate_resampling(self, image: np.ndarray, downsample_factor: float = 0.8) -> np.ndarray:
        """
        Simulate resampling artifacts by downsampling and upsampling.

        Args:
            image: Input image array
            downsample_factor: Factor by which to downsample (0.8 = 20% reduction)

        Returns:
            Image with resampling artifacts
        """
        if len(image.shape) == 2:
            # 2D image
            h, w = image.shape
            new_h, new_w = int(h * downsample_factor), int(w * downsample_factor)

            # Downsample
            downsampled = zoom(image, (new_h/h, new_w/w), order=1)

            # Upsample back to original size
            resampled = zoom(downsampled, (h/new_h, w/new_w), order=1)

        elif len(image.shape) == 3:
            # 3D volume
            d, h, w = image.shape
            new_d = int(d * downsample_factor)
            new_h = int(h * downsample_factor)
            new_w = int(w * downsample_factor)

            # Downsample
            downsampled = zoom(image, (new_d/d, new_h/h, new_w/w), order=1)

            # Upsample back to original size
            resampled = zoom(downsampled, (d/new_d, h/new_h, w/new_w), order=1)

        return resampled

    def add_noise_and_artifacts(self, image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """
        Add various types of noise and artifacts to simulate real-world conditions.

        Args:
            image: Input image array
            noise_level: Intensity of noise to add (0.0 to 1.0)

        Returns:
            Image with noise and artifacts
        """
        noisy_image = image.copy()

        # Add Gaussian noise
        gaussian_noise = np.random.normal(0, noise_level * 0.1, image.shape)
        noisy_image += gaussian_noise

        # Add salt and pepper noise
        salt_pepper_prob = noise_level * 0.05
        salt_mask = np.random.random(image.shape) < salt_pepper_prob
        pepper_mask = np.random.random(image.shape) < salt_pepper_prob
        noisy_image[salt_mask] = 1.0
        noisy_image[pepper_mask] = 0.0

        # Add motion artifacts (blur)
        if len(image.shape) == 2:
            motion_blur = gaussian_filter(noisy_image, sigma=noise_level * 2)
            noisy_image = 0.7 * noisy_image + 0.3 * motion_blur
        elif len(image.shape) == 3:
            motion_blur = gaussian_filter(noisy_image, sigma=(0, noise_level * 2, noise_level * 2))
            noisy_image = 0.7 * noisy_image + 0.3 * motion_blur

        # Add streak artifacts (simulate CT artifacts)
        if len(image.shape) == 2:
            h, w = image.shape
            streak_intensity = noise_level * 0.3
            for _ in range(int(noise_level * 5)):
                start_x = np.random.randint(0, w)
                start_y = np.random.randint(0, h)
                end_x = np.random.randint(0, w)
                end_y = np.random.randint(0, h)

                # Create streak line
                y_coords, x_coords = np.linspace(start_y, end_y, max(abs(end_y-start_y), 1)), \
                                   np.linspace(start_x, end_x, max(abs(end_x-start_x), 1))
                for y, x in zip(y_coords, x_coords):
                    if 0 <= int(y) < h and 0 <= int(x) < w:
                        noisy_image[int(y), int(x)] += streak_intensity

        return np.clip(noisy_image, 0, 1)

    def simulate_misalignment(self, mri: np.ndarray, ct: np.ndarray,
                             max_shift: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate misalignment between MRI and CT images.

        Args:
            mri: MRI image array
            ct: CT image array
            max_shift: Maximum shift as fraction of image size

        Returns:
            Tuple of (misaligned_mri, original_ct)
        """
        misaligned_mri = mri.copy()

        if len(mri.shape) == 2:
            h, w = mri.shape
            shift_h = int(np.random.uniform(-max_shift, max_shift) * h)
            shift_w = int(np.random.uniform(-max_shift, max_shift) * w)

            # Apply translation
            if shift_h != 0 or shift_w != 0:
                misaligned_mri = np.roll(misaligned_mri, (shift_h, shift_w), axis=(0, 1))

                # Add rotation
                rotation_angle = np.random.uniform(-5, 5)  # degrees
                misaligned_mri = rotate(misaligned_mri, rotation_angle, reshape=False, order=1)

        elif len(mri.shape) == 3:
            d, h, w = mri.shape
            shift_d = int(np.random.uniform(-max_shift, max_shift) * d)
            shift_h = int(np.random.uniform(-max_shift, max_shift) * h)
            shift_w = int(np.random.uniform(-max_shift, max_shift) * w)

            # Apply translation
            if shift_d != 0 or shift_h != 0 or shift_w != 0:
                misaligned_mri = np.roll(misaligned_mri, (shift_d, shift_h, shift_w), axis=(0, 1, 2))

        return misaligned_mri, ct

    def simulate_intensity_shifts(self, image: np.ndarray,
                                intensity_variation: float = 0.2) -> np.ndarray:
        """
        Simulate intensity shifts and variations.

        Args:
            image: Input image array
            intensity_variation: Amount of intensity variation (0.0 to 1.0)

        Returns:
            Image with intensity shifts
        """
        shifted_image = image.copy()

        # Global intensity shift
        global_shift = np.random.uniform(-intensity_variation, intensity_variation)
        shifted_image += global_shift

        # Local intensity variations
        if len(image.shape) == 2:
            # Create smooth intensity field
            h, w = image.shape
            y, x = np.mgrid[0:h, 0:w]

            # Random smooth field
            field = np.sin(x * np.random.uniform(0.01, 0.05)) * \
                   np.cos(y * np.random.uniform(0.01, 0.05)) * intensity_variation * 0.5
            shifted_image += field

        elif len(image.shape) == 3:
            # 3D intensity field
            d, h, w = image.shape
            z, y, x = np.mgrid[0:d, 0:h, 0:w]

            field = np.sin(x * np.random.uniform(0.01, 0.05)) * \
                   np.cos(y * np.random.uniform(0.01, 0.05)) * \
                   np.sin(z * np.random.uniform(0.01, 0.05)) * intensity_variation * 0.3
            shifted_image += field

        # Add contrast variation
        contrast_factor = np.random.uniform(0.8, 1.2)
        shifted_image = (shifted_image - 0.5) * contrast_factor + 0.5

        return np.clip(shifted_image, 0, 1)

    def simulate_edge_case(self, mri: np.ndarray, ct: np.ndarray,
                          case_type: str = "all") -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate a specific edge case or combination of edge cases.

        Args:
            mri: MRI image array
            ct: CT image array
            case_type: Type of edge case to simulate
                      ("resampling", "noise", "misalignment", "intensity", "all")

        Returns:
            Tuple of (modified_mri, ct)
        """
        modified_mri = mri.copy()

        if case_type == "resampling" or case_type == "all":
            modified_mri = self.simulate_resampling(modified_mri)

        if case_type == "noise" or case_type == "all":
            modified_mri = self.add_noise_and_artifacts(modified_mri)

        if case_type == "misalignment" or case_type == "all":
            modified_mri, ct = self.simulate_misalignment(modified_mri, ct)

        if case_type == "intensity" or case_type == "all":
            modified_mri = self.simulate_intensity_shifts(modified_mri)

        return modified_mri, ct

    def generate_edge_case_dataset(self, mri_images: list, ct_images: list,
                                 case_types: list = None) -> Tuple[list, list, list]:
        """
        Generate a dataset with various edge cases applied.

        Args:
            mri_images: List of MRI image arrays
            ct_images: List of CT image arrays
            case_types: List of edge case types to apply

        Returns:
            Tuple of (modified_mri_list, ct_list, case_labels)
        """
        if case_types is None:
            case_types = ["resampling", "noise", "misalignment", "intensity", "all"]

        modified_mris = []
        modified_cts = []
        case_labels = []

        for mri, ct in zip(mri_images, ct_images):
            for case_type in case_types:
                modified_mri, modified_ct = self.simulate_edge_case(mri, ct, case_type)
                modified_mris.append(modified_mri)
                modified_cts.append(modified_ct)
                case_labels.append(case_type)

        return modified_mris, modified_cts, case_labels


def create_edge_case_test_suite(mri_images: list, ct_images: list,
                              num_samples_per_case: int = 5) -> Dict:
    """
    Create a comprehensive edge case test suite.

    Args:
        mri_images: List of MRI image arrays
        ct_images: List of CT image arrays
        num_samples_per_case: Number of samples to generate per edge case

    Returns:
        Dictionary containing test cases organized by edge case type
    """
    simulator = EdgeCaseSimulator(seed=42)
    test_suite = {}

    case_types = ["resampling", "noise", "misalignment", "intensity", "all"]

    for case_type in case_types:
        test_suite[case_type] = {
            'mri_images': [],
            'ct_images': [],
            'original_indices': []
        }

        # Sample random images for this edge case
        sample_indices = np.random.choice(len(mri_images),
                                        min(num_samples_per_case, len(mri_images)),
                                        replace=False)

        for idx in sample_indices:
            modified_mri, modified_ct = simulator.simulate_edge_case(
                mri_images[idx], ct_images[idx], case_type)

            test_suite[case_type]['mri_images'].append(modified_mri)
            test_suite[case_type]['ct_images'].append(modified_ct)
            test_suite[case_type]['original_indices'].append(idx)

    return test_suite
