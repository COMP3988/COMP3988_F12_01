# Jan. 2023, by Junbo Peng, PhD Candidate, Georgia Tech
import glob
import random
import os
import numpy as np #V
import torch as torch #V

from torch.utils.data import Dataset
import SimpleITK as sitk
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.unaligned = unaligned

        exts = ("*.npy", "*.npz", "*.mha", "*.mhd")
        self.files_A = []
        self.files_B = []
        for e in exts:
            self.files_A += sorted(glob.glob(os.path.join(root, f"{mode}/a", e)))
            self.files_B += sorted(glob.glob(os.path.join(root, f"{mode}/b", e)))
        # If standard a/b layout is missing, fall back to SynthRAD Task2-style discovery:
        if len(self.files_A) == 0 or len(self.files_B) == 0:
            subjects = []
            for dirpath, dirnames, filenames in os.walk(root):
                files_lower = {f.lower() for f in filenames}
                if "ct.mha" in files_lower and "cbct.mha" in files_lower:
                    subjects.append(dirpath)
            subjects = sorted(subjects)
            if len(subjects) == 0:
                raise FileNotFoundError(f"No input files found under {root}/{mode}/a or {root}/{mode}/b and no subject folders with ct.mha/cbct.mha discovered")
            self.files_A = [os.path.join(s, "ct.mha") for s in subjects]
            self.files_B = [os.path.join(s, "cbct.mha") for s in subjects]

    def _load_any(self, path):
        path_lower = path.lower()
        if path_lower.endswith(".npy"):
            arr = np.load(path)
        elif path_lower.endswith(".npz"):
            d = np.load(path)
            key = "image" if "image" in d else list(d.keys())[0]
            arr = d[key]
        elif path_lower.endswith(".mha") or path_lower.endswith(".mhd"):
            img = sitk.ReadImage(path)
            arr = sitk.GetArrayFromImage(img)  # (D,H,W) or (H,W)
        else:
            raise ValueError(f"Unsupported file type: {path}")
        # Accept 2D or 3D; if 3D, pick center slice for 2D model
        if arr.ndim == 3:
            center_idx = int(arr.shape[0] // 2)
            arr2d = arr[center_idx]
        elif arr.ndim == 2:
            arr2d = arr
        else:
            raise ValueError(f"Expected 2D or 3D image, got shape {arr.shape} for {path}")
        return arr2d.astype(np.float32)

    def _to_tensor_chn_first(self, arr3d):
        # Model expects (C,H,W) with C=1 for 2D
        t = torch.from_numpy(arr3d)  # (H,W)
        t = torch.unsqueeze(t, 0)    # (1,H,W)
        return t

    def __getitem__(self, index):
        image_A = self._load_any(self.files_A[index % len(self.files_A)])
        image_B = self._load_any(self.files_B[index % len(self.files_B)])

        item_A = self._to_tensor_chn_first(image_A)
        item_B = self._to_tensor_chn_first(image_B)
        return {"a": item_A, "b": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
