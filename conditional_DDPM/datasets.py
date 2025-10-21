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
        if len(self.files_A) == 0 or len(self.files_B) == 0:
            raise FileNotFoundError(f"No input files found under {root}/{mode}/a or {root}/{mode}/b with extensions {exts}")

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
            arr = sitk.GetArrayFromImage(img)  # (D,H,W)
        else:
            raise ValueError(f"Unsupported file type: {path}")
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {arr.shape} for {path}")
        return arr.astype(np.float32)

    def _to_tensor_chn_first(self, arr3d):
        # Model expects (C,D,H,W) with C=1
        t = torch.from_numpy(arr3d)  # (D,H,W)
        t = torch.unsqueeze(t, 0)    # (1,D,H,W)
        return t

    def __getitem__(self, index):
        image_A = self._load_any(self.files_A[index % len(self.files_A)])
        image_B = self._load_any(self.files_B[index % len(self.files_B)])

        item_A = self._to_tensor_chn_first(image_A)
        item_B = self._to_tensor_chn_first(image_B)
        return {"a": item_A, "b": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

