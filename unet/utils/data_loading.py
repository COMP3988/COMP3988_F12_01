"""
file heavily adapted from original repo to fit SynthRad
specifications and requirements
"""

import logging
import numpy as np
import torch
from PIL import Image
import SimpleITK as sitk
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def _minmax_uint8(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize a 2D array to uint8 [0,255]."""
    a = arr.astype(np.float32)
    a -= a.min()
    denom = a.max() if a.max() != 0 else 1.0
    a = (a / denom) * 255.0
    return a.astype(np.uint8)


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str = None, scale: float = 1.0, mask_suffix: str = '', target: str = 'mask'):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.target = target  # 'mask' for segmentation, 'ct' for MRI->CT regression
        # Detect MHA tree mode: images_dir contains region folders like AB/HN/TH with patient subfolders.
        self.mha_mode = any((self.images_dir / d).exists() for d in ['AB', 'HN', 'TH'])
        self.samples = []

        if not self.mha_mode:
            self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
            if not self.ids:
                raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

            logging.info(f'Creating dataset with {len(self.ids)} examples')
            logging.info('Scanning mask files to determine unique values')
            with Pool() as p:
                unique = list(tqdm(
                    p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                    total=len(self.ids)
                ))

            self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
            logging.info(f'Unique mask values: {self.mask_values}')
        else:
            # Build slice-level samples from volumes under REGION/Patient/{mr.mha, ct.mha, mask.mha}
            regions = [d for d in ['AB', 'HN', 'TH'] if (self.images_dir / d).exists()]
            for region in regions:
                region_dir = self.images_dir / region
                for patient in sorted([p for p in listdir(region_dir) if (region_dir / p).is_dir()]):
                    pdir = region_dir / patient
                    mr_path = pdir / 'mr.mha'
                    ct_path = pdir / 'ct.mha'
                    mask_path = pdir / 'mask.mha'
                    if not mr_path.exists():
                        continue
                    # Read MR
                    mr_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(mr_path)))  # (D,H,W)
                    # Read target
                    if self.target == 'ct' and ct_path.exists():
                        tgt_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path)))
                    elif self.target == 'mask' and mask_path.exists():
                        tgt_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
                    else:
                        continue
                    D = min(mr_vol.shape[0], tgt_vol.shape[0])
                    for z in range(D):
                        name = f"{region}_{patient}_{z:03d}"
                        self.samples.append((mr_vol[z], tgt_vol[z], name))
            if not self.samples:
                raise RuntimeError(f'No slices found under {self.images_dir}. Expected REGION/Patient/{{mr.mha,ct.mha,mask.mha}}')
            # For MHA mode, we skip scanning unique mask values unless target == 'mask'
            if self.target == 'mask':
                self.mask_values = None  # masks are integer labels already
            else:
                self.mask_values = None
            logging.info(f'Creating MHA dataset with {len(self.samples)} slices (target={self.target})')

    def __len__(self):
        if self.mha_mode:
            return len(self.samples)
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            if mask_values is None:
                # assume img already contains integer class labels
                if img.ndim == 3:
                    # take first channel if accidentally expanded
                    img = img[..., 0]
                return img.astype(np.int64)
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        if not self.mha_mode:
            name = self.ids[idx]
            mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
            img_file = list(self.images_dir.glob(name + '.*'))

            assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
            assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
            mask = load_image(mask_file[0])
            img = load_image(img_file[0])

            assert img.size == mask.size, \
                f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

            img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
            mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

            return {
                'image': torch.as_tensor(img.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).long().contiguous()
            }
        else:
            mr_slice, tgt_slice, name = self.samples[idx]
            # Normalize each slice to uint8 for PIL-based resize pipeline
            mr_u8 = _minmax_uint8(mr_slice)
            tgt_u8 = _minmax_uint8(tgt_slice) if self.target == 'ct' else tgt_slice.astype(np.int32)
            img = self.preprocess(self.mask_values, Image.fromarray(mr_u8), self.scale, is_mask=False)
            if self.target == 'mask':
                mask = self.preprocess(self.mask_values, Image.fromarray(tgt_u8.astype(np.uint8)), self.scale, is_mask=True)
            else:
                # regression target as float in [0,1]
                mask = self.preprocess(self.mask_values, Image.fromarray(tgt_u8), self.scale, is_mask=False)
            return {
                'image': torch.as_tensor(img.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).float().contiguous() if self.target == 'ct' else torch.as_tensor(mask.copy()).long().contiguous()
            }
