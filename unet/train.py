import os
from pathlib import Path
from typing import Optional, List, Tuple, Union

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from torch.utils.data import Dataset

from functools import lru_cache

import argparse
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from model import UNet

@lru_cache(maxsize=2)
def _read_itk_volume(path: str) -> np.ndarray:
    """Read a MetaImage volume and return NumPy array shaped (D,H,W). Cached to avoid re-reading."""
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def _minmax_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize image to uint8"""
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min == 0:
        return np.zeros_like(img, dtype=np.uint8)
    img_norm = (img - img_min) / (img_max - img_min)
    img_uint8 = (img_norm * 255).astype(np.uint8)
    return img_uint8

def _resize_array_uint8(arr: np.ndarray, scale: float, is_mask: bool) -> np.ndarray:
    if scale == 1.0:
        return arr
    img = Image.fromarray(arr)
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    w, h = img.size
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return np.asarray(img.resize(new_size, resample=resample))

def _to_chw_float01(arr_u8: np.ndarray) -> np.ndarray:
    x = arr_u8.astype(np.float32) / 255.0
    return x[None, ...]  # (1,H,W)

class BasicDataset(Dataset):
    def __init__(self,
                 root_dir: Union[str, Path],
                 mask_dir: Optional[Union[str, Path]] = None,
                 scale: float = 1.0,
                 target: str = 'mask',
                 max_per_section: int = 0):
        self.root_dir = Path(root_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.scale = scale
        self.target = target  # 'mask' or 'ct'
        self.max_per_section = int(max_per_section) if max_per_section is not None else 0

        self.samples = []

        # Determine mode by presence of MHA files
        mha_mode = any(self.root_dir.glob('**/*.mha'))

        if mha_mode:
            # Load MHA volumes lazily by storing file paths and slice indices
            for region_dir in self.root_dir.iterdir():
                if not region_dir.is_dir():
                    continue
                region = region_dir.name
                used = 0
                limit = self.max_per_section if self.max_per_section > 0 else None
                for patient_dir in region_dir.iterdir():
                    if not patient_dir.is_dir():
                        continue
                    if patient_dir.name.lower() == 'overviews':
                        continue
                    if limit is not None and used >= limit:
                        break
                    patient = patient_dir.name
                    mr_path = patient_dir / 'mr.mha'
                    ct_path = patient_dir / 'ct.mha'
                    mask_path = patient_dir / 'mask.mha' if self.target == 'mask' else None

                    # Determine depth without keeping arrays alive

                    mr_img = sitk.ReadImage(str(mr_path))
                    tgt_img = sitk.ReadImage(str(ct_path if self.target == 'ct' else mask_path))
                    D = min(mr_img.GetSize()[2], tgt_img.GetSize()[2])  # D = slices

                    added = 0
                    for z in range(D):
                        if limit is not None and used >= limit:
                            break
                        name = f"{region}_{patient}_{z:03d}"
                        # Store file paths and slice index only
                        self.samples.append((str(mr_path), str(ct_path if self.target == 'ct' else mask_path), z, name))
                        added += 1
                    used += added

        else:
            # Non-MHA mode: load all slices into memory
            for region_dir in self.root_dir.iterdir():
                if not region_dir.is_dir():
                    continue
                region = region_dir.name
                used = 0
                limit = self.max_per_section if self.max_per_section > 0 else None
                for patient_dir in region_dir.iterdir():
                    if not patient_dir.is_dir():
                        continue
                    if patient_dir.name.lower() == 'overviews':
                        continue
                    if limit is not None and used >= limit:
                        break
                    patient = patient_dir.name
                    mr_path = patient_dir / 'mr.png'
                    mask_path = patient_dir / 'mask.png' if self.target == 'mask' else None
                    ct_path = patient_dir / 'ct.png' if self.target == 'ct' else None

                    mr_img = Image.open(mr_path).convert('L')
                    mr_vol = np.array(mr_img)

                    if self.target == 'mask':
                        tgt_img = Image.open(mask_path).convert('L')
                        tgt_vol = np.array(tgt_img)
                    else:
                        tgt_img = Image.open(ct_path).convert('L')
                        tgt_vol = np.array(tgt_img)

                    D = min(mr_vol.shape[0], tgt_vol.shape[0])
                    added = 0
                    for z in range(D):
                        if limit is not None and used >= limit:
                            break
                        name = f"{region}_{patient}_{z:03d}"
                        self.samples.append((mr_vol[z], tgt_vol[z], name))
                        added += 1
                    used += added

        self.mask_values = None
        if self.target == 'mask':
            self.mask_values = self._get_mask_values()

        self.out_size = None if self.scale == 1.0 else self.scale

    def _get_mask_values(self):
        # Placeholder for mask value extraction logic
        return [0, 1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mha_mode = isinstance(self.samples[0][0], str)

        if mha_mode:
            mr_path, tgt_path, z, name = self.samples[idx]
            mr_vol = _read_itk_volume(mr_path)
            tgt_vol = _read_itk_volume(tgt_path)
            mr_slice = mr_vol[z]
            tgt_slice = tgt_vol[z]
        else:
            mr_slice, tgt_slice, name = self.samples[idx]

        # Normalize slices
        mr_u8 = _minmax_uint8(mr_slice)
        if self.target == 'ct':
            tgt_u8 = _minmax_uint8(tgt_slice)
        else:
            tgt_u8 = tgt_slice.astype(np.int32)

        # Optional resize
        mr_u8 = _resize_array_uint8(mr_u8, self.scale, is_mask=False)
        if self.target == 'mask':
            tgt_arr = _resize_array_uint8(tgt_u8.astype(np.uint8), self.scale, is_mask=True).astype(np.int64)
        else:
            tgt_arr = _resize_array_uint8(tgt_u8, self.scale, is_mask=False)

        # To tensors
        img = torch.as_tensor(_to_chw_float01(mr_u8).copy()).float().contiguous()
        if self.target == 'mask':
            mask = torch.as_tensor(tgt_arr.copy()).long().contiguous()
        else:
            mask = torch.as_tensor(_to_chw_float01(tgt_arr).copy()).float().contiguous()

        return {'image': img, 'mask': mask}

# ----------------------- Training script -----------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, default='./synthRAD2025_Task1_Train/Task1')
    p.add_argument('--target', type=str, choices=['ct','mask'], default='ct')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--validation', type=float, default=5.0)
    p.add_argument('--scale', type=float, default=1.0)
    p.add_argument('--bilinear', action='store_true', default=True)
    p.add_argument('--samples-per-section', type=int, default=0, help='Max slices to use per section (AB/HN/TH). 0 = all')
    return p.parse_args()

def make_loaders(ds, batch_size, val_percent):
    n_val = max(1, int(len(ds) * (val_percent / 100.0)))
    n_train = len(ds) - n_val
    train_set, val_set = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=False)
    return (
        DataLoader(train_set, shuffle=True, **loader_args),
        DataLoader(val_set, shuffle=False, drop_last=False, **loader_args),
        n_train, n_val
    )

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    logging.info(f'Using device {device}')

    # Dataset and loaders
    dataset = BasicDataset(args.data_root, scale=args.scale, target=args.target, max_per_section=args.samples_per_section)
    train_loader, val_loader, n_train, n_val = make_loaders(dataset, args.batch_size, args.validation)
    logging.info(f'Dataset slices: train={n_train}, val={n_val}')

    # Model
    in_ch = 1
    out_ch = 1 if args.target == 'ct' else 2
    model = UNet(n_channels=in_ch, n_classes=out_ch, bilinear=args.bilinear).to(device)

    # Loss
    if args.target == 'ct':
        criterion = nn.L1Loss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{args.epochs}', unit='batch')
        for batch in train_loader:
            imgs = batch['image'].to(device)
            tgts = batch['mask'].to(device)
            preds = model(imgs)
            if args.target == 'ct':
                loss = criterion(preds, tgts)
            else:
                loss = criterion(preds, tgts)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))
            pbar.update(1)
        pbar.close()

        # Simple val
        model.eval()
        with torch.no_grad():
            losses = []
            for batch in val_loader:
                imgs = batch['image'].to(device)
                tgts = batch['mask'].to(device)
                preds = model(imgs)
                if args.target == 'ct':
                    loss = criterion(preds, tgts)
                else:
                    loss = criterion(preds, tgts)
                losses.append(float(loss.detach().cpu()))
        logging.info(f'Val loss: {np.mean(losses) if losses else float("nan")}')

        # Save checkpoint
        here = Path(__file__).resolve().parent
        ckpt_dir = here / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        logging.info(f'Checkpoint saved: checkpoints/checkpoint_epoch{epoch}.pth')
