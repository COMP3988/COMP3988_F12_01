import argparse
import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
import glob
import scipy.io
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from random import randint
import random
import time
import re
import itertools
from timm.models.layers import DropPath
from einops import rearrange
from scipy import ndimage
from skimage import io
from skimage import transform
from natsort import natsorted
from skimage.transform import rotate, AffineTransform
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    RandAffined,
    RandCropByLabelClassesd,
    SpatialPadd,
    RandAdjustContrastd,
    RandShiftIntensityd,
    ScaleIntensityd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    ScaleIntensityRangePercentilesd,
    Resized,
    Transposed,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    ResizeWithPadOrCropd
)
from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    RandAffined,
    SpatialPadd,
    RandAdjustContrastd,
    ScaleIntensityd,
    NormalizeIntensityd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    ScaleIntensityRangePercentilesd,
    Resized,
    Transposed,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
    ResizeWithPadOrCropd,
)
from monai.transforms.compose import MapTransform
from monai.metrics import SSIMMetric
from monai.config import print_config
from monai.metrics import DiceMetric
from skimage.transform import resize
import scipy.io
import matplotlib.pyplot as plt

import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F

import copy
from diffusion.Create_diffusion import *
from diffusion.resampler import *
from monai.inferers import SlidingWindowInferer

# ---- diffusion helpers used for evaluation ----
diffusion_steps = 1000
learn_sigma = True
sigma_small = False
noise_schedule = "linear"
use_kl = False
predict_xstart = True
rescale_timesteps = True
rescale_learned_sigmas = True

def make_eval_diffusion(num_steps=10):
    return create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=[num_steps],
    )

def make_eval_inferer(overlap=0.0, sw_batch_size=32):
    return SlidingWindowInferer(
        roi_size=(64, 64, 2),
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="constant",
    )

def diffusion_sampling_with(diffusion_obj, condition, model):
    return diffusion_obj.p_sample_loop(
        model,
        (condition.shape[0], 1, condition.shape[2], condition.shape[3], condition.shape[4]),
        condition=condition,
        clip_denoised=True,
    )

from network.Diffusion_model_transformer import SwinVITModel

# # Build the data loader using the monai library

# Here are the dataloader hyper-parameters, including the batch size,
# image size, image spacing (don't forget to adjust the spacing to your desired number)
BATCH_SIZE_TRAIN = 4*1
img_size = (256,256,128)
patch_size = (64,64,2)
spacing = (2,2,2)
patch_num = 1
channels = 1
metric = torch.nn.L1Loss()

# # Data processing for mat files (contain {'image', 'label'}) (for nii files, in the next block)

# Here we use pre-processed matlab file, which has already normalized to -1 to 1, with same spacing, same orientations.
# The reason we do that is because it can save us time to processing the data on the fly. If you don't like it,
# we provide the standard processing pipeline for nii.gz files below

# --- replace the whole CustomDataset with this npz version ---

from pathlib import Path
from natsort import natsorted
from monai.transforms import Compose, EnsureChannelFirstd, ResizeWithPadOrCropd, RandSpatialCropSamplesd, EnsureTyped

class CustomDataset(Dataset):
    def __init__(self, data_path, train_flag=True):
        self.data_path = Path(data_path)
        self.train_flag = train_flag

        self.files = natsorted([str(p) for p in self.data_path.glob("*.npz")])
        if not self.files:
            raise FileNotFoundError(f"No .npz files under {self.data_path}")

        self.train_transforms = Compose([
            EnsureChannelFirstd(keys=["image","label"], channel_dim="no_channel"),
            ResizeWithPadOrCropd(keys=["image","label"], spatial_size=img_size, constant_values=-1),
            RandSpatialCropSamplesd(keys=["image","label"], roi_size=patch_size, num_samples=patch_num, random_size=False),
            EnsureTyped(keys=["image","label"]),
        ])
        self.test_transforms = Compose([
            EnsureChannelFirstd(keys=["image","label"], channel_dim="no_channel"),
            ResizeWithPadOrCropd(keys=["image","label"], spatial_size=img_size, constant_values=-1),
            EnsureTyped(keys=["image","label"]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        with np.load(f) as d:
            mri  = d["image"].astype(np.float32)
            ct   = d["label"].astype(np.float32)

        sample = {"image": mri, "label": ct}

        if not self.train_flag:
            out = self.test_transforms(sample)
            img_tensor   = out["image"].to(torch.float)
            label_tensor = out["label"].to(torch.float)
        else:
            outs = self.train_transforms(sample)   # list of dicts (num_samples)
            img   = np.zeros([patch_num, patch_size[0], patch_size[1], patch_size[2]], dtype=np.float32)
            label = np.zeros_like(img)
            for i, s in enumerate(outs):
                img[i]   = s["image"]
                label[i] = s["label"]
            img_tensor   = torch.from_numpy(img).unsqueeze(1)    # [N,1,D,H,W]
            label_tensor = torch.from_numpy(label).unsqueeze(1)

        return img_tensor, label_tensor

def main():
    parser = argparse.ArgumentParser(description="Evaluate SSIM on SynthRAD imagesTs using a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="synthRAD_checkpoints/Synth_A_to_B.pth", help="Path to .pth checkpoint")
    parser.add_argument("--test_dir", type=str, default="SynthRAD/imagesTs", help="Directory of .npz test files")
    parser.add_argument("--num_steps", type=int, default=10, help="Diffusion steps for sampling")
    parser.add_argument("--overlap", type=float, default=0.0, help="Sliding-window overlap [0,1)")
    parser.add_argument("--sw_batch", type=int, default=4, help="Sliding-window batch size")
    parser.add_argument("--batch_size", type=int, default=1, help="DataLoader batch size")
    parser.add_argument("--out_csv", type=str, default="", help="Optional path to save per-sample SSIM as CSV")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model parameters must match training
    model = SwinVITModel(
        image_size=(64, 64, 2),  # same as patch_size used in training
        in_channels=2,
        model_channels=64,
        out_channels=2,
        dims=3,
        sample_kernel=([2, 2, 2], [2, 2, 1], [2, 2, 1], [2, 2, 1]),
        num_res_blocks=[1, 1, 1, 1],
        attention_resolutions=(32, 16, 8),
        channel_mult=(1, 2, 3, 4),
        num_heads=[4, 4, 8, 16],
        window_size=[[4, 4, 2], [4, 4, 2], [4, 4, 2], [4, 4, 2]],
        num_head_channels=64,
        use_scale_shift_norm=True,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    test_set = CustomDataset(args.test_dir, train_flag=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    eval_diffusion = make_eval_diffusion(num_steps=args.num_steps)
    eval_inferer = make_eval_inferer(overlap=args.overlap, sw_batch_size=args.sw_batch)

    ssim_metric = SSIMMetric(spatial_dims=3, data_range=2.0)  # values are in [-1,1]
    ssim_scores = []

    with torch.no_grad():
        for x1, y1 in test_loader:
            x1 = x1.to(device, dtype=torch.float32)
            y1 = y1.to(device, dtype=torch.float32)

            pred = eval_inferer(
                x1,
                lambda c, m: diffusion_sampling_with(eval_diffusion, c, m),
                model
            )

            score = ssim_metric(pred, y1)
            ssim_scores.append(score.item())

    import numpy as np
    print("Mean SSIM:", float(np.mean(ssim_scores)))
    if args.out_csv:
        import numpy as np
        np.savetxt(args.out_csv, np.array(ssim_scores), delimiter=",")
        print(f"Saved per-sample SSIM to {args.out_csv}")

if __name__ == "__main__":
    main()