"""
Converted from the Jupyter notebook using nbconvert
"""

import argparse
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20,
                    help="Number of epochs to train")
args = parser.parse_args()

# ^^^ MANUALLY ADDED ^^^

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


# # Data processing for nii files (including reading, adding channels, align orientation, align spacing, normalization (MRI and CT  has different normalization), cropping and padding, and finally extracing patches for training.
# # But trust me, this process takes a lot of time. Try to process all the data before you run the model instead of processing them on-the-fly. At least try to get rid of the spacing and orientation.


# # Only be careful about the ResizeWithPadOrCropd. I am not sure should you use it or not. In my case,
# # I need a volume with fixed size.

# # One more thing, be careful of the normalization, CT is quantative and MRI is not, so they need different normalization here.
# # Maybe not your case.

# class CustomDataset(Dataset):
#     def __init__(self,imgs_path,labels_path, train_flag = True):
#         self.imgs_path = imgs_path
#         self.labels_path = labels_path
#         self.train_flag = train_flag
#         file_list = natsorted(glob.glob(self.imgs_path + "*nii.gz"), key=lambda y: y.lower())
#         label_list = natsorted(glob.glob(self.labels_path + "*nii.gz"), key=lambda y: y.lower())
#         self.data = []
#         self.label = []
#         for img_path in file_list:
#             class_name = img_path.split("/")[-1]
#             self.data.append([img_path, class_name])
#         for label_path in label_list:
#                 class_name = label_path.split("/")[-1]
#                 self.label.append([label_path, class_name])
#         self.train_transforms = Compose(
#                 [
#                     LoadImaged(keys=["image","label"],reader='nibabelreader'),
#                     EnsureChannelFirstd(keys=["image","label"]),
#                     Orientationd(keys=["image","label"], axcodes="RAS"),
#                     Spacingd(
#                         keys=["image","label"],
#                         pixdim=spacing,
#                         mode=("bilinear"),
#                     ),
#                     ScaleIntensityd(keys=["image"], minv=-1, maxv=1.0),
#                     ScaleIntensityRanged(
#                         keys=["label"],
#                         a_min=0,
#                         a_max=2674,
#                         b_min=-1,
#                         b_max=1.0,
#                         clip=True,
#                     ),
# #                     ResizeWithPadOrCropd(
# #                           keys=["image","label"],
# #                           spatial_size=img_size,
# #                           constant_values = -1,
# #                     ),
#                     RandSpatialCropSamplesd(keys=["image","label"],
#                                       roi_size = patch_size,
#                                       num_samples = patch_num,
#                                       random_size=False,
#                                       ),

#                     EnsureTyped(keys=["image","label"]),
#                 ]
#             )
#         self.test_transforms = Compose(
#                 [
#                     LoadImaged(keys=["image","label"],reader='nibabelreader'),
#                     EnsureChannelFirstd(keys=["image","label"]),
#                     Orientationd(keys=["image","label"], axcodes="RAS"),
#                     Spacingd(
#                         keys=["image","label"],
#                         pixdim=spacing,
#                         mode=("bilinear"),
#                     ),
#                     ScaleIntensityd(keys=["image"], minv=-1, maxv=1.0),
#                     ScaleIntensityRanged(
#                         keys=["label"],
#                         a_min=0,
#                         a_max=2674,
#                         b_min=-1,
#                         b_max=1.0,
#                         clip=True,
#                     ),

# #                     ResizeWithPadOrCropd(
# #                           keys=["image","label"],
# #                           spatial_size=img_size,
# #                           constant_values = -1,
# #                     ),
#                     EnsureTyped(keys=["image","label"]),
#                 ]
#             )
#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):

#         img_path, class_name = self.data[idx]
#         label_path, class_name = self.label[idx]
#         cao = {"image":img_path,'label':label_path}

#         if not self.train_flag:
#             affined_data_dict = self.test_transforms(cao)
#             img_tensor = affined_data_dict['image'].to(torch.float)
#             label_tensor = affined_data_dict['label'].to(torch.float)
#         else:
#             affined_data_dict = self.train_transforms(cao)
#             img = np.zeros([patch_num, patch_size[0], patch_size[1], patch_size[2]])
#             label = np.zeros([patch_num, patch_size[0], patch_size[1], patch_size[2]])
#             for i,after_l in enumerate(affined_data_dict):
#                 img[i,:,:,:] = after_l['image']
#                 label[i,:,:,:] = after_l['label']
#             img_tensor = torch.unsqueeze(torch.from_numpy(img.copy()), 1).to(torch.float)
#             label_tensor = torch.unsqueeze(torch.from_numpy(label.copy()), 1).to(torch.float)

#         return img_tensor,label_tensor

# # Build the MC-IDDPM process



# These three parameters: training steps number, learning variance or not (using improved DDPM or original DDPM), and inference
# timesteps number (only effective when using improved DDPM)
diffusion_steps=1000
learn_sigma=True
timestep_respacing=[50]

# Don't toch these parameters, they are irrelant to the image synthesis
sigma_small=False
class_cond=False
noise_schedule='linear'
use_kl=False
predict_xstart=True
rescale_timesteps=True
rescale_learned_sigmas=True
use_checkpoint=False


diffusion = create_gaussian_diffusion(
    steps=diffusion_steps,
    learn_sigma=learn_sigma,
    sigma_small=sigma_small,
    noise_schedule=noise_schedule,
    use_kl=use_kl,
    predict_xstart=predict_xstart,
    rescale_timesteps=rescale_timesteps,
    rescale_learned_sigmas=rescale_learned_sigmas,
    timestep_respacing=timestep_respacing,
)
schedule_sampler = UniformSampler(diffusion)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # Build the MC-IDDPM network



# Here enter your network parameters:num_channels means the initial channels in each block,
# channel_mult means the multipliers of the channels (in this case, 128,128,256,256,512,512 for the first to the sixth block),
# attention_resulution means we use the transformer blocks in the third to the sixth block
# number of heads, window size in each transformer block
#
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_channels=64
attention_resolutions="32,16,8"
channel_mult = (1, 2, 3, 4)
num_heads=[4,4,8,16]
window_size = [[4,4,2],[4,4,2],[4,4,2],[4,4,2]]
num_res_blocks = [1,1,1,1]
sample_kernel=([2,2,2],[2,2,1],[2,2,1],[2,2,1]),

attention_ds = []
for res in attention_resolutions.split(","):
    attention_ds.append(int(res))
class_cond = False
use_scale_shift_norm=True
resblock_updown = False
dropout = 0

from network.Diffusion_model_transformer import *
A_to_B_model = SwinVITModel(
          image_size=patch_size,
          in_channels=2,
          model_channels=num_channels,
          out_channels=2,
          dims=3,
          sample_kernel = sample_kernel,
          num_res_blocks=num_res_blocks,
          attention_resolutions=tuple(attention_ds),
          dropout=dropout,
          channel_mult=channel_mult,
          num_classes=None,
          use_checkpoint=False,
          use_fp16=False,
          num_heads=num_heads,
          window_size = window_size,
          num_head_channels=64,
          num_heads_upsample=-1,
          use_scale_shift_norm=use_scale_shift_norm,
          resblock_updown=resblock_updown,
          use_new_attention_order=False,
      ).to(device)


pytorch_total_params = sum(p.numel() for p in A_to_B_model.parameters())
print('parameter number is '+str(pytorch_total_params))
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
optimizer = torch.optim.AdamW(A_to_B_model.parameters(), lr=2e-5,weight_decay = 1e-4)
scaler = torch.cuda.amp.GradScaler()

# # Build the training function. Run the training function once = one epoch



# Here we explain the training process
def train(model, optimizer,data_loader1, loss_history, max_steps_per_epoch=None):

    #1: set the model to training mode
    model.train()
    total_samples = len(data_loader1.dataset)
    A_to_B_loss_sum = []
    total_time = 0

    #2: Loop the whole dataset, x1 (traindata) is the image batch
    pbar = tqdm(data_loader1, total=len(data_loader1), desc=f"Train {epoch}")
    for i, (x1,y1) in enumerate(pbar):

        if max_steps_per_epoch is not None and i >= max_steps_per_epoch:
            break

        if i % 5 == 0 and A_to_B_loss_sum:
            pbar.set_postfix(avg_loss=float(np.nanmean(A_to_B_loss_sum)))

        traintarget = y1.view(-1,1,patch_size[0],patch_size[1],patch_size[2]).to(device)

        traincondition = x1.view(-1,1,patch_size[0],patch_size[1],patch_size[2]).to(device)

        #3: extract random timestep for training
        t, weights = schedule_sampler.sample(traincondition.shape[0], device)

        aa = time.time()

        #4: Optimize the TDM network

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            all_loss = diffusion.training_losses(A_to_B_model,traintarget,traincondition, t)
            A_to_B_loss = (all_loss["loss"] * weights).mean()

            A_to_B_loss_sum.append(all_loss["loss"].mean().detach().cpu().numpy())

        scaler.scale(A_to_B_loss).backward()

        scaler.step(optimizer)

        scaler.update()

        #5:print out the intermediate loss for every 100 batches
        total_time += time.time()-aa

    #6: print out the average loss for this epoch
    average_loss = np.nanmean(A_to_B_loss_sum)
    loss_history.append(average_loss)
    print("Total time per sample is: "+str(total_time))
    print('Averaged loss is: '+ str(average_loss))
    return average_loss

# # Build the testing function.



# Use the window sliding method to translate the whole MRI to CT volume. Must used it.
# For example, if your whole volume is 64x64x64, and our window size is 64x64x4, so the function will automatically sliding down
# the whole volume with a certain overlapping ratio

# The window size (patch_size) is shown in the "Build the data loader using the monai library" section.
# patch_size: the size of sliding window
# img_num: the number of sliding window in each process, only related to your gpu memory, it will still run through the whole volume
# overlap: the overlapping ratio
from monai.inferers import SlidingWindowInferer
img_num = 12
overlap = 0.5
inferer = SlidingWindowInferer(patch_size, img_num, overlap=overlap, mode ='constant')
def diffusion_sampling(condition, model):
    sampled_images = diffusion.p_sample_loop(model,(condition.shape[0], 1,
                                                    condition.shape[2], condition.shape[3],condition.shape[4]),
                                                    condition = condition,clip_denoised=True)
    return sampled_images

# ---- fast eval helpers ----
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
        timestep_respacing=[num_steps],  # far fewer steps than train
    )

def make_eval_inferer(overlap=0.0, sw_batch_size=32):
    return SlidingWindowInferer(
        roi_size=patch_size,
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

# ---- cheap eval ----
def evaluate(model, epoch, out_dir, data_loader1, best_loss, save_outputs=False,
             max_batches=1, eval_steps=10, eval_overlap=0.0, eval_sw_batch=32):
    model.eval()
    loss_all, prediction, true, img = [], [], [], []
    eval_diffusion = make_eval_diffusion(num_steps=eval_steps)
    eval_inferer   = make_eval_inferer(overlap=eval_overlap, sw_batch_size=eval_sw_batch)

    t0 = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i, (x1, y1) in enumerate(tqdm(data_loader1, total=min(len(data_loader1), max_batches),
                                          desc=f"Eval {epoch}", leave=False)):
            target = y1.to(device)
            condition = x1.to(device)
            sampled = eval_inferer(condition, lambda c, m: diffusion_sampling_with(eval_diffusion, c, m), model)
            loss = metric(sampled, target)
            loss_all.append(loss.detach().cpu().numpy())

            if save_outputs:
                img.append(x1.cpu().numpy()); true.append(target.cpu().numpy()); prediction.append(sampled.cpu().numpy())

            if i + 1 >= max_batches:
                break

    avg = float(np.mean(loss_all)) if loss_all else float("nan")
    print(f"Eval time: {time.time()-t0:.2f}s  avg L1: {avg:.6f}")

    if save_outputs:
        data = {"img": img, "label": true, "prediction": prediction, "loss": avg}
        scipy.io.savemat(os.path.join(out_dir, f"test_example_epoch{epoch}.mat"), data)
        if avg < best_loss:
            scipy.io.savemat(os.path.join(out_dir, "all_final_test_another.mat"), data)
    return avg

# # Start the training and testing


training_path = os.path.join('SynthRAD', 'imagesTr')
testing_path = os.path.join('SynthRAD', 'imagesTs')

training_set1 = CustomDataset(training_path, train_flag=True)
testing_set1 = CustomDataset(testing_path, train_flag=False)

# Enter your data reader parameters
train_params = {
    "batch_size": BATCH_SIZE_TRAIN,
    "shuffle": True,
    "pin_memory": True,
    "drop_last": False,
    "num_workers": 4,
    "persistent_workers": True,
    "prefetch_factor": 4,
}
train_loader1 = torch.utils.data.DataLoader(training_set1, **train_params)

test_params = {**train_params, "batch_size": 1, "shuffle": False}
test_loader1 = torch.utils.data.DataLoader(testing_set1, **test_params)

# Enter your total number of epoch
N_EPOCHS = args.epochs

# Enter the address you save the checkpoint and the evaluation examples
checkpoint_dir = 'synthRAD_checkpoints'

os.makedirs(checkpoint_dir, exist_ok=True)
A_to_B_PATH = os.path.join(checkpoint_dir, 'Synth_A_to_B.pth')
best_loss = float('inf')

train_loss_history, test_loss_history = [], []

# Uncomment this when you resume the checkpoint
#A_to_B_model.load_state_dict(torch.load(A_to_B_PATH),strict=False)
for epoch in range(0, N_EPOCHS):
    print('Epoch:', epoch)
    start_time = time.time()
    train(A_to_B_model, optimizer,train_loader1, train_loss_history, 1) # TODO remove 5
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    if epoch % 5 == 0:
        print("Starting eval...")
        average_loss = evaluate(A_to_B_model, epoch, checkpoint_dir, test_loader1, best_loss,
                save_outputs=False, max_batches=1, eval_steps=10, eval_overlap=0.0, eval_sw_batch=32)
        print("Eval done.")
        if average_loss < best_loss:
            print('Save the latest best model')
            torch.save(A_to_B_model.state_dict(), A_to_B_PATH)
            best_loss = average_loss

print('Execution time')
