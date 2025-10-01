"""
Converted from the Jupyter notebook using nbconvert
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default="20",
                    help="Number of epochs to train")
args = parser.parse_args()

# ^^^ MANUALLY ADDED ^^^

#!/usr/bin/env python
# coding: utf-8

# # Here are the library you need to import

# In[ ]:


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
    AddChanneld,
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
    ToTensord,
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
from monai.transforms import (CastToTyped,
                              Compose, CropForegroundd, EnsureChannelFirstd, LoadImaged,
                              NormalizeIntensity, RandCropByPosNegLabeld,
                              RandFlipd, RandGaussianNoised,
                              RandGaussianSmoothd, RandScaleIntensityd,
                              RandZoomd, SpatialCrop, SpatialPadd, EnsureTyped)
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

# In[ ]:


# Here are the dataloader hyper-parameters, including the batch size,
# image size, image spacing (don't forget to adjust the spacing to your desired number)
BATCH_SIZE_TRAIN = 4*1
img_size = (256,256,128)
patch_size = (64,64,4)
spacing = (2,2,2)
patch_num = 2
channels = 1
metric = torch.nn.L1Loss()

# # Data processing for mat files (contain {'image', 'label'}) (for nii files, in the next block)

# In[ ]:


# Here we use pre-processed matlab file, which has already normalized to -1 to 1, with same spacing, same orientations.
# The reason we do that is because it can save us time to processing the data on the fly. If you don't like it,
# we provide the standard processing pipeline for nii.gz files below

class CustomDataset(Dataset):
    def __init__(self,imgs_path,labels_path, train_flag = True):
        self.imgs_path = imgs_path
        self.labels_path = labels_path
        self.train_flag = train_flag
        file_list = natsorted(glob.glob(self.imgs_path + "*"), key=lambda y: y.lower())
        label_list = natsorted(glob.glob(self.labels_path + "*"), key=lambda y: y.lower())

        self.data = []
        self.label = []
        for img_path in file_list:
            class_name = img_path.split("/")[-1]
            self.data.append([img_path, class_name])
        for label_path in label_list:
                class_name = label_path.split("/")[-1]
                self.label.append([label_path, class_name])
        self.train_transforms = Compose(
                [
                    AddChanneld(keys=["image","label"]),
                    ResizeWithPadOrCropd(
                          keys=["image","label"],
                          spatial_size=img_size,
                          constant_values = -1,
                    ),
                    RandSpatialCropSamplesd(keys=["image","label"],
                                      roi_size = patch_size,
                                      num_samples = patch_num,
                                      random_size=False,
                                      ),
                    ToTensord(keys=["image","label"]),
                ]
            )
        self.test_transforms = Compose(
                [
                    AddChanneld(keys=["image","label"]),
                    ResizeWithPadOrCropd(
                          keys=["image","label"],
                          spatial_size=img_size,
                          constant_values = -1,
                    ),
                    ToTensord(keys=["image","label"]),
                ]
            )
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path, class_name = self.data[idx]

        cao = scipy.io.loadmat(img_path)

        if not self.train_flag:
            affined_data_dict = self.test_transforms(cao)
            img_tensor = affined_data_dict['image'].to(torch.float)
            label_tensor = affined_data_dict['label'].to(torch.float)
        else:
            affined_data_dict = self.train_transforms(cao)
            img = np.zeros([patch_num, patch_size[0], patch_size[1], patch_size[2]])
            label = np.zeros([patch_num, patch_size[0], patch_size[1], patch_size[2]])
            for i,after_l in enumerate(affined_data_dict):
                img[i,:,:,:] = after_l['image']
                label[i,:,:,:] = after_l['label']
            img_tensor = torch.unsqueeze(torch.from_numpy(img.copy()), 1).to(torch.float)
            label_tensor = torch.unsqueeze(torch.from_numpy(label.copy()), 1).to(torch.float)


        return img_tensor,label_tensor

# # Data processing for nii files (including reading, adding channels, align orientation, align spacing, normalization (MRI and CT  has different normalization), cropping and padding, and finally extracing patches for training.
#
# # But trust me, this process takes a lot of time. Try to process all the data before you run the model instead of processing them on-the-fly. At least try to get rid of the spacing and orientation.

# In[ ]:


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
#                     AddChanneld(keys=["image","label"]),
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

#                     ToTensord(keys=["image","label"]),
#                 ]
#             )
#         self.test_transforms = Compose(
#                 [
#                     LoadImaged(keys=["image","label"],reader='nibabelreader'),
#                     AddChanneld(keys=["image","label"]),
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
#                     ToTensord(keys=["image","label"]),
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

# In[ ]:


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

# In[ ]:


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
window_size = [[4,4,4],[4,4,4],[4,4,2],[4,4,2]]
num_res_blocks = [2,2,2,2]
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

# In[ ]:


# #In case you want to use CNN
# from network.Diffusion_model_Unet import *
# A_to_B_model = UNetModel(
#         img_size = patch_size,
#         image_size=patch_size[0],
#         in_channels=2,
#         model_channels=num_channels,
#         out_channels=2,
#         dims = 3,
#         num_res_blocks=num_res_blocks[0],
#         attention_resolutions=tuple(attention_ds),
#         dropout=0.,
#         sample_kernel=sample_kernel,
#         channel_mult=channel_mult,
#         num_classes=(128 if class_cond else None),
#         use_checkpoint=False,
#         use_fp16=False,
#         num_heads=4,
#         num_head_channels=64,
#         num_heads_upsample=-1,
#         use_scale_shift_norm=use_scale_shift_norm,
#         resblock_updown=False,
#         use_new_attention_order=False,
#     ).to(device)

# # Call the optimizer and ready for start

# In[ ]:


pytorch_total_params = sum(p.numel() for p in A_to_B_model.parameters())
print('parameter number is '+str(pytorch_total_params))
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(A_to_B_model.parameters(), lr=2e-5,weight_decay = 1e-4)
scaler = torch.cuda.amp.GradScaler()

# # Build the training function. Run the training function once = one epoch

# In[ ]:


# Here we explain the training process
def train(model, optimizer,data_loader1, loss_history):

    #1: set the model to training mode
    model.train()
    total_samples = len(data_loader1.dataset)
    A_to_B_loss_sum = []
    total_time = 0

    #2: Loop the whole dataset, x1 (traindata) is the image batch
    for i, (x1,y1) in enumerate(data_loader1):

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
        if i % 30 == 0:
            print('optimization time: '+ str(time.time()-aa))
            print('[' +  '{:5}'.format(i * BATCH_SIZE_TRAIN) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader1)) + '%)]  A_to_B_Loss: ' +
                  '{:6.7f}'.format(np.nanmean(A_to_B_loss_sum)))

    #6: print out the average loss for this epoch
    average_loss = np.nanmean(A_to_B_loss_sum)
    loss_history.append(average_loss)
    print("Total time per sample is: "+str(total_time))
    print('Averaged loss is: '+ str(average_loss))
    return average_loss

# # Build the testing function.

# In[ ]:


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

# Run the evaluate function will translate the MRI to CT and will be save to a folder in MAT format
def evaluate(model,epoch,path,data_loader1,best_loss):
    model.eval()
    prediction = []
    true = []
    img = []
    loss_all = []
    aa = time.time()
    with torch.no_grad():
        for i, (x1,y1) in enumerate(data_loader1):
                # target is the target CT
                # condition is the input MRI
                # sampled_images is the synthetic CT
                target = y1.to(device)
                condition = x1.to(device)
                with torch.cuda.amp.autocast():
                      sampled_images = inferer(condition,diffusion_sampling,model)
                loss = metric(sampled_images,target)
                print('L1 loss: '+ str(loss))
                img.append(x1.cpu().numpy())
                true.append(target.cpu().numpy())
                prediction.append(sampled_images.cpu().numpy())
                loss_all.append(loss.cpu().numpy())


        print('optimization time: '+ str(1*(time.time()-aa)))
        # The save code, you can replace it by your code for other files, e.g. nii or dicom
        data = {"img":img,'label':true,'prediction':prediction,'loss':np.mean(loss_all)}
        scipy.io.savemat(path+ 'test_example_epoch'+str(epoch)+'.mat',data)
        if np.mean(loss_all) < best_loss:
            scipy.io.savemat(path+ 'all_final_test_another.mat',data)
        return np.mean(loss_all)

# # Start the training and testing

# In[ ]:


# Enter your data folder
from os import path

training_path = path.join('SynthRAD', 'imagesTr')
testing_path = path.join('SynthRAD', 'imagesTs')

training_set1 = CustomDataset(training_path, training_path, train_flag=True)
testing_set1 = CustomDataset(testing_path, testing_path, train_flag=False)

# Enter your data reader parameters
train_params = {'batch_size': BATCH_SIZE_TRAIN,
          'shuffle': True,
          'pin_memory': True,
          'drop_last': False}
train_loader1 = torch.utils.data.DataLoader(training_set1, **train_params)

test_params = {'batch_size': 1,
          'shuffle': False,
          'pin_memory': True,
          'drop_last': False}
test_loader1 = torch.utils.data.DataLoader(testing_set1, **test_params)

# Enter your total number of epoch
N_EPOCHS = args.epochs

# Enter the address you save the checkpoint and the evaluation examples
checkpoint_dir = 'synthRAD_checkpoints'
A_to_B_PATH = path.join(checkpoint_dir, 'Synth_A_to_B.pth')
best_loss = 1
if not os.path.exists(path):
  os.makedirs(path)
train_loss_history, test_loss_history = [], []

# Uncomment this when you resume the checkpoint
#A_to_B_model.load_state_dict(torch.load(A_to_B_PATH),strict=False)
for epoch in range(0, N_EPOCHS):
    print('Epoch:', epoch)
    start_time = time.time()
    train(A_to_B_model, optimizer,train_loader1, train_loss_history,)
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    if epoch % 5 == 0:
        average_loss = evaluate(A_to_B_model,epoch,path,test_loader1,best_loss)
        if average_loss < best_loss:
            print('Save the latest best model')
            torch.save(A_to_B_model.state_dict(), A_to_B_PATH)
            best_loss = average_loss
print('Execution time')
