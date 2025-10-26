import os
import torch
import numpy as np
import SimpleITK as sitk
import torch.nn.functional as F
from tqdm import tqdm

from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from utils import generate_conditional_ct

# configs
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# paths
root_dir = "synthRAD2025_Task1_Train/Task1/AB/1ABA005"
mr_path = os.path.join(root_dir, "mr.mha")
mask_path = os.path.join(root_dir, "mask.mha")
output_path = os.path.join(root_dir, "generated_ct.mha")

# model path
best_model_path = "/content/drive/MyDrive/denoising-diffusion-pytorch/checkpoints/best_model.pt"

# load model
model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=2)
diffusion = GaussianDiffusion(model, image_size=128, timesteps=1000)
diffusion.load_state_dict(torch.load(best_model_path, map_location=device))
diffusion.to(device)
diffusion.eval()

print(f"Loaded model weights from {best_model_path}")

# load mri
mr_image = sitk.ReadImage(mr_path)
mr_array = sitk.GetArrayFromImage(mr_image).astype(np.float32)

# normalise to [-1, 1]
mr_array = (mr_array - np.min(mr_array)) / (np.max(mr_array) - np.min(mr_array))
mr_array = mr_array * 2 - 1

# select middle slice if 3D
if mr_array.ndim == 3:
    slice_idx = mr_array.shape[0] // 2
    mr_slice = mr_array[slice_idx]
else:
    mr_slice = mr_array

# resize to 128x128 and ensure divisible by 8
mr_slice = torch.tensor(mr_slice).unsqueeze(0).unsqueeze(0).to(device)
mr_slice = F.interpolate(mr_slice, size=(128, 128), mode="bilinear", align_corners=False)

print("Generating CT image...")

# generate CT
with torch.no_grad():
    generated_ct = generate_conditional_ct(diffusion, mr_slice)

# convert back to [0, 1]
gen_np = (generated_ct.squeeze().cpu().numpy() + 1) / 2

# save output
# create a 2D SimpleITK image and set spacing/origin/direction safely
gen_ct_image = sitk.GetImageFromArray(gen_np.astype(np.float32))

spacing = mr_image.GetSpacing()
origin = mr_image.GetOrigin()
direction = mr_image.GetDirection()

if len(gen_np.shape) == 2:
    # set 2D spacing and origin
    gen_ct_image.SetSpacing(spacing[:2])
    gen_ct_image.SetOrigin(origin[:2])

    # extract in-plane 2x2 direction
    direction2d = np.array([[direction[0], direction[1]],
                            [direction[3], direction[4]]], dtype=np.float32)

    # fallback to identity if determinant is 0
    if np.linalg.det(direction2d) == 0:
        direction2d = np.array([[1.0, 0.0],
                                [0.0, 1.0]], dtype=np.float32)

    gen_ct_image.SetDirection(direction2d.flatten())
else:
    gen_ct_image.SetSpacing(spacing)
    gen_ct_image.SetOrigin(origin)
    gen_ct_image.SetDirection(direction)

sitk.WriteImage(gen_ct_image, output_path)
print(f"Generated CT saved to: {output_path}")
