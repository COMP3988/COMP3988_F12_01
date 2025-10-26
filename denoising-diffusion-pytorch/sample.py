# sample.py
"""
Generates a single, high-quality CT image from a given MRI slice.

This script is used for qualitative analysis to create a visual example 
of the model's performance. It loads a trained model, picks one MRI 
from the dataset, and saves the resulting conditional generation.
"""

import torch
from torchvision.utils import save_image

# Local module imports
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from dataset import SynthRADDataset
from utils import generate_conditional_ct

# Set a fixed seed for reproducible outputs.
torch.manual_seed(42)

# --- SETUP ---

# Initialise model and diffusion framework with the same settings as training.
model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=2)
diffusion = GaussianDiffusion(model, image_size=128, timesteps=1000)

# Load the trained model weights.
trained_model_path = 'diffusion_baseline_20_epochs.pt'
diffusion.load_state_dict(torch.load(trained_model_path))

# Load the dataset to get a sample MRI for conditioning.
dataset = SynthRADDataset(root_dir='data/synthrad_train', image_size=128)
mri_tensor, _ = dataset[0]
mri_tensor = mri_tensor.unsqueeze(0)

print("Successfully loaded model and one sample MRI.")

# --- GENERATE IMAGE ---

# Use the function from utils.py to perform conditional generation.
print("Generating conditional sample...")
generated_ct = generate_conditional_ct(diffusion, mri_tensor)

# --- SAVE RESULT ---

print("Sampling complete. Saving image...")
save_image(generated_ct, 'conditional_sample_output.png', normalize=True)
print("Conditional sample saved to 'conditional_sample_output.png'")