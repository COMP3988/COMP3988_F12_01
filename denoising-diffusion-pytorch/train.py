# train.py

"""
Main training script for the conditional diffusion model.

This script initialises the U-Net and Gaussian Diffusion models, loads a small subset
of the SynthRAD dataset for a controlled experiment, runs the main training loop,
and saves the final trained model weights as an artifact.
"""

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Local module imports
from dataset import SynthRADDataset

# --- MODEL CONFIGURATION ---

# Define the U-Net model architecture.
# channels = 2 because training on a stacked (MRI, CT) tensor.
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 2
)

# Wrap the U-Net in the Gaussian Diffusion model.
# This class manages the noising and denoising processes.
diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000
)

# --- DATA LOADING (CONTROLLED EXPERIMENT) ---

# Load the full dataset once.
dataset = SynthRADDataset(root_dir='data/synthrad_train', image_size=128)
dataloader = DataLoader(
    dataset,
    batch_size = 1,
    shuffle = True
)

# --- TRAINING SETUP ---
# Set up the optimizer
optimizer = Adam(
    diffusion.parameters(),
    lr = 8e-5
)

# --- TRAINING LOOP ---
epochs = 20 

for epoch in range(epochs):
    print(f"Starting Epoch {epoch+1}/{epochs}")
    for i, (mri_tensor, ct_tensor) in enumerate(dataloader):
        # Combine the MRI and CT into a single 2-channel tensor.
        # The model learns the joint distribution of these paired images.
        combined_tensor = torch.cat([mri_tensor, ct_tensor], dim=1)

        # Calculate the diffusion loss for a batch.
        loss = diffusion(combined_tensor)
        
        # Backpropagate and update model weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print loss every 50 steps
        if i % 50 == 0: 
            print(f"  Step {i}, Loss: {loss.item()}")

# --- SAVE FINAL MODEL ---

final_model_path = 'diffusion_baseline_20_epochs.pt'
print(f"Training complete. Saving final model to {final_model_path}...")
torch.save(diffusion.state_dict(), final_model_path)


# --- GENERATE SAMPLE IMAGES ---
print("Generating 4 sample images with the trained model...")
# Load the model we just saved
diffusion.load_state_dict(torch.load(final_model_path))

# This tests if the model has learned what a CT looks like in general.
sampled_images = diffusion.sample(batch_size = 4)

generated_ct_channel = sampled_images[:, 1, :, :].unsqueeze(1)

save_image(generated_ct_channel, 'diffusion_sample_output.png', nrow = 2, normalize=True)
print("Sample images saved to 'diffusion_sample_output.png'")