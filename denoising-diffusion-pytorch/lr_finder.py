# lr_finder.py
"""
Performs a Learning Rate Range Test with a more appropriate range to find
the optimal learning rate for the baseline training run.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torch.optim import Adam
from torch.utils.data import DataLoader

# local module imports
from dataset import SynthRADDataset

# set-up
model = Unet(dim=64, dim_mults=(1, 2, 4, 8), channels=2)
# move model to a device if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)

diffusion = GaussianDiffusion(model, image_size=128, timesteps=1000)
dataset = SynthRADDataset(root_dir='data/synthrad_train', image_size=128)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# lr range test setup
start_lr = 1e-8
end_lr = 1e-3
num_steps = len(dataloader)

optimizer = Adam(diffusion.parameters(), lr=start_lr)

lr_lambda = lambda step: (end_lr / start_lr) ** (step / num_steps)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

learning_rates = []
losses = []

# use a smoothing factor for the loss to make the plot cleaner
smoothing = 0.98
avg_loss = 0.0

print(f"Starting LR Range Test for {num_steps} steps...")
for i, (mri_tensor, ct_tensor) in enumerate(dataloader):
    # move data to the device
    combined_tensor = torch.cat([mri_tensor, ct_tensor], dim=1).to(device)
    
    loss = diffusion(combined_tensor)
    
    # loss smoothing
    avg_loss = smoothing * avg_loss + (1 - smoothing) * loss.item()
    smoothed_loss = avg_loss / (1 - smoothing**(i + 1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    losses.append(smoothed_loss)

    scheduler.step()
    
    # stop if the loss explodes
    if i > 10 and smoothed_loss > 4 * min(losses):
        print("Loss is exploding. Stopping test.")
        break

    if i % 50 == 0:
        print(f"  Step {i}/{num_steps}, LR: {current_lr:.8f}, Loss: {smoothed_loss:.4f}")

# plot the results
print("Test complete. Plotting results...")
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, losses)
plt.xscale('log')
plt.xlabel("Learning Rate")
plt.ylabel("Smoothed Loss")
plt.title("Learning Rate Range Test")
plt.grid(True)
plt.savefig('lr_range_test_plot_corrected.png')
print("Plot saved to 'lr_range_test_plot_corrected.png'")
