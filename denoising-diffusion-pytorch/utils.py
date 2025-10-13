import torch
from tqdm import tqdm

def generate_conditional_ct(diffusion_model, mri_tensor):
    """
    Generates a CT image conditioned on an MRI tensor using the inpainting trick.

    Args:
        diffusion_model: The trained GaussianDiffusion model.
        mri_tensor: A batch of MRI tensors to condition on, e.g., shape [1, 1, 128, 128].

    Returns:
        The generated CT tensor, with shape [1, 1, 128, 128].
    """
    # Set model to evaluation mode
    diffusion_model.eval()

    # Prepare the starting tensor: real MRI + noise for the CT channel
    noise_channel = torch.randn_like(mri_tensor)
    x_t = torch.cat([mri_tensor, noise_channel], dim=1)

    # Manually iterate backwards through the timesteps to denoise the image
    for t in tqdm(reversed(range(0, diffusion_model.num_timesteps)), desc='Conditional sampling'):
        with torch.no_grad():
            x_t, _ = diffusion_model.p_sample(x_t, t)

        # Re-insert the clean MRI at every step
        x_t = torch.cat([mri_tensor, x_t[:, 1, :, :].unsqueeze(1)], dim=1)

    # The final denoised image is in x_t
    generated_ct = x_t[:, 1, :, :].unsqueeze(1)
    return generated_ct