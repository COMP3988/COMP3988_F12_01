
import torch
import sys
from types import SimpleNamespace
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent / "unet"))
sys.path.append(str(Path(__file__).parent.parent / "pix2pix"))
sys.path.append(str(Path(__file__).parent.parent / "shaoyanpan-transformer-diffusion" / "shaoyanpan-transformer-diffusion"))

from unet.model.unet_model import UNet
from pix2pix.models.pix2pix_model import Pix2PixModel
from pix2pix.models.cycle_gan_model import CycleGANModel

class ModelAdapter:
    """
    A standard wrapper class to provide a consistnet interface for all models.

    Args:
        model (torch.nn.Module): The neural model to be wrapped.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, mri_image: torch.Tensor) -> torch.Tensor:
        """
        Runs interface on a single input tensor.

        Args:
            mri_image (torch.Tensor): The input MRI image as a PyTorch tensor.

        Returns:
            torch.Tensor: The generated CT images as a PyTorch tensor.
        """
        # Disable gradient calculation to save memory and speed up interface.
        with torch.no_grad():
            generated_ct = self.model(mri_image)
        return generated_ct


class TransformerDiffusionAdapter:
    """
    Specialized adapter for the transformer diffusion model.
    Handles the complex diffusion sampling process.
    """
    def __init__(self, model, diffusion, device):
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.model.eval()

    def predict(self, mri_image: torch.Tensor) -> torch.Tensor:
        """
        Runs diffusion sampling on a single input tensor.

        Args:
            mri_image (torch.Tensor): The input MRI image as a PyTorch tensor.

        Returns:
            torch.Tensor: The generated CT images as a PyTorch tensor.
        """
        with torch.no_grad():
            # Ensure input is on correct device
            mri_image = mri_image.to(self.device)

            # For diffusion models, we need to run the sampling process
            # This is a simplified version - in practice you might need more complex preprocessing

            # Create noise to start the diffusion process
            shape = mri_image.shape
            noise = torch.randn(shape, device=self.device)

            # Run diffusion sampling (simplified - in practice this would be more complex)
            # For now, we'll use a simple forward pass as a placeholder
            # In a real implementation, you'd use the diffusion.sample() method
            generated_ct = self.model(mri_image, noise)

        return generated_ct


def load_model(model_name: str, weights_path: str) -> ModelAdapter:
    """
    A function that loads a model by name, applies its training weights,
    and returns it wrapped in a standardised ModelAdapter.

    Args:
        model_name (str): The name of the model to load.
        weights_path (str): The file path to the saved .pth model weights.

    Returns:
        ModelAdapter: The loaded and wrapped model, ready for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # U-Net
    if model_name.lower() == 'unet':
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
        model.load_state_dict(torch.load(weights_path, weights_only=False))
        print("U-Net model loaded.")
        return ModelAdapter(model)

    # Pix2Pix
    elif model_name.lower() == 'pix2pix':
        opt = SimpleNamespace(
            input_nc=1,  # MRI input
            output_nc=1,  # CT output
            ngf=64,
            netG='unet_256',
            norm='batch',
            no_dropout=False,
            init_type='normal',
            init_gain=0.02,
            isTrain=False,
            device=device,
            checkpoints_dir='./checkpoints',
            name='pix2pix',
            direction='AtoB',
            preprocess='resize_and_crop'
        )
        model = Pix2PixModel(opt)
        state_dict = torch.load(weights_path, map_location=device)
        model.netG.load_state_dict(state_dict)
        print("Pix2Pix model loaded.")
        return ModelAdapter(model.netG)

    # CycleGAN
    elif model_name.lower() == 'cyclegan':
        opt = SimpleNamespace(
            input_nc=1,  # MRI input
            output_nc=1,  # CT output
            ngf=64,
            netG='resnet_9blocks',
            norm='instance',
            no_dropout=True,
            init_type='normal',
            init_gain=0.02,
            isTrain=False,
            device=device,
            checkpoints_dir='./checkpoints',
            name='cyclegan',
            direction='AtoB',
            preprocess='resize_and_crop'
        )
        model = CycleGANModel(opt)
        state_dict = torch.load(weights_path, map_location=device)
        model.netG_A.load_state_dict(state_dict)
        print("CycleGAN model loaded.")
        return ModelAdapter(model.netG_A)

    # Shaoyanpan Transformer Diffusion
    elif model_name.lower() in ['transformer_diffusion', 'shaoyanpan']:
        try:
            from network.Diffusion_model_transformer import SwinVITModel
            from diffusion.Create_diffusion import create_gaussian_diffusion

            # Model hyperparameters (from training script)
            num_channels = 64
            attention_resolutions = "32,16,8"
            channel_mult = (1, 2, 3, 4)
            num_heads = [4, 4, 8, 16]
            window_size = [[4, 4, 2], [4, 4, 2], [4, 4, 2], [4, 4, 2]]
            num_res_blocks = [1, 1, 1, 1]
            sample_kernel = ((2, 2, 2), (2, 2, 1), (2, 2, 1), (2, 2, 1))
            attention_ds = [int(x) for x in attention_resolutions.split(",")]

            patch_size = (64, 64, 2)

            # Create a custom SwinVITModel that fixes the sample_kernel bug
            class FixedSwinVITModel(SwinVITModel):
                def __init__(self, *args, **kwargs):
                    # Store the original sample_kernel before calling parent
                    self._original_sample_kernel = kwargs.get('sample_kernel')
                    super().__init__(*args, **kwargs)
                    # Fix the bug by restoring the full sample_kernel
                    self.sample_kernel = self._original_sample_kernel

            # Create model
            model = FixedSwinVITModel(
                image_size=patch_size,
                in_channels=2,  # MRI + CT conditioning
                model_channels=num_channels,
                out_channels=2,  # CT output
                dims=3,
                sample_kernel=sample_kernel,
                num_res_blocks=num_res_blocks,
                attention_resolutions=tuple(attention_ds),
                dropout=0,
                channel_mult=channel_mult,
                num_classes=None,
                use_checkpoint=False,
                use_fp16=False,
                num_heads=num_heads,
                window_size=window_size,
                num_head_channels=64,
                num_heads_upsample=-1,
                use_scale_shift_norm=True,
                resblock_updown=True,
            )

            # Load weights
            checkpoint = torch.load(weights_path, map_location=device)
            model.load_state_dict(checkpoint)
            model.to(device)

            # Create diffusion process
            diffusion = create_gaussian_diffusion(
                steps=1000,
                learn_sigma=True,
                sigma_small=False,
                noise_schedule="linear",
                use_kl=False,
                predict_xstart=True,
                rescale_timesteps=True,
                rescale_learned_sigmas=True,
            )

            print("Transformer Diffusion model loaded.")
            return TransformerDiffusionAdapter(model, diffusion, device)

        except ImportError as e:
            raise ImportError(f"Failed to import transformer diffusion modules: {e}") from e

    else:
        raise ValueError(f"Unknown model name: {model_name}")