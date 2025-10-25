
import torch
import sys
from types import SimpleNamespace
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent / "unet"))
sys.path.append(str(Path(__file__).parent.parent / "pix2pix"))
sys.path.append(str(Path(__file__).parent.parent / "shaoyanpan-transformer-diffusion"))

from unet.model.unet_model import UNet
from pix2pix.models.pix2pix_model import Pix2PixModel
from pix2pix.models.cycle_gan_model import CycleGANModel
from preprocess_utils import postprocess_ct_output

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
    def __init__(self, model, diffusion, device, diffusion_sampling_fn=None):
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.diffusion_sampling_fn = diffusion_sampling_fn
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

            # For transformer-diffusion, we need to handle 3D volumes
            # The model expects input shape: [B, C, D, H, W] where C=1 for MRI
            if len(mri_image.shape) == 4:  # [B, C, D, H, W] - 3D case from SynthRAD2025Dataset
                # This is already 3D data with channel dimension added by dataset
                # Shape is [B, C, D, H, W] where C=1
                pass
            elif len(mri_image.shape) == 5:  # [B, C, D, H, W] - 3D case (already 5D)
                # Already 5D, no need to modify
                pass
            elif len(mri_image.shape) == 6:  # [B, B, C, D, H, W] - Extra batch dimension from DataLoader
                # Remove the extra batch dimension
                mri_image = mri_image.squeeze(0)  # [B, C, D, H, W]
            elif len(mri_image.shape) == 3:  # [C, H, W] - 2D case, need to add batch and depth
                # Convert 2D to 3D by adding batch and depth dimensions
                mri_image = mri_image.unsqueeze(0).unsqueeze(2)  # [1, C, 1, H, W]
            else:
                raise ValueError(f"Unexpected input shape: {mri_image.shape}")

            # Ensure we have the right number of channels for conditioning
            if mri_image.shape[1] == 1:
                # Keep as 1-channel input (MRI only)
                pass  # [B, 1, D, H, W]
            elif mri_image.shape[1] == 3:
                # Convert RGB to grayscale
                mri_image = mri_image.mean(dim=1, keepdim=True)  # [B, 1, D, H, W]
            elif mri_image.shape[1] == 2:
                # Take only the first channel (MRI)
                mri_image = mri_image[:, :1, :, :, :]  # [B, 1, D, H, W]
            else:
                raise ValueError(f"Unexpected number of channels: {mri_image.shape[1]}")

            # For large volumes, use sliding window inference
            # The model expects patches of size (64, 64, 2)
            patch_size = (64, 64, 2)

            # Check if we need sliding window inference
            if (mri_image.shape[2] > patch_size[0] or
                mri_image.shape[3] > patch_size[1] or
                mri_image.shape[4] > patch_size[2]):

                # For very large volumes, resize to manageable size for testing
                # This prevents the framework from getting stuck on huge volumes
                max_size = (128, 128, 16)  # Reasonable size for testing

                if (mri_image.shape[2] > max_size[0] or
                    mri_image.shape[3] > max_size[1] or
                    mri_image.shape[4] > max_size[2]):

                    print(f"Volume too large for efficient testing: {mri_image.shape}")
                    print(f"Resizing to manageable size: {max_size}")

                    mri_image = torch.nn.functional.interpolate(
                        mri_image,
                        size=max_size,
                        mode='trilinear',
                        align_corners=False
                    )
                    print(f"Resized volume shape: {mri_image.shape}")

                # Use sliding window inference for large volumes
                try:
                    from monai.inferers import SlidingWindowInferer

                    # Create sliding window inferer with optimized settings
                    inferer = SlidingWindowInferer(
                        roi_size=patch_size,
                        sw_batch_size=1,  # Small batch size for memory efficiency
                        overlap=0.1,      # Reduced overlap for speed
                        mode='constant'
                    )

                    # Define the inference function
                    def inference_fn(x):
                        # Use the same diffusion sampling function as infer_single.py
                        if self.diffusion_sampling_fn is not None:
                            return self.diffusion_sampling_fn(self.diffusion, x, self.model)
                        else:
                            # Fallback to original method
                            output_shape = (x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4])
                            return self.diffusion.p_sample_loop(
                                self.model,
                                output_shape,
                                condition=x,
                                clip_denoised=True,
                            )

                    # Run sliding window inference
                    generated_ct = inferer(mri_image, inference_fn)

                except ImportError:
                    # Fallback: resize to patch size if monai is not available
                    print("Warning: MONAI not available, resizing to patch size")
                    mri_image_resized = torch.nn.functional.interpolate(
                        mri_image, size=patch_size, mode='trilinear', align_corners=False
                    )

                    # Run diffusion sampling using the same function as infer_single.py
                    if self.diffusion_sampling_fn is not None:
                        generated_ct_small = self.diffusion_sampling_fn(self.diffusion, mri_image_resized, self.model)
                    else:
                        # Fallback to original method
                        output_shape = (mri_image_resized.shape[0], 1, mri_image_resized.shape[2],
                                      mri_image_resized.shape[3], mri_image_resized.shape[4])
                        generated_ct_small = self.diffusion.p_sample_loop(
                            self.model,
                            output_shape,
                            condition=mri_image_resized,
                            clip_denoised=True,
                        )

                    # Resize back to original size
                    generated_ct = torch.nn.functional.interpolate(
                        generated_ct_small,
                        size=(mri_image.shape[2], mri_image.shape[3], mri_image.shape[4]),
                        mode='trilinear', align_corners=False
                    )
            else:
                # Small volume, run directly using the same function as infer_single.py
                if self.diffusion_sampling_fn is not None:
                    generated_ct = self.diffusion_sampling_fn(self.diffusion, mri_image, self.model)
                else:
                    # Fallback to original method
                    output_shape = (mri_image.shape[0], 1, mri_image.shape[2], mri_image.shape[3], mri_image.shape[4])
                    generated_ct = self.diffusion.p_sample_loop(
                        self.model,
                        output_shape,
                        condition=mri_image,
                        clip_denoised=True,
                    )

            # Remove extra depth dimension if input was 2D
            if len(mri_image.shape) == 5 and mri_image.shape[2] == 1:
                generated_ct = generated_ct.squeeze(2)  # [B, 1, H, W]

            # Postprocess CT output from [-1, 1] back to HU units
            # Convert to numpy for postprocessing
            generated_ct_np = generated_ct.detach().cpu().numpy()
            generated_ct_np = postprocess_ct_output(generated_ct_np)
            # Convert back to tensor
            generated_ct = torch.from_numpy(generated_ct_np).to(self.device)

        return generated_ct


def load_model(model_name: str, weights_path: str, diffusion_steps: int = 2) -> ModelAdapter:
    """
    A function that loads a model by name, applies its training weights,
    and returns it wrapped in a standardised ModelAdapter.

    Args:
        model_name (str): The name of the model to load.
        weights_path (str): The file path to the saved .pth model weights.
        diffusion_steps (int): Number of diffusion steps for inference (default: 2).

    Returns:
        ModelAdapter: The loaded and wrapped model, ready for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # U-Net
    if model_name.lower() == 'unet':
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
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
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
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
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.netG_A.load_state_dict(state_dict)
        print("CycleGAN model loaded.")
        return ModelAdapter(model.netG_A)

    # Shaoyanpan Transformer Diffusion
    elif model_name.lower() in ['transformer_diffusion', 'shaoyanpan']:
        try:
            # Import inference functions from infer_single.py to ensure consistency
            import sys
            from pathlib import Path
            infer_single_path = Path(__file__).parent.parent / 'shaoyanpan-transformer-diffusion'
            sys.path.append(str(infer_single_path))

            from infer_single import build_model, build_diffusion, diffusion_sampling_with

            # Build model and diffusion using infer_single functions
            model = build_model(device)
            diffusion = build_diffusion(steps=diffusion_steps)

            # Load weights
            checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint, strict=False)

            print("Transformer Diffusion model loaded using infer_single.py functions.")
            return TransformerDiffusionAdapter(model, diffusion, device, diffusion_sampling_with)

        except ImportError as e:
            raise ImportError(f"Failed to import transformer diffusion modules: {e}") from e

    else:
        raise ValueError(f"Unknown model name: {model_name}")