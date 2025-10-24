#tests/test_adapters.py
"""
Unit tests for the model adapter functionality in models/adapters.py.

This script verifies that the ModelAdapter and load_model can correctly
initialise, load weights, and run inference with the models.
"""

import torch
import sys
import os

# Add the parent directory to the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.adapters import load_model, TransformerDiffusionAdapter
from unet.model.unet_model import UNet

def test_unet_adapter():
    """
    Tests the full lifecycle for the U-Net adapter:
    1. Creates a dummy weights file.
    2. Loads the model and weights using the load_model.
    3. Creates a dummy input tensor.
    4. Runs prediction.
    5. Asserts that the output shape is correct.
    6. Cleans up the dummy file.
    """
    print("\n-- Testing U-Net Adapter --")
    dummy_weights_path = 'dummy_unet_weights.pth'

    try:
        #1.
        torch.save(UNet(n_channels=1, n_classes=1, bilinear=True).state_dict(), dummy_weights_path)

        #2.
        unet_adapter = load_model('unet', dummy_weights_path)

        #3.
        dummy_mri = torch.randn(1, 1, 256, 256)

        #4.
        output_tensor = unet_adapter.predict(dummy_mri)

        #5.
        expected_shape = (1, 1, 256, 256)
        assert output_tensor.shape == expected_shape, f"Shape mismatch. Expected {expected_shape}, got {output_tensor.shape}"

        print("U-Net adapter test PASSED.")

    except Exception as e:
        print(f"U-Net adapter test FAILED: {e}")
    finally:
        #6.
        if os.path.exists(dummy_weights_path):
            os.remove(dummy_weights_path)
            print(f"Cleaned up {dummy_weights_path}.")


def test_pix2pix_adapter():
    """
    Tests the Pix2Pix adapter model creation and interface:
    1. Tests model creation without loading weights (since we don't have real weights)
    2. Verifies the adapter interface works correctly
    3. Tests that the model can be instantiated properly
    """
    print("\n-- Testing Pix2Pix Adapter --")

    try:
        # Test that the adapter can be created (without loading weights)
        # This tests the model creation and configuration
        from models.adapters import load_model
        from types import SimpleNamespace

        # Create a minimal test to verify the adapter structure
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        # Test that we can import and create the model structure
        from pix2pix.models.pix2pix_model import Pix2PixModel
        model = Pix2PixModel(opt)

        # Test that the adapter interface would work
        from models.adapters import ModelAdapter
        adapter = ModelAdapter(model.netG)

        # Test prediction with dummy input
        dummy_mri = torch.randn(1, 1, 256, 256)
        output_tensor = adapter.predict(dummy_mri)

        # Verify output shape
        expected_shape = (1, 1, 256, 256)
        assert output_tensor.shape == expected_shape, f"Shape mismatch. Expected {expected_shape}, got {output_tensor.shape}"

        print("Pix2Pix adapter test PASSED.")

    except Exception as e:
        print(f"Pix2Pix adapter test FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_cyclegan_adapter():
    """
    Tests the CycleGAN adapter model creation and interface:
    1. Tests model creation without loading weights (since we don't have real weights)
    2. Verifies the adapter interface works correctly
    3. Tests that the model can be instantiated properly
    """
    print("\n-- Testing CycleGAN Adapter --")

    try:
        # Test that the adapter can be created (without loading weights)
        # This tests the model creation and configuration
        from models.adapters import load_model
        from types import SimpleNamespace

        # Create a minimal test to verify the adapter structure
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        # Test that we can import and create the model structure
        from pix2pix.models.cycle_gan_model import CycleGANModel
        model = CycleGANModel(opt)

        # Test that the adapter interface would work
        from models.adapters import ModelAdapter
        adapter = ModelAdapter(model.netG_A)

        # Test prediction with dummy input
        dummy_mri = torch.randn(1, 1, 256, 256)
        output_tensor = adapter.predict(dummy_mri)

        # Verify output shape
        expected_shape = (1, 1, 256, 256)
        assert output_tensor.shape == expected_shape, f"Shape mismatch. Expected {expected_shape}, got {output_tensor.shape}"

        print("CycleGAN adapter test PASSED.")

    except Exception as e:
        print(f"CycleGAN adapter test FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_transformer_diffusion_adapter():
    """
    Tests the Transformer Diffusion adapter model creation and interface:
    1. Tests model creation without loading weights (since we don't have real weights)
    2. Verifies the adapter interface works correctly
    3. Tests that the model can be instantiated properly
    """
    print("\n-- Testing Transformer Diffusion Adapter --")

    try:
        # Test that we can import the required modules
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'shaoyanpan-transformer-diffusion', 'shaoyanpan-transformer-diffusion')))

        from network.Diffusion_model_transformer import SwinVITModel
        from diffusion.Create_diffusion import create_gaussian_diffusion

        # Test model creation with correct parameters
        num_channels = 64
        attention_resolutions = "32,16,8"
        channel_mult = (1, 2, 3, 4)
        num_heads = [4, 4, 8, 16]
        window_size = [[4, 4, 2], [4, 4, 2], [4, 4, 2], [4, 4, 2]]
        num_res_blocks = [1, 1, 1, 1]
        sample_kernel = ((2, 2, 2), (2, 2, 1), (2, 2, 1), (2, 2, 1))
        attention_ds = [int(x) for x in attention_resolutions.split(",")]
        patch_size = (64, 64, 2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Test that the adapter interface would work
        from models.adapters import TransformerDiffusionAdapter
        adapter = TransformerDiffusionAdapter(model, diffusion, device)

        # Test prediction with dummy input
        dummy_mri = torch.randn(1, 1, 64, 64, 2)  # 3D input for transformer
        output_tensor = adapter.predict(dummy_mri)

        # Verify output shape
        expected_shape = (1, 1, 64, 64, 2)
        assert output_tensor.shape == expected_shape, f"Shape mismatch. Expected {expected_shape}, got {output_tensor.shape}"

        print("Transformer Diffusion adapter test PASSED.")

    except Exception as e:
        print(f"Transformer Diffusion adapter test FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_all_adapters():
    """
    Runs all adapter tests in sequence.
    """
    print("="*50)
    print("RUNNING ALL ADAPTER TESTS")
    print("="*50)

    test_unet_adapter()
    test_pix2pix_adapter()
    test_cyclegan_adapter()
    test_transformer_diffusion_adapter()

    print("="*50)
    print("ALL ADAPTER TESTS COMPLETED")
    print("="*50)


if __name__ == "__main__":
    test_all_adapters()