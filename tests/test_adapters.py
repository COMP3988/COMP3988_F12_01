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
    Tests the Transformer Diffusion adapter by using infer_single.py directly:
    1. Creates a dummy checkpoint file
    2. Creates a dummy input file
    3. Calls infer_single.py as a subprocess
    4. Verifies the output file is created
    5. Cleans up temporary files
    """
    print("\n-- Testing Transformer Diffusion Adapter (via infer_single.py) --")

    import subprocess
    import tempfile
    import numpy as np
    from pathlib import Path

    try:
        # Create temporary directory for test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # 1. Create a dummy checkpoint file
            dummy_ckpt_path = temp_path / "dummy_transformer_diffusion.pth"

            # Import the model and create a dummy state dict
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'shaoyanpan-transformer-diffusion')))
            from config import get_model_config
            from network.Diffusion_model_transformer import SwinVITModel

            # Create model with config
            model = SwinVITModel(**get_model_config())
            torch.save(model.state_dict(), dummy_ckpt_path)

            # 2. Create a dummy input file (.npz format)
            dummy_input_path = temp_path / "dummy_input.npz"

            # Create dummy MRI and CT volumes
            mri_volume = np.random.randn(64, 64, 2).astype(np.float32)
            ct_volume = np.random.randn(64, 64, 2).astype(np.float32)

            # Save as .npz file
            np.savez(dummy_input_path, image=mri_volume, label=ct_volume)

            # 3. Create output path
            dummy_output_path = temp_path / "dummy_output.mha"

            # 4. Call infer_single.py as subprocess
            infer_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'shaoyanpan-transformer-diffusion', 'infer_single.py'))

            cmd = [
                'python', infer_script_path,
                '--ckpt', str(dummy_ckpt_path),
                '--input', str(dummy_input_path),
                '--output', str(dummy_output_path),
                '--sanity'  # Use minimal settings for quick testing
            ]

            print(f"Running command: {' '.join(cmd)}")

            # Run the inference
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            # 5. Verify the output
            if result.returncode == 0:
                if dummy_output_path.exists():
                    print("✅ Transformer Diffusion inference completed successfully!")
                    print(f"✅ Output file created: {dummy_output_path}")
                    print("✅ Transformer Diffusion adapter test PASSED.")
                else:
                    print("❌ Output file was not created")
                    print(f"❌ Transformer Diffusion adapter test FAILED: No output file")
            else:
                print(f"❌ Inference failed with return code: {result.returncode}")
                print(f"❌ STDOUT: {result.stdout}")
                print(f"❌ STDERR: {result.stderr}")
                print("❌ Transformer Diffusion adapter test FAILED: Subprocess error")

    except subprocess.TimeoutExpired:
        print("❌ Transformer Diffusion adapter test FAILED: Timeout")
    except Exception as e:
        print(f"❌ Transformer Diffusion adapter test FAILED: {e}")
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