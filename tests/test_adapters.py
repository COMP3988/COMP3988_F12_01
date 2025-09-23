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

from models.adapters import load_model
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
            
if __name__ == "__main__":
    test_unet_adapter()