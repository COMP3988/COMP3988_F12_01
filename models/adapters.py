
import torch
from types import SimpleNamespace

from unet.model.unet_model import UNet
from pix2pix.models.pix2pix_model import Pix2PixModel

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
    
def load_model(model_name: str, weights_path: str) -> ModelAdapter:
    """ 
    A function that loads a model by name, applies its training weights, 
    and returns it wrapped in a standardised ModelAdapter.
    
    Args:
        model_name (str): The name of the model to load.
        weight_path (str): The file path to the saved .pth model weights.
        
    Returns:
        ModelAdapter: The loaded and wrapped model, ready for inference.
    """
    # U-Net
    if model_name.lower() == 'unet':
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
        model.load_state_dict(torch.load(weights_path, weights_only=False))
        print("U-Net model loaded.")
        return ModelAdapter(model)
    
    # Pix2Pix
    elif model_name.lower() == 'pix2pix':
        opt = SimpleNamespace(
            input_nc=3,
            output_nc=3,
            ngf=64,
            netG='unet_256',
            norm='batch',
            no_dropout=False,
            init_type='normal',
            init_gain=0.02,
            isTrain=False,
            device='cpu'
        )
        model = Pix2PixModel(opt)
        state_dict = torch.load(weights_path, map_location=opt.device)
        model.netG.load_state_dict(state_dict)
        print("Pix2Pix model loaded.")
        return ModelAdapter(model.netG)
    else:
        raise ValueError(f"Unknown model name: {model_name}")