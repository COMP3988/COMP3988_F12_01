"""
Pix2Pix Model for Medical Image Synthesis

This script trains a Pix2Pix model to convert MRI brain scans to synthetic CT scans.
This is useful for radiotherapy planning where we need CT scans for dose calculation
but want to avoid exposing patients to additional radiation from CT scans.

Input:  MRI brain scan (no radiation exposure)
Output: Synthetic CT brain scan (for radiotherapy planning)

The model uses a Generative Adversarial Network (GAN) approach:
- Generator: Creates fake CT from MRI input
- Discriminator: Judges whether CT is real or fake
- Both networks compete and improve together
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check if we have a GPU available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PairedData(Dataset):
    """
    Custom dataset class for loading paired MRI and CT images.
    
    This class handles the loading of image pairs where:
    - MRI images are the input (what we feed to the AI)
    - CT images are the target (what we want the AI to create)
    
    The dataset is organized in folders:
    - trainA/testA: CT images (target)
    - trainB/testB: MRI images (input)
    """
    
    def __init__(self, root_dir, transform=None, mode="train", max_samples=50):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.max_samples = max_samples
        
        # Set up the correct folder paths based on train/test mode
        if self.mode == "train":
            # For training: MRI is input, CT is target
            self.input_dir = os.path.join(root_dir, "trainB")  # MRI images (input)
            self.target_dir = os.path.join(root_dir, "trainA")  # CT images (target)
        elif self.mode == "test":
            # For testing: MRI is input, CT is target
            self.input_dir = os.path.join(root_dir, "testB")   # MRI images (input)
            self.target_dir = os.path.join(root_dir, "testA")  # CT images (target)
        else:
            raise ValueError("Mode must be 'train' or 'test'")
        
        # Get list of image files and limit to max_samples for quick demo
        self.input_images = sorted(os.listdir(self.input_dir))[:max_samples]
        self.target_images = sorted(os.listdir(self.target_dir))[:max_samples]
        
    def __len__(self):
        """Return the number of image pairs in the dataset"""
        return len(self.input_images)
    
    def __getitem__(self, idx):
        """Load and return a single MRI-CT image pair"""
        # Get the file paths for this image pair
        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        target_image_path = os.path.join(self.target_dir, self.target_images[idx])
        
        # Load the images and convert to RGB format
        mri_image = Image.open(input_image_path).convert("RGB")  # MRI input
        ct_image = Image.open(target_image_path).convert("RGB")  # CT target
        
        # Apply any transformations (resize, normalize, etc.)
        if self.transform:
            mri_image = self.transform(mri_image)
            ct_image = self.transform(ct_image)
            
        return mri_image, ct_image  # Return: (MRI input, CT target)

class SimpleGenerator(nn.Module):
    """
    The Generator Network - The "Artist" of our GAN
    
    This network takes an MRI image as input and tries to create a realistic CT image.
    It uses a U-Net like architecture with:
    - Encoder: Compresses the MRI image into a compact representation
    - Decoder: Expands the representation back into a CT image
    
    Think of it like an artist who learns to paint CT scans by looking at MRI scans.
    """
    
    def __init__(self):
        super(SimpleGenerator, self).__init__()
        
        # Encoder: Compress MRI image into smaller representation
        # Each layer makes the image smaller but captures more complex features
        self.encoder = nn.Sequential(
            # Layer 1: 256x256 -> 128x128, 3 channels -> 64 channels
            nn.Conv2d(3, 64, 4, 2, 1),      # Convolution: extract basic features
            nn.LeakyReLU(0.2),              # Activation: helps with learning
            
            # Layer 2: 128x128 -> 64x64, 64 channels -> 128 channels  
            nn.Conv2d(64, 128, 4, 2, 1),    # Extract more complex features
            nn.BatchNorm2d(128),            # Normalization: stabilizes training
            nn.LeakyReLU(0.2),
            
            # Layer 3: 64x64 -> 32x32, 128 channels -> 256 channels
            nn.Conv2d(128, 256, 4, 2, 1),   # Extract high-level features
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # Layer 4: 32x32 -> 16x16, 256 channels -> 512 channels
            nn.Conv2d(256, 512, 4, 2, 1),   # Final compression
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        # Decoder: Expand representation back into CT image
        # Each layer makes the image larger and more detailed
        self.decoder = nn.Sequential(
            # Layer 1: 16x16 -> 32x32, 512 channels -> 256 channels
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # Upsample and reduce channels
            nn.BatchNorm2d(256),
            nn.ReLU(),                               # Different activation for decoder
            
            # Layer 2: 32x32 -> 64x64, 256 channels -> 128 channels
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Layer 3: 64x64 -> 128x128, 128 channels -> 64 channels
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 4: 128x128 -> 256x256, 64 channels -> 3 channels (RGB)
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()  # Output between -1 and 1 (will be normalized later)
        )
        
    def forward(self, x):
        """
        Forward pass: MRI image -> CT image
        
        Args:
            x: MRI image tensor (batch_size, 3, 256, 256)
            
        Returns:
            Generated CT image tensor (batch_size, 3, 256, 256)
        """
        # Compress the MRI image into a compact representation
        encoded = self.encoder(x)
        
        # Expand the representation back into a CT image
        decoded = self.decoder(encoded)
        
        return decoded

class SimpleDiscriminator(nn.Module):
    """
    The Discriminator Network - The "Critic" of our GAN
    
    This network looks at a pair of images (MRI + CT) and decides whether the CT is real or fake.
    It's like an art critic who learns to spot the difference between real paintings and forgeries.
    
    The discriminator helps train the generator by providing feedback on how realistic
    the generated CT images look.
    """
    
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        
        # The discriminator takes 6 channels as input:
        # 3 channels from MRI + 3 channels from CT = 6 channels total
        self.model = nn.Sequential(
            # Layer 1: 256x256 -> 128x128, 6 channels -> 64 channels
            nn.Conv2d(6, 64, 4, 2, 1),      # Look at MRI+CT pair together
            nn.LeakyReLU(0.2),              # Activation function
            
            # Layer 2: 128x128 -> 64x64, 64 channels -> 128 channels
            nn.Conv2d(64, 128, 4, 2, 1),    # Extract more complex patterns
            nn.BatchNorm2d(128),            # Normalization for stable training
            nn.LeakyReLU(0.2),
            
            # Layer 3: 64x64 -> 32x32, 128 channels -> 256 channels
            nn.Conv2d(128, 256, 4, 2, 1),   # Look for high-level features
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # Layer 4: 32x32 -> 16x16, 256 channels -> 512 channels
            nn.Conv2d(256, 512, 4, 2, 1),   # Final feature extraction
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # Layer 5: 16x16 -> 16x16, 512 channels -> 1 channel (decision)
            nn.Conv2d(512, 1, 4, 1, 1),     # Make final decision
            nn.Sigmoid()                    # Output between 0 (fake) and 1 (real)
        )
        
    def forward(self, x):
        """
        Forward pass: MRI+CT pair -> real/fake decision
        
        Args:
            x: Concatenated MRI+CT tensor (batch_size, 6, 256, 256)
            
        Returns:
            Real/fake score tensor (batch_size, 1, 16, 16)
            Values close to 1 = "looks real"
            Values close to 0 = "looks fake"
        """
        return self.model(x)

def train_mri_to_ct():
    """
    Main training function for the MRI to CT conversion model.
    
    This function sets up the data, models, and training loop to teach the AI
    how to convert MRI brain scans into synthetic CT scans.
    """
    
    # resize images and convert to numbers the ai can understand
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # load dataset - using small samples for quick demo
    root_dir = "../Dataset/images"
    train_data = PairedData(root_dir=root_dir, transform=transform, mode='train', max_samples=20)
    test_data = PairedData(root_dir=root_dir, transform=transform, mode='test', max_samples=10)
    
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=2, shuffle=False)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print("Model: MRI → CT (for radiotherapy planning)")
    
    # create the two ai models - generator makes fake cts, discriminator judges them
    generator = SimpleGenerator().to(device)        
    discriminator = SimpleDiscriminator().to(device)  
    
    # loss functions and optimizers
    criterion_l1 = nn.L1Loss()    # pixel accuracy loss
    criterion_bce = nn.BCELoss()  # adversarial loss
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    num_epochs = 5
    lambda_l1 = 100
    
    print("Starting MRI → CT training...")
    
    # training loop
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        for i, (mri_images, ct_images) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            mri_images = mri_images.to(device)
            ct_images = ct_images.to(device)
            
            # train discriminator - teach it to tell real from fake
            optimizer_d.zero_grad()
            
            # real pairs
            real_pairs = torch.cat([mri_images, ct_images], dim=1)
            real_output = discriminator(real_pairs)
            real_labels = torch.ones_like(real_output)
            d_loss_real = criterion_bce(real_output, real_labels)
            
            # fake pairs
            fake_ct = generator(mri_images)
            fake_pairs = torch.cat([mri_images, fake_ct.detach()], dim=1)
            fake_output = discriminator(fake_pairs)
            fake_labels = torch.zeros_like(fake_output)
            d_loss_fake = criterion_bce(fake_output, fake_labels)
            
            d_loss = (d_loss_real + d_loss_fake) * 0.5
            d_loss.backward()
            optimizer_d.step()
            
            # train generator - teach it to fool the discriminator
            optimizer_g.zero_grad()
            
            fake_ct = generator(mri_images)
            fake_pairs = torch.cat([mri_images, fake_ct], dim=1)
            fake_output = discriminator(fake_pairs)
            
            g_loss_adv = criterion_bce(fake_output, real_labels)
            g_loss_l1 = criterion_l1(fake_ct, ct_images)
            g_loss = g_loss_adv + lambda_l1 * g_loss_l1
            
            g_loss.backward()
            optimizer_g.step()
            
            if i % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(train_loader)}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
        
        # save results after each epoch
        generator.eval()
        with torch.no_grad():
            mri_sample, ct_sample = next(iter(test_loader))
            mri_sample = mri_sample.to(device)
            fake_ct = generator(mri_sample)
            save_mri_to_ct_results(mri_sample, ct_sample, fake_ct, epoch + 1)
    
    print("MRI → CT training completed!")
    return generator, discriminator

def save_mri_to_ct_results(mri_images, real_ct, fake_ct, epoch):
    """
    Save visualization of the model's results after each training epoch.
    
    This function creates a comparison image showing:
    - Row 1: Original MRI input
    - Row 2: Real CT target  
    - Row 3: Generated CT output
    
    This helps us see how well the model is learning to convert MRI to CT.
    """
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Convert normalized images back to displayable format
    def denormalize(tensor):
        """Convert from [-1, 1] range back to [0, 1] range for display"""
        return (tensor + 1) / 2
    
    # Denormalize all images for proper display
    mri_images = denormalize(mri_images)
    real_ct = denormalize(real_ct)
    fake_ct = denormalize(fake_ct)
    
    # Create a figure with 3 rows and 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    
    # Display results for 2 sample images
    for i in range(2):
        # Row 1: Original MRI input
        axes[0, i].imshow(mri_images[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].set_title("MRI Input")
        axes[0, i].axis('off')
        
        # Row 2: Real CT target
        axes[1, i].imshow(real_ct[i].permute(1, 2, 0).cpu().numpy())
        axes[1, i].set_title("Real CT")
        axes[1, i].axis('off')
        
        # Row 3: Generated CT output
        axes[2, i].imshow(fake_ct[i].permute(1, 2, 0).cpu().numpy())
        axes[2, i].set_title("Generated CT")
        axes[2, i].axis('off')
    
    # Add overall title and save the figure
    plt.suptitle(f"MRI → CT Conversion (Epoch {epoch})", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"results/mri_to_ct_epoch_{epoch}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"MRI → CT results saved for epoch {epoch}")

if __name__ == "__main__":
    # check if dataset exists
    if not os.path.exists("../Dataset/images"):
        print("Error: Dataset not found at ../Dataset/images")
        print("Please make sure the dataset is in the correct location")
        exit(1)
    
    print("Starting Pix2Pix MRI to CT training...")
    print("This will train a model to convert MRI brain scans to synthetic CT scans.")
    print()
    
    # start training
    generator, discriminator = train_mri_to_ct()
    
    # save models
    print("\nSaving trained models...")
    torch.save(generator.state_dict(), "mri_to_ct_generator.pth")
    torch.save(discriminator.state_dict(), "mri_to_ct_discriminator.pth")
    print("Models saved!")
    print("\nTraining completed!")
    print("Check the 'results' folder to see the generated images.")
