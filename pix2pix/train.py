"""
MRI to CT Pix2Pix Demo - Correct direction for radiotherapy planning
Input: MRI brain scan (no radiation)
Output: Synthetic CT brain scan (for radiotherapy planning)
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PairedData(Dataset):
    def __init__(self, root_dir, transform=None, mode="train", max_samples=50):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.max_samples = max_samples
        
        if self.mode == "train":
            # FLIPPED: MRI is input, CT is target
            self.input_dir = os.path.join(root_dir, "trainB")  # MRI (input)
            self.target_dir = os.path.join(root_dir, "trainA")  # CT (target)
        elif self.mode == "test":
            # FLIPPED: MRI is input, CT is target
            self.input_dir = os.path.join(root_dir, "testB")   # MRI (input)
            self.target_dir = os.path.join(root_dir, "testA")  # CT (target)
        else:
            raise ValueError("Mode must be 'train' or 'test'")
        
        self.input_images = sorted(os.listdir(self.input_dir))[:max_samples]
        self.target_images = sorted(os.listdir(self.target_dir))[:max_samples]
        
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):
        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        target_image_path = os.path.join(self.target_dir, self.target_images[idx])
        
        # FLIPPED: MRI is input, CT is target
        mri_image = Image.open(input_image_path).convert("RGB")  # MRI input
        ct_image = Image.open(target_image_path).convert("RGB")  # CT target
        
        if self.transform:
            mri_image = self.transform(mri_image)
            ct_image = self.transform(ct_image)
        return mri_image, ct_image  # MRI → CT

class SimpleGenerator(nn.Module):
    def __init__(self):
        super(SimpleGenerator, self).__init__()
        
        # Simple U-Net like architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 4, 2, 1),  # 6 channels: 3 MRI + 3 CT
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)

def train_mri_to_ct():
    # Data setup with small sample for quick demo
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    root_dir = "../Dataset/images"
    train_data = PairedData(root_dir=root_dir, transform=transform, mode='train', max_samples=20)
    test_data = PairedData(root_dir=root_dir, transform=transform, mode='test', max_samples=10)
    
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=2, shuffle=False)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print("Model: MRI → CT (for radiotherapy planning)")
    
    # Initialize models
    generator = SimpleGenerator().to(device)
    discriminator = SimpleDiscriminator().to(device)
    
    # Loss functions and optimizers
    criterion_l1 = nn.L1Loss()
    criterion_bce = nn.BCELoss()
    
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Training parameters
    num_epochs = 5  # Very small for demo
    lambda_l1 = 100
    
    print("Starting MRI → CT training...")
    
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        for i, (mri_images, ct_images) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            mri_images = mri_images.to(device)  # MRI input
            ct_images = ct_images.to(device)    # CT target
            
            # Train Discriminator
            optimizer_d.zero_grad()
            
            # Real pairs (MRI + real CT)
            real_pairs = torch.cat([mri_images, ct_images], dim=1)
            real_output = discriminator(real_pairs)
            real_labels = torch.ones_like(real_output)
            d_loss_real = criterion_bce(real_output, real_labels)
            
            # Fake pairs (MRI + fake CT)
            fake_ct = generator(mri_images)  # MRI → CT
            fake_pairs = torch.cat([mri_images, fake_ct.detach()], dim=1)
            fake_output = discriminator(fake_pairs)
            fake_labels = torch.zeros_like(fake_output)
            d_loss_fake = criterion_bce(fake_output, fake_labels)
            
            d_loss = (d_loss_real + d_loss_fake) * 0.5
            d_loss.backward()
            optimizer_d.step()
            
            # Train Generator
            optimizer_g.zero_grad()
            
            fake_ct = generator(mri_images)  # MRI → CT
            fake_pairs = torch.cat([mri_images, fake_ct], dim=1)
            fake_output = discriminator(fake_pairs)
            
            # Generator loss: adversarial + L1
            g_loss_adv = criterion_bce(fake_output, real_labels)
            g_loss_l1 = criterion_l1(fake_ct, ct_images)
            g_loss = g_loss_adv + lambda_l1 * g_loss_l1
            
            g_loss.backward()
            optimizer_g.step()
            
            if i % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(train_loader)}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
        
        # Save sample results every epoch
        generator.eval()
        with torch.no_grad():
            mri_sample, ct_sample = next(iter(test_loader))
            mri_sample = mri_sample.to(device)
            fake_ct = generator(mri_sample)  # MRI → CT
            
            # Save results
            save_mri_to_ct_results(mri_sample, ct_sample, fake_ct, epoch + 1)
    
    print("MRI → CT training completed!")
    return generator, discriminator

def save_mri_to_ct_results(mri_images, real_ct, fake_ct, epoch):
    """Save sample results for visualization"""
    os.makedirs("results", exist_ok=True)
    
    # Denormalize images
    def denormalize(tensor):
        return (tensor + 1) / 2
    
    mri_images = denormalize(mri_images)
    real_ct = denormalize(real_ct)
    fake_ct = denormalize(fake_ct)
    
    # Create comparison image
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    
    for i in range(2):
        axes[0, i].imshow(mri_images[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].set_title("MRI Input")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(real_ct[i].permute(1, 2, 0).cpu().numpy())
        axes[1, i].set_title("Real CT")
        axes[1, i].axis('off')
        
        axes[2, i].imshow(fake_ct[i].permute(1, 2, 0).cpu().numpy())
        axes[2, i].set_title("Generated CT")
        axes[2, i].axis('off')
    
    plt.suptitle(f"MRI → CT Conversion (Epoch {epoch})", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"results/mri_to_ct_epoch_{epoch}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"MRI → CT results saved for epoch {epoch}")

if __name__ == "__main__":
    # Check if dataset exists
    if not os.path.exists("../Dataset/images"):
        print("Error: Dataset not found at ../Dataset/images")
        print("Please make sure the dataset is in the correct location")
        exit(1)
    
    # Start training
    generator, discriminator = train_mri_to_ct()
    
    # Save final models
    torch.save(generator.state_dict(), "mri_to_ct_generator.pth")
    torch.save(discriminator.state_dict(), "mri_to_ct_discriminator.pth")
    print("MRI → CT models saved!")
