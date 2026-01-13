import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.utils as vutils

IMAGE_SIZE = 16
CHANNELS = 3
BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
LATENT_DIM = 100
DATA_DIR = "D://Data/Malware_images_16/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RGBImageDataset(Dataset):
    def __init__(self, folder):
        self.files = glob.glob(os.path.join(folder, "**/*.png"), recursive=True)
        if not self.files:
            print(f"Warning: No PNG files found in folder {folder}!")
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        if img is None:
            raise ValueError(f"Failed to load image: {self.files[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0
        img = img * 2 - 1
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)

class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=3, feature_map_size=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_map_size * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, channels=3, feature_map_size=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, feature_map_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_size, feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(feature_map_size * 2 * 4 * 4, 1),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)

def train_gan(generator, discriminator, dataloader, optimizer_G, optimizer_D, num_epochs):
    criterion = nn.BCELoss()
    generator.train()
    discriminator.train()
    for epoch in range(num_epochs):
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        for real_images in dataloader:
            real_images = real_images.to(DEVICE)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1, device=DEVICE)
            fake_labels = torch.zeros(batch_size, 1, device=DEVICE)
            optimizer_D.zero_grad()
            output_real = discriminator(real_images)
            d_loss_real = criterion(output_real, real_labels)
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(output_fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            optimizer_G.zero_grad()
            output_fake_for_G = discriminator(fake_images)
            g_loss = criterion(output_fake_for_G, real_labels)
            g_loss.backward()
            optimizer_G.step()
            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}] Generator Loss: {g_loss_epoch/len(dataloader):.6f} | Discriminator Loss: {d_loss_epoch/len(dataloader):.6f}")

@torch.no_grad()
def sample_gan(generator, num_samples):
    generator.eval()
    noise = torch.randn(num_samples, LATENT_DIM, 1, 1, device=DEVICE)
    synthetic_images = generator(noise)
    synthetic_images = (synthetic_images + 1) / 2
    synthetic_images = torch.clamp(synthetic_images, 0, 1)
    synthetic_images = (synthetic_images * 255).to(torch.uint8)
    return synthetic_images

if __name__ == "__main__":
    dataset = RGBImageDataset(DATA_DIR)
    num_samples_to_use = min(10000, len(dataset))
    subset_indices = list(range(num_samples_to_use))
    subset = Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    generator = Generator(latent_dim=LATENT_DIM, channels=CHANNELS).to(DEVICE)
    discriminator = Discriminator(channels=CHANNELS).to(DEVICE)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    train_gan(generator, discriminator, dataloader, optimizer_G, optimizer_D, num_epochs=NUM_EPOCHS)
    num_samples = 10
    synthetic_images = sample_gan(generator, num_samples)
    output_folder = "USTC_Malware_images_synthetic_GAN"
    os.makedirs(output_folder, exist_ok=True)
    for i, img in enumerate(synthetic_images):
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_path = os.path.join(output_folder, f"synthetic_image_{i+1}.png")
        cv2.imwrite(img_path, img_bgr)
    print(f"Synthetic images saved to folder: {output_folder}")
