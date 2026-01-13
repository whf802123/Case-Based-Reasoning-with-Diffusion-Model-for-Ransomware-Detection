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
NUM_EPOCHS = 2000
LEARNING_RATE = 1e-6
LATENT_DIM = 128
BETA = 0.1
DATA_DIR = "D:\Data\isot_Malware_images"     # D://Data/Malware_images_16/   D://Data/Benign_images_16/          D:\Data\isot_Malware_images  D:\Data\isot_Benign_images
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

class VAE(nn.Module):
    def __init__(self, image_size=16, channels=3, latent_dim=64, base_channels=64):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(base_channels * 2 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(base_channels * 2 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, base_channels * 2 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (base_channels * 2, 4, 4)),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z):
        h = self.fc_decode(z)
        x_recon = self.decoder(h)
        return x_recon
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss(x_recon, x, mu, logvar):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * BETA * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

def train_vae(model, dataloader, optimizer, num_epochs=NUM_EPOCHS):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(batch)
            loss = vae_loss(x_recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader.dataset):.6f}")

@torch.no_grad()
def sample_vae(model, num_samples):
    model.eval()
    z = torch.randn(num_samples, model.fc_mu.out_features, device=DEVICE)
    samples = model.decode(z)
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    samples = (samples * 255).to(torch.uint8)
    return samples

if __name__ == "__main__":
    dataset = RGBImageDataset(DATA_DIR)
    num_samples_to_use = min(10000, len(dataset))
    subset_indices = list(range(num_samples_to_use))
    subset = Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    model = VAE(image_size=IMAGE_SIZE, channels=CHANNELS, latent_dim=LATENT_DIM, base_channels=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_vae(model, dataloader, optimizer, num_epochs=NUM_EPOCHS)
    num_samples = 10
    synthetic_images = sample_vae(model, num_samples)
    output_folder = "ISOT_Malware_images_synthetic_VAE"
    os.makedirs(output_folder, exist_ok=True)
    for i, img in enumerate(synthetic_images):
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_path = os.path.join(output_folder, f"synthetic_image_{i+1}.png")
        cv2.imwrite(img_path, img_bgr)
    print(f"Synthetic images saved to folder: {output_folder}")
