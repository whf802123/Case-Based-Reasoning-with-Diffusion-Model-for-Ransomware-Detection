import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.utils as vutils


IMAGE_SIZE = 16        # Size
CHANNELS = 3
T = 1000                # Diffusion steps
BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
DATA_DIR = "D://Data/Malware_images_16/"     # D://Data/Benign_images_16/   D:\Data\isot_Benign_images               D://Data/Malware_images_16/    D:\Data\isot_Malware_images
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


betas = torch.linspace(1e-4, 0.005, T, device=DEVICE)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


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


class UNet2DModel(nn.Module):
    def __init__(self, sample_size, in_channels, out_channels, layers_per_block,
                 block_out_channels, down_block_types, up_block_types):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        return type("Output", (), {"sample": self.conv(x)})


class DiffusionModelWithUNet(nn.Module):
    def __init__(self, image_size=32, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(base_channels, base_channels * 2, base_channels * 4, base_channels * 8),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
        )

    def forward(self, x, t):
        t = t.squeeze()
        return self.unet(x, t).sample


def forward_diffusion(x0, t):
    batch_size = x0.shape[0]
    alpha_bar_t = alpha_bars[t].view(batch_size, 1, 1, 1)
    noise = torch.randn_like(x0)
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
    return xt, noise

# Training
def train_diffusion(model, dataloader, optimizer, num_epochs=NUM_EPOCHS):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            t = torch.randint(0, T, (batch.shape[0],), device=DEVICE)
            xt, noise = forward_diffusion(batch, t)
            t_norm = t.float() / T
            noise_pred = model(xt, t_norm)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.6f}")


@torch.no_grad()
def sample(model, num_samples):
    model.eval()
    x = torch.randn(num_samples, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    for t in reversed(range(T)):
        t_tensor = torch.ones(num_samples, device=DEVICE) * (t / T)
        noise_pred = model(x, t_tensor)
        alpha_bar_t = alpha_bars[t].view(1, 1, 1, 1).expand(num_samples, 1, 1, 1)
        beta_t = betas[t].view(1, 1, 1, 1).expand(num_samples, 1, 1, 1)
        x = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        if t > 0:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(beta_t) * noise
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    x = (x * 255).to(torch.uint8)
    return x


if __name__ == "__main__":
    dataset = RGBImageDataset(DATA_DIR)
    num_samples_to_use = min(10000, len(dataset))
    subset_indices = list(range(num_samples_to_use))
    subset = Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

    model = DiffusionModelWithUNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_diffusion(model, dataloader, optimizer, num_epochs=NUM_EPOCHS)

    num_samples = 10
    synthetic_images = sample(model, num_samples)

    output_folder = "USTC_Malware_images_synthetic"        # USTC_Benign_images_synthetic   USTC_Malware_images_synthetic          ISOT_Benign_images_synthetic   ISOT_Malware_images_synthetic
    os.makedirs(output_folder, exist_ok=True)
    for i, img in enumerate(synthetic_images):
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_path = os.path.join(output_folder, f"synthetic_image_{i+1}.png")
        cv2.imwrite(img_path, img_bgr)
    print(f"Synthetic images saved to folder: {output_folder}")
