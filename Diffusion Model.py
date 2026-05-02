import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

IMAGE_SIZE = 16
CHANNELS = 3
T = 1000                                # Time step 
BATCH_SIZE = 32
NUM_EPOCHS = 70
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5
DATA_DIR = "D://Data/Benign_images_16/"  # D://Data/Malware_images_16/   D://Data/Benign_images_16/          D:\Data\isot_Malware_images  D:\Data\isot_Benign_images
OUTPUT_FOLDER = "USTC_Benign_images_synthetic"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

betas = torch.linspace(1e-4, 0.01, T, device=DEVICE)
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

class ComplexUNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, num_down=3):
        super().__init__()
        time_dim = base_channels * 4
        self.time_embed = nn.Sequential(nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))

      # Downsampling 
        self.downs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.down_time_projs = nn.ModuleList()
        curr_channels = in_channels

        for i in range(num_down - 1):
            out_ch = base_channels * (2 ** i)
            block = nn.Sequential(
                nn.Conv2d(curr_channels, out_ch, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU()
            )
            self.downs.append(block)
            self.skip_convs.append(nn.Conv2d(out_ch, out_ch, kernel_size=1))
            self.down_time_projs.append(nn.Linear(time_dim, out_ch))
            curr_channels = out_ch
        bottleneck_channels = base_channels * (2 ** (num_down - 1))

      # Bottleneck 
        self.down_bottleneck = nn.Sequential(
            nn.Conv2d(curr_channels, bottleneck_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, bottleneck_channels),
            nn.SiLU(),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, bottleneck_channels),
            nn.SiLU()
        )

        self.down_bottleneck_time_proj = nn.Linear(time_dim, bottleneck_channels)
        curr_channels = bottleneck_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(curr_channels, curr_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, curr_channels),
            nn.SiLU()
        )
        self.bottleneck_time_proj = nn.Linear(time_dim, curr_channels)

      # Upsampling block 
        self.ups = nn.ModuleList()
        self.up_time_projs = nn.ModuleList()

        for i in reversed(range(num_down - 1)):
            in_ch = curr_channels
            out_ch = base_channels * (2 ** i)
            block = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.SiLU()
            )
            self.ups.append(block)
            self.up_time_projs.append(nn.Linear(time_dim, out_ch))
            curr_channels = out_ch

        self.final_up = nn.ConvTranspose2d(curr_channels, curr_channels, kernel_size=4, stride=2, padding=1)
        self.final_time_proj = nn.Linear(time_dim, curr_channels)
        self.final_conv = nn.Conv2d(curr_channels, out_channels, kernel_size=1)

    def add_time_embedding(self, x, temb, proj):
        time_feature = proj(temb).view(x.shape[0], x.shape[1], 1, 1)
        return x + time_feature

    def forward(self, x, t):
        t = t.view(-1, 1).float()
        temb = self.time_embed(t)
        skips = []

        for down, skip_conv, time_proj in zip(self.downs, self.skip_convs, self.down_time_projs):
            x = down(x)
            x = self.add_time_embedding(x, temb, time_proj)
            skips.append(skip_conv(x))
        x = self.down_bottleneck(x)
        x = self.add_time_embedding(x, temb, self.down_bottleneck_time_proj)
        x = self.bottleneck(x)
        x = self.add_time_embedding(x, temb, self.bottleneck_time_proj)
        for up, time_proj in zip(self.ups, self.up_time_projs):
            x = up(x)
            x = self.add_time_embedding(x, temb, time_proj)
            skip = skips.pop()
            skip = F.interpolate(skip, size=x.shape[-2:], mode="nearest")
            x = x + skip
        x = self.final_up(x)
        x = self.add_time_embedding(x, temb, self.final_time_proj)
        x = self.final_conv(x)
        return type("Output", (), {"sample": x})

class DiffusionModelWithComplexUNet(nn.Module):
    def __init__(self, image_size=16, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        self.unet = ComplexUNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels, num_down=3)
    def forward(self, x, t):
        t = t.squeeze()
        return self.unet(x, t).sample

def forward_diffusion(x0, t):
    batch_size = x0.shape[0]
    alpha_bar_t = alpha_bars[t].view(batch_size, 1, 1, 1)
    noise = torch.randn_like(x0)
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
    return xt, noise

def train_diffusion(model, dataloader, optimizer, scheduler=None, num_epochs=NUM_EPOCHS):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        if scheduler is not None:
            scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}, LR: {current_lr:.8f}")

@torch.no_grad()
def sample(model, num_samples):
    model.eval()
    x = torch.randn(num_samples, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    for t in reversed(range(T)):
        t_tensor = torch.ones(num_samples, device=DEVICE) * (t / T)
        noise_pred = model(x, t_tensor)
        alpha_t = alphas[t].view(1, 1, 1, 1).expand(num_samples, 1, 1, 1)
        alpha_bar_t = alpha_bars[t].view(1, 1, 1, 1).expand(num_samples, 1, 1, 1)
        beta_t = betas[t].view(1, 1, 1, 1).expand(num_samples, 1, 1, 1)
        x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred)
        if t > 0:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(beta_t) * noise
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    x = (x * 255).to(torch.uint8)
    return x

@torch.no_grad()
def generate_and_save_images(model, total_samples, output_folder, sample_batch_size=2):
    model.eval()
    os.makedirs(output_folder, exist_ok=True)
    saved_count = 0

    while saved_count < total_samples:
        current_batch_size = min(sample_batch_size, total_samples - saved_count)
        synthetic_images = sample(model, current_batch_size)
        for i, img in enumerate(synthetic_images):
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_path = os.path.join(output_folder, f"synthetic_image_{saved_count + i + 1}.png")
            cv2.imwrite(img_path, img_bgr)
        saved_count += current_batch_size
        print(f"Saved {saved_count}/{total_samples} synthetic images")

if __name__ == "__main__":
    dataset = RGBImageDataset(DATA_DIR)
    num_samples_to_use = min(10000, len(dataset))
    subset_indices = list(range(num_samples_to_use))
    subset = Subset(dataset, subset_indices)
    dataloader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    model = DiffusionModelWithComplexUNet(image_size=IMAGE_SIZE, in_channels=CHANNELS, out_channels=CHANNELS, base_channels=64).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=3e-6)
    train_diffusion(model, dataloader, optimizer, scheduler=scheduler, num_epochs=NUM_EPOCHS)
    total_samples = 10000              # number of samples
    generate_and_save_images(model=model, total_samples=total_samples, output_folder=OUTPUT_FOLDER, sample_batch_size=2)
    print(f"Synthetic images saved to folder: {OUTPUT_FOLDER}")

