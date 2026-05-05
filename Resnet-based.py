import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import glob
import os
import cv2
import numpy as np

# Parameters
IMAGE_SIZE = 16
CHANNELS = 3
T = 1000
BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5

DATA_DIR = "D://Data/Malware_images_16/"
OUTPUT_FOLDER = "USTC_Malware_images_resunet_synthetic"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
betas = torch.linspace(1e-4, 0.01, T, device=DEVICE)
alphas = 1.0 - betas
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
        # normalize to [-1, 1]
        img = img * 2.0 - 1.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img)

# Residual Block with Time Embedding
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(self.act1(self.norm1(x)))
        time_feature = self.time_proj(temb).view(temb.shape[0], -1, 1, 1)
        h = h + time_feature
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)

# ResUNet Diffusion Model
class DiffusionModelWithResUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=64,
        time_dim=256
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        # 16x16
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        # Encoder
        # 16x16 -> 8x8
        self.enc1 = ResBlock(base_channels, base_channels, time_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        # 8x8 -> 4x4
        self.enc2 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        # 4x4 -> 2x2
        self.enc3 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.down3 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1)
        # Bottleneck
        self.mid1 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        # Decoder
        # 2x2 -> 4x4
        self.up3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.dec3 = ResBlock(base_channels * 8, base_channels * 4, time_dim)
        # 4x4 -> 8x8
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResBlock(base_channels * 4, base_channels * 2, time_dim)
        # 8x8 -> 16x16
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResBlock(base_channels * 2, base_channels, time_dim)
        self.final_norm = nn.GroupNorm(8, base_channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, t):
        # t: [B], normalized to [0, 1]
        if t.dim() == 1:
            t = t.view(-1, 1)
        temb = self.time_embed(t.float())
        x = self.init_conv(x)
        # Encoder
        e1 = self.enc1(x, temb)          # [B, 64, 16, 16]
        x = self.down1(e1)              # [B, 128, 8, 8]
        e2 = self.enc2(x, temb)         # [B, 128, 8, 8]
        x = self.down2(e2)              # [B, 256, 4, 4]
        e3 = self.enc3(x, temb)         # [B, 256, 4, 4]
        x = self.down3(e3)              # [B, 256, 2, 2]
        # Bottleneck
        x = self.mid1(x, temb)
        x = self.mid2(x, temb)
        # Decoder
        x = self.up3(x)                 # [B, 256, 4, 4]
        x = torch.cat([x, e3], dim=1)   # [B, 512, 4, 4]
        x = self.dec3(x, temb)
        x = self.up2(x)                 # [B, 128, 8, 8]
        x = torch.cat([x, e2], dim=1)   # [B, 256, 8, 8]
        x = self.dec2(x, temb)
        x = self.up1(x)                 # [B, 64, 16, 16]
        x = torch.cat([x, e1], dim=1)   # [B, 128, 16, 16]
        x = self.dec1(x, temb)
        x = self.final_conv(self.final_act(self.final_norm(x)))
        return x

# Forward Diffusion
def forward_diffusion(x0, t):
    batch_size = x0.shape[0]
    alpha_bar_t = alpha_bars[t].view(batch_size, 1, 1, 1)
    noise = torch.randn_like(x0)
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
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
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {avg_loss:.6f}, "
            f"LR: {current_lr:.8f}"
        )

@torch.no_grad()
def sample(model, num_samples):
    model.eval()
    x = torch.randn(num_samples, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    for t in reversed(range(T)):
        t_tensor = torch.ones(num_samples, device=DEVICE) * (t / T)
        noise_pred = model(x, t_tensor)
        alpha_t = alphas[t].view(1, 1, 1, 1)
        alpha_bar_t = alpha_bars[t].view(1, 1, 1, 1)
        beta_t = betas[t].view(1, 1, 1, 1)
        x = (1.0 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * noise_pred
        )
        if t > 0:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(beta_t) * noise
    x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0.0, 1.0)
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
            img_path = os.path.join(
                output_folder,
                f"synthetic_resunet_{saved_count + i + 1}.png"
            )
            cv2.imwrite(img_path, img_bgr)
        saved_count += current_batch_size
        print(f"Saved {saved_count}/{total_samples} synthetic images")

if __name__ == "__main__":
    dataset = RGBImageDataset(DATA_DIR)
    num_samples_to_use = min(10000, len(dataset))
    subset = Subset(dataset, list(range(num_samples_to_use)))
    dataloader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if DEVICE.type == "cuda" else False
    )
    model = DiffusionModelWithResUNet(
        in_channels=CHANNELS,
        out_channels=CHANNELS,
        base_channels=64,
        time_dim=256
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=3e-6
    )
    train_diffusion(
        model,
        dataloader,
        optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS
    )
    total_samples = 10
    generate_and_save_images(
        model=model,
        total_samples=total_samples,
        output_folder=OUTPUT_FOLDER,
        sample_batch_size=2
    )
    print(f"Synthetic images saved to folder: {OUTPUT_FOLDER}")
