import argparse, yaml, torch, os
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from scipy import linalg
from tqdm import tqdm
from models.model import DiffusionModel

# ------------------------------
# LeNet Feature Extractor
# ------------------------------
import torch.nn as nn

class LeNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 256), nn.ReLU(),
        )

    def forward(self, x):
        return self.features(x)

def get_features(dataloader, model, device):
    model.eval()
    features = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            feat = model(x).cpu().numpy()
            features.append(feat)
    return np.concatenate(features, axis=0)

def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)

def compute_fid(real_loader, fake_loader, device):
    model = LeNetFeatureExtractor().to(device)
    real_feats = get_features(real_loader, model, device)
    fake_feats = get_features(fake_loader, model, device)

    mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_f, sigma_f = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)
    return calculate_fid(mu_r, sigma_r, mu_f, sigma_f)

# ------------------------------
# Evaluation Script Entry
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--ckpt",   required=True, help="Path to checkpoint (.pt)")
parser.add_argument("--n_gen",  type=int, default=5000)
args = parser.parse_args()
cfg = yaml.safe_load(open(args.config))

device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

# ---------- Load Real Images ----------
tfm = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
real_ds = MNIST(cfg["data_root"], train=False, download=True, transform=tfm)
real_loader = DataLoader(real_ds, batch_size=128, shuffle=False)

# ---------- Load Model ----------
model = DiffusionModel(cfg).to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=device)["ema"])
model.eval()

# ---------- Generate Samples ----------
print(f"Generating {args.n_gen} samples...")
samples = []
batch_size = 256
steps = args.n_gen // batch_size
with torch.no_grad():
    for _ in tqdm(range(steps)):
        x_gen = model.sample(batch_size)
        samples.append(x_gen.cpu())
samples = torch.cat(samples, dim=0)
samples = (samples * 0.5 + 0.5).clamp(0, 1)  # scale to [0,1]

# ---------- Wrap into Fake Loader ----------
fake_ds = TensorDataset(samples, torch.zeros(len(samples)))
fake_loader = DataLoader(fake_ds, batch_size=128, shuffle=False)

# ---------- Compute FID ----------
fid_score = compute_fid(real_loader, fake_loader, device)
print(f"FID (LeNet-based) = {fid_score:.4f}")
