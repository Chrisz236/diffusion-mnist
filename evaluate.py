import argparse, yaml, torch, os
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from models.model import DiffusionModel

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--ckpt",   required=True)
parser.add_argument("--n_gen",  type=int, default=10000)
args = parser.parse_args()
cfg = yaml.safe_load(open(args.config))
device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

# ---------- GT ----------
tfm = transforms.Compose([transforms.Resize(28),
                          transforms.ToTensor(),
                          transforms.Normalize([0.5],[0.5])])
gt_ds = MNIST(cfg["data_root"], train=False, download=True, transform=tfm)
gt_loader = DataLoader(gt_ds, batch_size=128, shuffle=False, num_workers=4)

# ---------- model ----------
model = DiffusionModel(cfg).to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=device)["ema"])
model.eval()

fid = FrechetInceptionDistance(feature=64).to(device)  # small feature size OK for MNIST

# add real
with torch.no_grad():
    for x,_ in gt_loader:
        fid.update((x*0.5+0.5).to(device), real=True)

# add fake
batch = 256
for _ in range(args.n_gen // batch):
    fake = model.sample(batch).to(device)*0.5 + 0.5
    fid.update(fake, real=False)

print(f"FID = {fid.compute().item():.4f}")
