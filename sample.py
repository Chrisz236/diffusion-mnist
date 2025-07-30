import argparse, yaml, torch
from torchvision.utils import save_image
from models.model import DiffusionModel

parser = argparse.ArgumentParser()
parser.add_argument("--config",  default="configs/default.yaml")
parser.add_argument("--ckpt",    required=True, help="checkpoint .pt")
parser.add_argument("--out",     default=None)
args = parser.parse_args()
cfg  = yaml.safe_load(open(args.config))

device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
model  = DiffusionModel(cfg).to(device)
state  = torch.load(args.ckpt, map_location=device)
model.load_state_dict(state["ema"])        # use EMA weights by default
samples = model.sample(64).cpu()*0.5 + 0.5
out = args.out or args.ckpt.replace(".pt",".png")
save_image(samples, out, nrow=8)
print(f"generated grid → {out}")
