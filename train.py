import yaml, argparse, math, os, torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from models.model import DiffusionModel

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--output", default=None)          # optional override
args = parser.parse_args()
cfg  = yaml.safe_load(open(args.config))
if args.output: cfg["output_root"] = args.output

os.makedirs(cfg["output_root"], exist_ok=True)

# ---------- deterministic ----------
torch.manual_seed(cfg["seed"])
device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

# ---------- data ----------
transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5]),
])
ds = MNIST(cfg["data_root"], train=True, download=True, transform=transform)
loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)
steps_per_epoch = math.ceil(len(ds)/cfg["batch_size"])
cfg["scheduler"]["T_max"] = cfg["epochs"] * steps_per_epoch

# ---------- model / opt / ema ----------
model = DiffusionModel(cfg).to(device)
opt   = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
ema   = AveragedModel(model, avg_fn=lambda a,b,_: cfg["ema_decay"]*a + (1-cfg["ema_decay"])*b)
sched = CosineAnnealingLR(opt, **cfg["scheduler"])

# ---------- training ----------
for epoch in range(1, cfg["epochs"]+1):
    model.train()
    running = 0.0
    for x,_ in tqdm(loader, desc=f"Epoch {epoch}/{cfg['epochs']}"):
        x = x.to(device)
        loss = model.loss(x)

        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        ema.update_parameters(model)
        running += loss.item()

    print(f"Epoch {epoch}: {running/len(loader):.4f}")

    # checkpoint & sample every 5 epochs
    if epoch % 5 == 0:
        ckpt_path = f"{cfg['output_root']}/ckpt_{epoch:03d}.pt"
        torch.save({"model":model.state_dict(),"ema":ema.module.state_dict()}, ckpt_path)
        from torchvision.utils import save_image
        samples = model.sample(64, ema_net=ema.module).cpu()*0.5 + 0.5
        grid_path = f"{cfg['output_root']}/sample_{epoch:03d}.png"
        save_image(samples, grid_path, nrow=8)
        print(f"✓ saved {ckpt_path} & {grid_path}")
