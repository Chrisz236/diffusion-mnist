"""
The ONLY file students must modify for new architectures / tricks.
Keep the public API identical so train.py / sample.py keep working.
"""
from types import SimpleNamespace
import torch, torch.nn as nn
from .unet import UNet3

def cosine_beta_schedule(T, s=0.008, device="cpu"):
    import math, torch
    steps = torch.linspace(0, T, T+1, dtype=torch.float32, device=device)
    alphas = torch.cos(((steps / T + s) / (1 + s)) * (math.pi/2))**2
    beta = torch.clip(1 - alphas[1:] / alphas[:-1], 0.0, 0.999)
    alpha = 1.0 - beta
    return beta, alpha, torch.cumprod(alpha, 0)

class DiffusionModel(nn.Module):
    """
    Wrapper that owns:
      • the backbone network (default UNet3)
      • the noise schedule
      • loss computation helpers
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg   = SimpleNamespace(**cfg)            # yaml dict → dot access
        self.net   = UNet3(in_ch=1, time_dim=cfg["time_dim"])
        self.T     = cfg["T"]
        beta, alpha, abar = cosine_beta_schedule(self.T, device="cpu")
        self.register_buffer("beta",  beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("abar",  abar)

    # ---------- public helpers called by scripts ----------
    def forward(self, x, t):                # noise prediction
        return self.net(x, t)

    def loss(self, x0):
        B = x0.size(0)
        device = x0.device
        t  = torch.randint(0, self.T, (B,), device=device)
        eps = torch.randn_like(x0)
        abar_t = self.abar[t].view(-1,1,1,1)
        xt = torch.sqrt(abar_t)*x0 + torch.sqrt(1 - abar_t)*eps
        eps_pred = self(xt, t)
        return ((eps - eps_pred)**2).mean()

    @torch.no_grad()
    def sample(self, n, ema_net=None):
        """Generate `n` samples in [-1,1]"""
        net = ema_net if ema_net is not None else self
        device, T = next(net.parameters()).device, self.T
        x = torch.randn(n,1,28,28, device=device)
        for i in reversed(range(T)):
            t_full = torch.full((n,), i, device=device, dtype=torch.long)
            eps = net(x, t_full)
            β, α, ā = self.beta[i], self.alpha[i], self.abar[i]
            x0 = (x - torch.sqrt(1-ā)*eps) / torch.sqrt(ā)
            x0.clamp_(-1,1)
            if i > 0:
                ā_prev = self.abar[i-1]
                mean = (β*torch.sqrt(ā_prev)/(1-ā))*x0 + ((1-ā_prev)*torch.sqrt(α)/(1-ā))*x
                std  = torch.sqrt(β*(1-ā_prev)/(1-ā))
                x = mean + std*torch.randn_like(x)
            else:
                x = x0
        return x
