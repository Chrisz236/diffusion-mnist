"""
models/unet.py
==============

Baseline UNet‑style backbone for 28×28 MNIST diffusion.

• `UNet3` predicts ε(t) given a noisy image xₜ and timestep t.
• `sinusoidal_embedding` provides standard DDPM time embeddings.
• `ResidualBottleneck` adds a lightweight ResNet block.

Feel free to extend or replace this file for the assignment.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# 0. Sinusoidal time embedding (from the original DDPM paper)
# ----------------------------------------------------------------------
def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Parameters
    ----------
    t   : (B,) integer timesteps
    dim : embedding dimension (must be even)

    Returns
    -------
    (B, dim) float tensor
    """
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device).float() / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)      # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


# ----------------------------------------------------------------------
# 1. Building block: simple residual bottleneck
# ----------------------------------------------------------------------
class ResidualBottleneck(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        mid = channels // 2
        self.conv1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, channels, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(channels)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.act(out + x)


# ----------------------------------------------------------------------
# 2. UNet3 backbone (depth‑5 encoder‑decoder)
# ----------------------------------------------------------------------
class UNet3(nn.Module):
    """
    A UNet with five resolution levels, tuned for 28×28 images.

    Args
    ----
    in_ch    : number of image channels (1 for MNIST)
    time_dim : embedding size for the timestep

    Notes
    -----
    * Spatial sizes: 28 → 14 → 7 → 4 (stride‑2 convs) and back.
    * `time_dim` is concatenated at the bottleneck and merged upward.
    """

    def __init__(self, in_ch: int = 1, time_dim: int = 128):
        super().__init__()

        C = [128, 256, 512, 1024, 1024]          # channels at each level

        # ---- time embedding MLP ----
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ---- encoder ----
        self.enc1 = nn.Conv2d(in_ch, C[0], 3, padding=1)  # 28×28
        self.rb1  = ResidualBottleneck(C[0])

        self.enc2 = nn.Conv2d(C[0], C[1], 3, stride=2, padding=1)  # 14×14
        self.rb2  = ResidualBottleneck(C[1])

        self.enc3 = nn.Conv2d(C[1], C[2], 3, stride=2, padding=1)  # 7×7
        self.rb3  = ResidualBottleneck(C[2])

        self.enc4 = nn.Conv2d(C[2], C[3], 3, stride=2, padding=1)  # 4×4
        self.rb4  = ResidualBottleneck(C[3])

        self.rb5  = ResidualBottleneck(C[4])  # extra depth at bottleneck

        # ---- decoder ----
        # Level 5 → 4
        self.up5  = nn.ConvTranspose2d(C[4] + time_dim, C[3], 2, stride=2)
        self.rb5d = ResidualBottleneck(C[3])

        # Level 4 → 3
        self.up4  = nn.ConvTranspose2d(C[3]*2, C[2], 2, stride=2)
        self.rb4d = ResidualBottleneck(C[2])

        # Level 3 → 2
        self.up3  = nn.ConvTranspose2d(C[2]*2, C[1], 2, stride=2)
        self.rb3d = ResidualBottleneck(C[1])

        # Level 2 → 1
        self.up2  = nn.ConvTranspose2d(C[1]*2, C[0], 2, stride=2)
        self.rb2d = ResidualBottleneck(C[0])

        # Final 1×1 conv (concat with original x1)
        self.final = nn.Conv2d(C[0]*2, in_ch, 1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_ch, 28, 28)
        t : (B,) integer timestep indices
        """
        # ---- time embedding ----
        te   = sinusoidal_embedding(t, self.time_mlp[0].in_features)   # (B, time_dim)
        temb = self.time_mlp(te).unsqueeze(-1).unsqueeze(-1)           # (B, time_dim, 1, 1)

        # ---- encoder ----
        x1 = F.relu(self.enc1(x)); x1 = self.rb1(x1)   # 28×28
        x2 = F.relu(self.enc2(x1)); x2 = self.rb2(x2)  # 14×14
        x3 = F.relu(self.enc3(x2)); x3 = self.rb3(x3)  # 7×7
        x4 = F.relu(self.enc4(x3)); x4 = self.rb4(x4)  # 4×4
        x5 = self.rb5(x4)                              # 4×4

        # ---- decoder ----
        # level 5 → 4
        y = torch.cat([x5, temb.expand(-1, -1, x5.size(2), x5.size(3))], dim=1)
        y = self.up5(y)                                # 8×8
        y = F.interpolate(y, size=x4.shape[2:], mode='bilinear', align_corners=False)
        y = self.rb5d(y)

        # level 4 → 3
        y = torch.cat([y, x4], dim=1)
        y = self.up4(y)                                # 16×16
        y = F.interpolate(y, size=x3.shape[2:], mode='bilinear', align_corners=False)
        y = self.rb4d(y)

        # level 3 → 2
        y = torch.cat([y, x3], dim=1)
        y = self.up3(y)                                # 28×28 (or 14×14 upstream)
        y = F.interpolate(y, size=x2.shape[2:], mode='bilinear', align_corners=False)
        y = self.rb3d(y)

        # level 2 → 1
        y = torch.cat([y, x2], dim=1)
        y = self.up2(y)
        y = F.interpolate(y, size=x1.shape[2:], mode='bilinear', align_corners=False)
        y = self.rb2d(y)

        # final 1×1 conv
        y = torch.cat([y, x1], dim=1)
        return self.final(y)


# What should be imported when using `from models.unet import *`
__all__ = ["UNet3", "ResidualBottleneck", "sinusoidal_embedding"]
