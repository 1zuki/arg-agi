# =====================================================================
# Shared model definitions for the ARC-AGI-3 Dyna-WM agent.
#
# Single source of truth for the networks + the observation encoding,
# imported by BOTH my_agent.py (online play) and pretrain.py (offline
# pretraining). No arcengine / agents imports here, so it loads in any
# plain Python+torch environment.
#
#   Encoder       grid -> latent z + spatial feature
#   WorldModel    (z, discrete action) -> next z, predicted reward
#   ClickHead     feature -> 64x64 click Q-map
#   LatentDecoder z -> 16x64x64 grid logits (for WM-prediction visualization)
#   Net           bundles all of the above
#   obs_tensor    raw grid -> input tensor (16 one-hot + 2 coord planes)
# =====================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- shared hyperparameters ----
G = 64               # grid side
N_DISC = 5           # ACTION1..ACTION5
IN_CH = 18           # 16 one-hot colors + row/col coord planes
D = 128              # latent dim
GAMMA = 0.99
TAU = 0.01           # target soft-update rate
BETA_INT = 0.5       # curiosity weight
LR = 3e-4
GHW = G * G          # click action count


class Encoder(nn.Module):
    def __init__(self, in_ch=IN_CH):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.proj = nn.Linear(128, D)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.max_pool2d(x, 2)            # 64 -> 32
        x = F.relu(self.c2(x))
        x = F.max_pool2d(x, 2)            # 32 -> 16
        feat = F.relu(self.c3(x))         # [B,128,16,16]
        z = self.proj(feat.mean(dim=[2, 3]))
        return z, feat


class ClickHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(128, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 1, 1)

    def forward(self, feat):
        h = F.relu(self.c1(feat))
        q = self.c2(h)                                   # [B,1,16,16]
        q = F.interpolate(q, size=(G, G), mode="bilinear", align_corners=False)
        return q.squeeze(1)                              # [B,64,64]


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(D + N_DISC, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.delta = nn.Linear(256, D)
        self.rew = nn.Linear(256, 1)

    def forward(self, z, a_onehot):
        h = self.body(torch.cat([z, a_onehot], dim=-1))
        z_next = z + self.delta(h)                       # residual dynamics
        return z_next, self.rew(h).squeeze(-1)


class LatentDecoder(nn.Module):
    """z (128-d latent) -> 16x64x64 grid logits.

    Decodes from the PURE latent (not the spatial feature map), because the
    world model predicts a latent z_next and this is what lets us render that
    prediction as a picture. The latent is a 128-number bottleneck, so decoded
    grids are inherently soft/blurry -- that is exactly what the WM "sees".
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(D, 128 * 8 * 8)
        self.c1 = nn.Conv2d(128, 64, 3, padding=1)
        self.c2 = nn.Conv2d(64, 32, 3, padding=1)
        self.out = nn.Conv2d(32, 16, 1)

    def forward(self, z):
        h = self.fc(z).view(-1, 128, 8, 8)                     # [B,128,8,8]
        h = F.interpolate(h, scale_factor=2, mode="nearest")   # 16
        h = F.relu(self.c1(h))
        h = F.interpolate(h, scale_factor=2, mode="nearest")   # 32
        h = F.relu(self.c2(h))
        h = F.interpolate(h, size=(G, G), mode="nearest")      # 64
        return self.out(h)                                     # [B,16,64,64] logits


class Net(nn.Module):
    def __init__(self, in_ch=IN_CH):
        super().__init__()
        self.enc = Encoder(in_ch)
        self.qd = nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, N_DISC))
        self.click = ClickHead()
        self.wm = WorldModel()
        self.dec = LatentDecoder()

    def encode(self, x):
        return self.enc(x)

    def q_discrete(self, z):
        return self.qd(z)

    def q_click(self, feat):
        return self.click(feat)

    def decode(self, z):
        return self.dec(z)


def obs_tensor(raw):
    """raw 2D grid (ints 0-15) -> [18,64,64] float tensor: 16 one-hot + 2 coord planes."""
    raw = np.clip(np.asarray(raw, dtype=np.int64), 0, 15)
    oh = torch.zeros(16, G, G)
    oh.scatter_(0, torch.from_numpy(raw).unsqueeze(0), 1.0)
    rp = torch.linspace(0, 1, G).view(G, 1).repeat(1, G)
    cp = torch.linspace(0, 1, G).view(1, G).repeat(G, 1)
    return torch.cat([oh, rp.unsqueeze(0), cp.unsqueeze(0)], 0)
