# =====================================================================
# Convolutional grid->grid world model for ARC-AGI-3.
#
# ARC frames are ALREADY discrete: a 64x64 grid of values 0-15. There is
# no messy RGB to discretize, so a VQ tokenizer only throws information
# away (it rounds rare sprite-moves to the same code as the background).
# This model skips tokenization entirely:
#
#   frame [B,64,64] ints 0-15  +  action 0-7
#     -> one-hot [B,16,64,64]
#     -> U-Net (skip connections preserve fine spatial detail)
#     -> FiLM-conditioned on the action at the bottleneck
#     -> per-cell next-frame logits [B,16,64,64]   (SHARP, lossless-capable)
#        + reward scalar
#        + game state 4-way (NOT_PLAYED/NOT_FINISHED/WIN/GAME_OVER)
#
# Imagination is ONE forward pass (not 256 autoregressive samples), so it
# is cheap enough to roll out under the per-game action budget on a T4.
# No arcengine / agents imports -- loads in any plain Python+torch env.
# =====================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

G = 64                  # grid side
N_COLORS = 16           # cell vocabulary (0-15)
N_ACTIONS = 8           # RESET + ACTION1..7  (ids 0..7)
N_STATES = 4
STATE_NAMES = ["NOT_PLAYED", "NOT_FINISHED", "WIN", "GAME_OVER"]
CH = 64                 # base channel width


def grids_to_long(grids):
    """list/array of raw grids -> LongTensor [B,64,64] clamped to 0..15."""
    arr = np.asarray(grids)
    if arr.ndim == 2:
        arr = arr[None]
    arr = np.clip(arr.astype(np.int64), 0, N_COLORS - 1)
    return torch.from_numpy(arr)


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.ReLU(),
    )


class ConvWorldModel(nn.Module):
    """U-Net grid->grid dynamics with FiLM action conditioning.

    Skip connections carry fine spatial detail (exact sprite positions) past
    the bottleneck, so a one-cell move survives -- this is what VQ destroyed.
    The action modulates the bottleneck via FiLM (per-channel scale+shift),
    so the same frame yields different next-frames for different actions.
    """
    def __init__(self):
        super().__init__()
        # +1 input channel: a spatial plane hot at the clicked cell (ACTION6).
        # This is how the model finally SEES where a click landed -- the action
        # id alone collapsed every click to one embedding (CLICK acc was 23%).
        self.enc1 = _block(N_COLORS + 1, CH)        # 64
        self.enc2 = _block(CH, CH * 2)              # 32
        self.enc3 = _block(CH * 2, CH * 4)          # 16
        self.pool = nn.MaxPool2d(2)

        self.bott = _block(CH * 4, CH * 4)          # 8
        # FiLM: action embedding -> per-channel (scale, shift) at bottleneck
        self.act_emb = nn.Embedding(N_ACTIONS, CH * 4)
        self.film = nn.Linear(CH * 4, CH * 4 * 2)

        self.up3 = nn.ConvTranspose2d(CH * 4, CH * 4, 2, stride=2)   # 8->16
        self.dec3 = _block(CH * 8, CH * 2)
        self.up2 = nn.ConvTranspose2d(CH * 2, CH * 2, 2, stride=2)   # 16->32
        self.dec2 = _block(CH * 4, CH)
        self.up1 = nn.ConvTranspose2d(CH, CH, 2, stride=2)          # 32->64
        self.dec1 = _block(CH * 2, CH)

        self.head_cell = nn.Conv2d(CH, N_COLORS, 1)     # per-cell next-frame logits
        self.head_rew = nn.Linear(CH * 4, 1)            # reward, from bottleneck
        self.head_state = nn.Linear(CH * 4, N_STATES)   # game state, from bottleneck

    def _one_hot(self, grid_long):
        return F.one_hot(grid_long, N_COLORS).permute(0, 3, 1, 2).float()

    def _click_plane(self, grid_long, click_xy):
        """Build the [B,1,64,64] click channel: 1.0 at the clicked cell, else 0.
        click_xy: [B,2] long (x,y), or None. Rows with x<0 mean 'no click'."""
        B = grid_long.size(0)
        plane = torch.zeros(B, 1, G, G, device=grid_long.device)
        if click_xy is None:
            return plane
        xy = click_xy.to(grid_long.device).long()
        for i in range(B):
            x, y = int(xy[i, 0]), int(xy[i, 1])
            if 0 <= x < G and 0 <= y < G:
                plane[i, 0, y, x] = 1.0
        return plane

    def forward(self, grid_long, action_ids, click_xy=None):
        """grid_long [B,64,64] ints, action_ids [B] 0..7, click_xy [B,2] or None.
        Returns (cell_logits [B,16,64,64], reward [B], state_logits [B,4])."""
        x = torch.cat([self._one_hot(grid_long),
                       self._click_plane(grid_long, click_xy)], dim=1)
        e1 = self.enc1(x)                       # [B,CH,64,64]
        e2 = self.enc2(self.pool(e1))           # [B,2CH,32,32]
        e3 = self.enc3(self.pool(e2))           # [B,4CH,16,16]
        b = self.bott(self.pool(e3))            # [B,4CH,8,8]

        # FiLM action conditioning at the bottleneck
        gamma_beta = self.film(self.act_emb(action_ids.long()))      # [B,8CH]
        gamma, beta = gamma_beta.chunk(2, dim=-1)                    # [B,4CH] each
        b = b * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]

        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))   # [B,2CH,16,16]
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))  # [B,CH,32,32]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))  # [B,CH,64,64]

        cell_logits = self.head_cell(d1)                  # [B,16,64,64]
        pooled = b.mean(dim=[2, 3])                       # [B,4CH]
        reward = self.head_rew(pooled).squeeze(-1)        # [B]
        state_logits = self.head_state(pooled)            # [B,4]
        return cell_logits, reward, state_logits

    @torch.no_grad()
    def imagine(self, grid_long, action_ids, click_xy=None):
        """One forward pass. Returns (next_grid [B,64,64] int, reward [B],
        state_idx [B], state_logits [B,4])."""
        self.eval()
        cell_logits, reward, state_logits = self.forward(grid_long, action_ids, click_xy)
        next_grid = cell_logits.argmax(1)                 # [B,64,64]
        return next_grid, reward, state_logits.argmax(-1), state_logits


class ValueNet(nn.Module):
    """Small conv state-value net: grid [B,64,64] ints -> V(grid) scalar.

    The only thing the agent trains ONLINE per game. Kept light so it adapts
    in a few hundred real steps under the action budget. The CWM (perception
    + dynamics) is frozen; this learns 'how close to winning is this state'.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(N_COLORS, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),        # 16
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),        # 8
        )
        self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, grid_long):
        """grid_long [B,64,64] ints -> V [B]."""
        x = F.one_hot(grid_long, N_COLORS).permute(0, 3, 1, 2).float()
        feat = self.net(x).mean(dim=[2, 3])               # [B,64]
        return self.head(feat).squeeze(-1)                # [B]