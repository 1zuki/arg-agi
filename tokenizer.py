# =====================================================================
# ARC patch VQ tokenizer (Stage 1 of the IRIS-style token world model).
#
# ARC-AGI-3 frames are ALREADY discrete: a 64x64 grid of values 0-15.
# So unlike IRIS (which learns a VQ-VAE over messy RGB Atari frames),
# here the tokenizer's only job is spatial compression -- turn 4096
# per-cell symbols into a short sequence of patch tokens the world model
# can attend over, while staying sharp enough to decode back per-cell.
#
#   grid [B,64,64] ints 0-15
#     -> one-hot [B,16,64,64]
#     -> conv encoder, stride 2 x2:  64 -> 32 -> 16
#     -> [B,EMB,16,16]  == 256 patch latents (4x4 cells each)
#     -> vector quantize against a CODEBOOK-entry codebook
#     -> tokens [B,256]  (indices into the codebook)
#   decode(tokens) -> [B,16,64,64] per-cell logits  (sharp, per-cell)
#
# 4x4 patches (not 8x8): an 8x8 patch crushed to one code erases a
# one-cell sprite move (before/after patch round to the same token).
# 4x4 patches preserve those small changes -- the whole point of the WM.
#
# No arcengine / agents imports -- loads in any plain Python+torch env.
# =====================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

G = 64                 # grid side
N_COLORS = 16          # cell vocabulary (0-15)
PATCH = 4              # patch side (4x4: fine enough to capture 1-cell moves)
NP = G // PATCH        # 16 patches per side
N_TOKENS = NP * NP     # 256 frame tokens
CODEBOOK = 256         # number of codebook entries
EMB = 64               # codebook vector dim
COMMIT_BETA = 0.25     # commitment loss weight


def grids_to_long(grids):
    """list/array of raw grids -> LongTensor [B,64,64] clamped to 0..15."""
    arr = np.asarray(grids)
    if arr.ndim == 2:
        arr = arr[None]
    arr = np.clip(arr.astype(np.int64), 0, N_COLORS - 1)
    return torch.from_numpy(arr)


class VectorQuantizer(nn.Module):
    """Standard VQ-VAE quantizer with straight-through gradient."""
    def __init__(self, n_codes=CODEBOOK, dim=EMB, beta=COMMIT_BETA):
        super().__init__()
        self.n_codes = n_codes
        self.dim = dim
        self.beta = beta
        self.codebook = nn.Embedding(n_codes, dim)
        self.codebook.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)

    def forward(self, z_e):
        # z_e: [B, dim, H, W] -> flatten to [N, dim]
        B, C, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, C)            # [N,dim]
        # distances to codebook entries
        cb = self.codebook.weight                                # [n_codes,dim]
        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ cb.t()
             + cb.pow(2).sum(1))                                 # [N,n_codes]
        idx = d.argmin(1)                                        # [N]
        z_q = self.codebook(idx).view(B, H, W, C).permute(0, 3, 1, 2)  # [B,dim,H,W]

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commit_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        z_q_st = z_e + (z_q - z_e).detach()                      # straight-through
        tokens = idx.view(B, H * W)                              # [B,N_TOKENS]
        return z_q_st, tokens, vq_loss

    def lookup(self, tokens):
        """tokens [B,N] -> z_q [B,dim,NP,NP] (for decoding sampled tokens)."""
        B, N = tokens.shape
        side = int(round(N ** 0.5))
        z = self.codebook(tokens)                                # [B,N,dim]
        return z.view(B, side, side, self.dim).permute(0, 3, 1, 2)


class Tokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder: one-hot 16ch, 64 -> 16 via 2 stride-2 convs (4x4 patches)
        self.enc = nn.Sequential(
            nn.Conv2d(N_COLORS, 64, 4, stride=2, padding=1), nn.ReLU(),   # 32
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),        # 16
            nn.Conv2d(128, EMB, 3, stride=1, padding=1),                  # 16
        )
        self.vq = VectorQuantizer()
        # decoder: 16 -> 64 via 2 stride-2 transpose convs -> 16ch logits
        self.dec = nn.Sequential(
            nn.Conv2d(EMB, 128, 3, stride=1, padding=1), nn.ReLU(),           # 16
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),   # 32
            nn.ConvTranspose2d(64, N_COLORS, 4, stride=2, padding=1),         # 64
        )

    def _one_hot(self, grid_long):
        # grid_long [B,64,64] -> [B,16,64,64] float
        oh = F.one_hot(grid_long, N_COLORS).permute(0, 3, 1, 2).float()
        return oh

    def encode(self, grid_long):
        """grid_long [B,64,64] -> (z_q_st, tokens, vq_loss)."""
        z_e = self.enc(self._one_hot(grid_long))                 # [B,EMB,16,16]
        return self.vq(z_e)

    def tokens_of(self, grid_long):
        """grid_long [B,64,64] -> tokens [B,256] (no grad path)."""
        with torch.no_grad():
            _, tokens, _ = self.encode(grid_long)
        return tokens

    def decode(self, z_q):
        """z_q [B,EMB,16,16] -> per-cell logits [B,16,64,64]."""
        return self.dec(z_q)

    def decode_tokens(self, tokens):
        """tokens [B,256] -> per-cell logits [B,16,64,64]."""
        return self.dec(self.vq.lookup(tokens))

    def forward(self, grid_long):
        """Training step: returns (recon_logits, tokens, vq_loss)."""
        z_q, tokens, vq_loss = self.encode(grid_long)
        recon = self.dec(z_q)                                    # [B,16,64,64]
        return recon, tokens, vq_loss
