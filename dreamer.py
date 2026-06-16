# =====================================================================
# Compact DreamerV3-style RSSM world model for ARC-AGI-3.
#
# This is the FAITHFUL CORE of DreamerV3's world model, not the full
# paper. What's kept:
#   - Recurrent State-Space Model (RSSM): a deterministic GRU state h
#     plus a DISCRETE stochastic latent z (categorical, straight-through),
#     exactly DreamerV3's latent design.
#   - prior  p(z_t | h_t)            -- imagine the next latent
#   - posterior q(z_t | h_t, e_t)    -- correct it with the observed frame
#   - decoder ([h,z] -> per-cell grid logits), reward head, continue/state head
#   - KL balancing between prior and posterior (the DreamerV3 trick that
#     keeps the prior learnable without the posterior collapsing).
#
# What's CUT vs the paper (and why it's fine here):
#   - symlog transforms  : ARC has no large-magnitude continuous signals.
#   - two-hot reward head : our rewards are small integers; MSE is enough.
#   - free bits / fixed KL scale tuning : kept a single kl_scale.
#   - actor-critic in imagination : this file is the WORLD MODEL only, so
#     it can be compared head-to-head with cwm.py on next-frame fidelity.
#
# Why this is expected to look BLURRIER than cwm.py against ground truth:
# the RSSM predicts in a compact LATENT (h+z), then decodes -- fine spatial
# detail is bottlenecked, the same effect that made our first latent WM
# blurry. DreamerV3's latent is built for control sample-efficiency, not
# pixel fidelity, so this is the expected trade, not a bug.
#
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

DET = 256               # deterministic state h size
N_CAT = 16              # number of categorical latent variables
N_CLS = 16              # classes per categorical
STOCH = N_CAT * N_CLS   # flattened stochastic latent z size (256)
EMB = 256               # frame embedding size
HID = 256               # hidden width in heads
KL_SCALE = 1.0
KL_BALANCE = 0.8        # weight on prior-side KL (DreamerV3 default ~0.8)


def grids_to_long(grids):
    """list/array of raw grids -> LongTensor [B,64,64] clamped to 0..15."""
    arr = np.asarray(grids)
    if arr.ndim == 2:
        arr = arr[None]
    arr = np.clip(arr.astype(np.int64), 0, N_COLORS - 1)
    return torch.from_numpy(arr)


class Encoder(nn.Module):
    """grid [B,64,64] ints -> embedding [B,EMB]."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(N_COLORS, 32, 4, stride=2, padding=1), nn.ReLU(),   # 32
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),         # 16
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),        # 8
            nn.Conv2d(128, 128, 4, stride=2, padding=1), nn.ReLU(),       # 4
        )
        self.fc = nn.Linear(128 * 4 * 4, EMB)

    def forward(self, grid_long):
        x = F.one_hot(grid_long, N_COLORS).permute(0, 3, 1, 2).float()
        h = self.net(x).flatten(1)
        return F.relu(self.fc(h))


class Decoder(nn.Module):
    """latent feature [B,DET+STOCH] -> per-cell grid logits [B,16,64,64]."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(DET + STOCH, 128 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1), nn.ReLU(),  # 8
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),   # 16
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),    # 32
            nn.ConvTranspose2d(32, N_COLORS, 4, stride=2, padding=1),         # 64
        )

    def forward(self, feat):
        h = self.fc(feat).view(-1, 128, 4, 4)
        return self.net(h)


class RSSM(nn.Module):
    """DreamerV3-style recurrent state-space model with discrete latents.

    State = (h deterministic, z stochastic-discrete). One step:
      h_t       = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
      prior     z_t ~ p(. | h_t)
      posterior z_t ~ q(. | h_t, e_t)     (e_t = encoder(frame_t))
    Latents are straight-through one-hot samples over N_CAT categoricals.
    """
    def __init__(self):
        super().__init__()
        self.act_emb = nn.Embedding(N_ACTIONS, 64)
        self.in_proj = nn.Linear(STOCH + 64, DET)
        self.cell = nn.GRUCell(DET, DET)
        self.prior = nn.Sequential(nn.Linear(DET, HID), nn.ReLU(),
                                   nn.Linear(HID, STOCH))
        self.post = nn.Sequential(nn.Linear(DET + EMB, HID), nn.ReLU(),
                                  nn.Linear(HID, STOCH))

    def _logits(self, x):
        return x.view(-1, N_CAT, N_CLS)

    def _sample(self, logits):
        """Straight-through categorical sample -> flat one-hot [B,STOCH]."""
        probs = F.softmax(logits, dim=-1)
        idx = torch.multinomial(probs.reshape(-1, N_CLS), 1).view(-1, N_CAT)
        onehot = F.one_hot(idx, N_CLS).float()
        # straight-through: gradient flows through probs
        onehot = onehot + probs - probs.detach()
        return onehot.reshape(-1, STOCH)

    def initial(self, batch, device):
        return (torch.zeros(batch, DET, device=device),
                torch.zeros(batch, STOCH, device=device))

    def step(self, h_prev, z_prev, action_ids, embed=None):
        """One RSSM step. Returns (h, z, prior_logits, post_logits).
        If embed is given, z is the POSTERIOR sample; else the PRIOR sample."""
        a = self.act_emb(action_ids.long())
        x = F.relu(self.in_proj(torch.cat([z_prev, a], dim=-1)))
        h = self.cell(x, h_prev)
        prior_logits = self._logits(self.prior(h))
        if embed is not None:
            post_logits = self._logits(self.post(torch.cat([h, embed], dim=-1)))
            z = self._sample(post_logits)
        else:
            post_logits = None
            z = self._sample(prior_logits)
        return h, z, prior_logits, post_logits


class Dreamer(nn.Module):
    """World model: encoder + RSSM + decoder + reward + state heads.

    For a single (frame, action) -> next-frame comparison we use:
      observe(frame)   -> posterior (h,z) summarizing the current frame
      imagine_step(.)  -> roll the prior one step under the action,
                          decode the predicted next frame + reward + state.
    """
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.rssm = RSSM()
        self.dec = Decoder()
        self.head_rew = nn.Sequential(nn.Linear(DET + STOCH, HID), nn.ReLU(),
                                      nn.Linear(HID, 1))
        self.head_state = nn.Sequential(nn.Linear(DET + STOCH, HID), nn.ReLU(),
                                        nn.Linear(HID, N_STATES))

    def _feat(self, h, z):
        return torch.cat([h, z], dim=-1)

    def observe(self, grid_long, action_ids=None, h=None, z=None):
        """Fold one observed frame into the latent state, returning posterior
        (h,z). action_ids defaults to RESET(0); h,z default to zeros."""
        B = grid_long.size(0)
        dev = grid_long.device
        if h is None or z is None:
            h, z = self.rssm.initial(B, dev)
        if action_ids is None:
            action_ids = torch.zeros(B, dtype=torch.long, device=dev)
        e = self.enc(grid_long)
        h, z, _prior, _post = self.rssm.step(h, z, action_ids, embed=e)
        return h, z

    def imagine_step(self, h, z, action_ids):
        """Roll the PRIOR one step under action_ids (no observation).
        Returns (h2, z2, cell_logits, reward, state_logits)."""
        h2, z2, _prior, _post = self.rssm.step(h, z, action_ids, embed=None)
        feat = self._feat(h2, z2)
        cell_logits = self.dec(feat)
        reward = self.head_rew(feat).squeeze(-1)
        state_logits = self.head_state(feat)
        return h2, z2, cell_logits, reward, state_logits

    @torch.no_grad()
    def imagine(self, grid_long, action_ids, click_xy=None):
        """Single-step next-frame prediction, same signature shape as
        cwm.imagine (click_xy ignored -- RSSM has no spatial click input).
        Returns (next_grid [B,64,64] int, reward [B], state_idx [B],
        state_logits [B,4])."""
        self.eval()
        h, z = self.observe(grid_long)
        _h2, _z2, cell_logits, reward, state_logits = self.imagine_step(h, z, action_ids)
        next_grid = cell_logits.argmax(1)
        return next_grid, reward, state_logits.argmax(-1), state_logits


def kl_loss(post_logits, prior_logits):
    """KL-balanced loss between posterior and prior categoricals (DreamerV3).
    Mixes KL(sg(post)||prior) and KL(post||sg(prior)) by KL_BALANCE."""
    post = torch.distributions.Categorical(logits=post_logits)
    prior = torch.distributions.Categorical(logits=prior_logits)
    post_sg = torch.distributions.Categorical(logits=post_logits.detach())
    prior_sg = torch.distributions.Categorical(logits=prior_logits.detach())
    kl_lhs = torch.distributions.kl_divergence(post_sg, prior).sum(-1).mean()
    kl_rhs = torch.distributions.kl_divergence(post, prior_sg).sum(-1).mean()
    return KL_BALANCE * kl_lhs + (1 - KL_BALANCE) * kl_rhs
