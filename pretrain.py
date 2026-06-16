#!/usr/bin/env python3
# =====================================================================
# Offline pretraining for the ARC-AGI-3 Dyna-WM agent.
#
# Drives the OFFLINE ARC-AGI engine (no API key, ~2000 FPS) over the
# public games, collects transitions by random/exploratory play, and
# trains the SHARED encoder + world model so the online agent starts
# with a transferable dynamics prior instead of from scratch.
#
# Why a decoder here:
#   Training (z,a)->z_next alone collapses (encoder -> constant scores
#   zero loss). We ground it with a reconstruction decoder (feat->grid)
#   that is used ONLY during pretraining and discarded. The saved
#   weights are exactly model.Net's state_dict, so my_agent.py's
#   shape-matched loader picks them up unchanged.
#
# Losses:
#   L_recon  CE(decode(feat), grid)         keeps latent informative
#   L_dyn    MSE(wm(z,a), sg(encode(s2)))   latent dynamics (target stop-grad)
#   L_rew    MSE(wm_r(z,a), reward)          reward prediction
#
# Usage:
#   python pretrain.py --steps 4000 --epochs 3 --out pretrained_weights.pt
#   python pretrain.py --games ls20 ft09 --steps 2000
#   python pretrain.py --buffer dump.npz        # train from a saved buffer
# =====================================================================
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import Net, obs_tensor, G, N_DISC, D, LR
from tokenizer import Tokenizer, grids_to_long, N_TOKENS, CODEBOOK
from twm import TokenWorldModel, N_ACTIONS, N_STATES
from cwm import ConvWorldModel
import dreamer as DR


# =====================================================================
# Training core  (engine-independent, unit-testable)
#
# Decoder is now net.dec (model.LatentDecoder), so it is SAVED with the
# weights and the comparison replay can render WM predictions. Losses:
#   L_recon  CE(decode(z), s)               latent stays informative
#   L_dyn    MSE(wm(z,a), sg(encode(s2)))   latent dynamics
#   L_rew    MSE(wm_r(z,a), r)              reward prediction
#   L_pred   CE(decode(wm(z,a)), s2)        predicted NEXT grid <- the picture
# =====================================================================
def train_on_buffer(net, buffer, epochs=3, batch=64, lr=LR, device="cpu",
                    log_every=50):
    """buffer: list of dicts {s, a, r, s2}. s/s2 are raw int grids (HxW);
    a is the discrete action index 0..N_DISC-1, or -1 for click/other
    (reconstruction uses every sample; latent dynamics + prediction use
    discrete-action samples only)."""
    if len(buffer) < batch:
        raise ValueError(f"buffer too small ({len(buffer)} < {batch})")
    dev = torch.device(device)
    net.to(dev)
    opt = optim.Adam(net.parameters(), lr=lr)

    n = len(buffer)
    steps_per_epoch = max(1, n // batch)
    last = {}
    for ep in range(epochs):
        order = np.random.permutation(n)
        for bi in range(steps_per_epoch):
            sel = order[bi * batch:(bi + 1) * batch]
            if len(sel) < batch:
                break
            b = [buffer[i] for i in sel]
            grids = np.stack([np.clip(np.asarray(e["s"], dtype=np.int64), 0, 15) for e in b])
            grids2 = np.stack([np.clip(np.asarray(e["s2"], dtype=np.int64), 0, 15) for e in b])
            st = torch.stack([obs_tensor(e["s"]) for e in b]).to(dev)
            st2 = torch.stack([obs_tensor(e["s2"]) for e in b]).to(dev)
            a = torch.tensor([e["a"] for e in b], dtype=torch.long, device=dev)
            r = torch.tensor([e["r"] for e in b], dtype=torch.float32, device=dev)
            labels = torch.from_numpy(grids).to(dev)            # [B,64,64]
            labels2 = torch.from_numpy(grids2).to(dev)          # [B,64,64]

            z, _ = net.encode(st)
            l_recon = F.cross_entropy(net.decode(z), labels)

            with torch.no_grad():
                z2, _ = net.encode(st2)
            disc = a >= 0
            if disc.any():
                a_oh = F.one_hot(a[disc].clamp(min=0), N_DISC).float()
                z_pred, r_pred = net.wm(z[disc], a_oh)
                l_dyn = F.mse_loss(z_pred, z2[disc].detach())
                l_rew = F.mse_loss(r_pred, r[disc])
                l_pred = F.cross_entropy(net.decode(z_pred), labels2[disc])
            else:
                l_dyn = torch.zeros((), device=dev)
                l_rew = torch.zeros((), device=dev)
                l_pred = torch.zeros((), device=dev)

            loss = l_recon + l_dyn + l_rew + l_pred
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

            last = {"epoch": ep, "loss": loss.item(), "recon": l_recon.item(),
                    "dyn": float(l_dyn.detach()), "rew": float(l_rew.detach()),
                    "pred": float(l_pred.detach() if torch.is_tensor(l_pred) else l_pred)}
            if (bi % log_every) == 0:
                print(f"  ep{ep} step{bi}/{steps_per_epoch} "
                      f"loss={last['loss']:.3f} recon={last['recon']:.3f} "
                      f"dyn={last['dyn']:.4f} rew={last['rew']:.4f} pred={last['pred']:.3f}")
    return last


# =====================================================================
# IRIS-style token world model training  (Stage 1 tokenizer, Stage 2 TWM)
# =====================================================================
def _unique_grids(buffer):
    """All distinct grids in the buffer (both s and s2), as a [N,64,64] array.
    Tokenizer training only needs the frames, not transitions."""
    seen = {}
    for e in buffer:
        for key in ("s", "s2"):
            g = np.clip(np.asarray(e[key], dtype=np.int64), 0, 15)
            if g.shape != (G, G):
                continue
            h = g.tobytes()
            if h not in seen:
                seen[h] = g
    return np.stack(list(seen.values())) if seen else np.zeros((0, G, G), dtype=np.int64)


def train_tokenizer(tok, buffer, epochs=4, batch=128, lr=3e-4, device="cpu",
                    log_every=50):
    """Train the patch VQ tokenizer on every distinct grid in the buffer.
    Loss = per-cell CE(recon, grid) + vq_loss. Reports per-cell recon accuracy."""
    grids = _unique_grids(buffer)
    n = len(grids)
    if n < batch:
        raise ValueError(f"too few distinct grids ({n} < {batch})")
    dev = torch.device(device)
    tok.to(dev)
    opt = optim.Adam(tok.parameters(), lr=lr)
    steps_per_epoch = max(1, n // batch)
    last = {}
    for ep in range(epochs):
        order = np.random.permutation(n)
        for bi in range(steps_per_epoch):
            sel = order[bi * batch:(bi + 1) * batch]
            if len(sel) < batch:
                break
            g = torch.from_numpy(grids[sel]).long().to(dev)        # [B,64,64]
            recon, _, vq_loss = tok(g)                             # recon [B,16,64,64]
            ce = F.cross_entropy(recon, g)
            loss = ce + vq_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(tok.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                acc = (recon.argmax(1) == g).float().mean().item()
            last = {"epoch": ep, "loss": loss.item(), "ce": ce.item(),
                    "vq": float(vq_loss.detach()), "acc": acc}
            if (bi % log_every) == 0:
                print(f"  [tok] ep{ep} step{bi}/{steps_per_epoch} "
                      f"loss={last['loss']:.3f} ce={last['ce']:.3f} "
                      f"vq={last['vq']:.4f} acc={acc:.3f}")
    return last


def train_twm(twm, tok, buffer, epochs=4, batch=64, lr=3e-4, device="cpu",
              log_every=50):
    """Freeze the tokenizer, tokenize each transition, and train the GPT token
    world model on next-frame tokens + reward + game state. Reports next-token
    top-1 accuracy and state accuracy."""
    dev = torch.device(device)
    tok.to(dev).eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    twm.to(dev)
    opt = optim.Adam(twm.parameters(), lr=lr)

    # keep only transitions with two valid 64x64 grids
    items = [e for e in buffer
             if np.asarray(e["s"]).shape == (G, G)
             and np.asarray(e["s2"]).shape == (G, G)]
    n = len(items)
    if n < batch:
        raise ValueError(f"too few valid transitions ({n} < {batch})")
    steps_per_epoch = max(1, n // batch)
    last = {}
    for ep in range(epochs):
        order = np.random.permutation(n)
        for bi in range(steps_per_epoch):
            sel = order[bi * batch:(bi + 1) * batch]
            if len(sel) < batch:
                break
            b = [items[i] for i in sel]
            cur = grids_to_long(np.stack([np.clip(np.asarray(e["s"], dtype=np.int64), 0, 15) for e in b])).to(dev)
            nxt = grids_to_long(np.stack([np.clip(np.asarray(e["s2"], dtype=np.int64), 0, 15) for e in b])).to(dev)
            aid = torch.tensor([int(e.get("aid", e["a"] + 1)) for e in b],
                               dtype=torch.long, device=dev).clamp(0, N_ACTIONS - 1)
            rew = torch.tensor([e["r"] for e in b], dtype=torch.float32, device=dev)
            st2 = torch.tensor([int(e.get("st2", 1)) for e in b],
                               dtype=torch.long, device=dev).clamp(0, N_STATES - 1)

            cur_tok = tok.tokens_of(cur)                          # [B,64]
            nxt_tok = tok.tokens_of(nxt)                          # [B,64]
            tok_logits, reward, state_logits = twm(cur_tok, aid, nxt_tok)

            l_tok = F.cross_entropy(tok_logits.reshape(-1, CODEBOOK), nxt_tok.reshape(-1))
            l_rew = F.mse_loss(reward, rew)
            l_state = F.cross_entropy(state_logits, st2)
            loss = l_tok + l_rew + l_state
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(twm.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                top1 = (tok_logits.argmax(-1) == nxt_tok).float().mean().item()
                sacc = (state_logits.argmax(-1) == st2).float().mean().item()
            last = {"epoch": ep, "loss": loss.item(), "tok": l_tok.item(),
                    "rew": float(l_rew.detach()), "state": l_state.item(),
                    "top1": top1, "sacc": sacc}
            if (bi % log_every) == 0:
                print(f"  [twm] ep{ep} step{bi}/{steps_per_epoch} "
                      f"loss={last['loss']:.3f} tok={last['tok']:.3f} "
                      f"rew={last['rew']:.4f} state={last['state']:.3f} "
                      f"top1={top1:.3f} sacc={sacc:.3f}")
    return last


# =====================================================================
# Conv grid->grid world model training (change-weighted loss)
#
# The whole failure of the VQ route was that loss was dominated by the
# ~99% static background, so the model learned near-identity copying.
# Here we up-weight CHANGED cells (next != current) in the per-cell CE so
# the model is actually pushed to predict what moves. We report held-out
# changed-cell accuracy -- the only number that proved diagnostic before.
# =====================================================================
def train_cwm(cwm, buffer, epochs=6, batch=64, lr=3e-4, device="cpu",
              change_weight=20.0, log_every=50):
    from cwm import N_ACTIONS as CWM_NA, N_STATES as CWM_NS
    dev = torch.device(device)
    cwm.to(dev)
    opt = optim.Adam(cwm.parameters(), lr=lr)
    items = [e for e in buffer
             if np.asarray(e["s"]).shape == (G, G)
             and np.asarray(e["s2"]).shape == (G, G)]
    n = len(items)
    if n < batch:
        raise ValueError(f"too few valid transitions ({n} < {batch})")
    steps_per_epoch = max(1, n // batch)
    last = {}
    for ep in range(epochs):
        order = np.random.permutation(n)
        for bi in range(steps_per_epoch):
            sel = order[bi * batch:(bi + 1) * batch]
            if len(sel) < batch:
                break
            b = [items[i] for i in sel]
            cur = grids_to_long(np.stack([np.clip(np.asarray(e["s"], dtype=np.int64), 0, 15) for e in b])).to(dev)
            nxt = grids_to_long(np.stack([np.clip(np.asarray(e["s2"], dtype=np.int64), 0, 15) for e in b])).to(dev)
            aid = torch.tensor([int(e.get("aid", e["a"] + 1)) for e in b],
                               dtype=torch.long, device=dev).clamp(0, CWM_NA - 1)
            rew = torch.tensor([e["r"] for e in b], dtype=torch.float32, device=dev)
            st2 = torch.tensor([int(e.get("st2", 1)) for e in b],
                               dtype=torch.long, device=dev).clamp(0, CWM_NS - 1)
            # click coords as [B,2]; (-1,-1) means "no click" -> empty plane
            cxy = torch.tensor([list(e["cxy"]) if e.get("cxy") else [-1, -1] for e in b],
                               dtype=torch.long, device=dev)

            cell_logits, reward, state_logits = cwm(cur, aid, cxy)
            # per-cell CE, weighted up where the cell actually changed
            ce = F.cross_entropy(cell_logits, nxt, reduction="none")     # [B,64,64]
            changed = (cur != nxt).float()
            w = 1.0 + change_weight * changed
            l_cell = (ce * w).sum() / w.sum()
            l_rew = F.mse_loss(reward, rew)
            l_state = F.cross_entropy(state_logits, st2)
            loss = l_cell + l_rew + l_state
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(cwm.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                pred = cell_logits.argmax(1)
                cm = changed.bool()
                acc_ch = (pred[cm] == nxt[cm]).float().mean().item() if cm.any() else 0.0
                sacc = (state_logits.argmax(-1) == st2).float().mean().item()
                # CLICK-only changed-cell accuracy (the action we are fixing)
                clk = (aid == 6)
                clk_cm = cm & clk[:, None, None]
                acc_clk = (pred[clk_cm] == nxt[clk_cm]).float().mean().item() if clk_cm.any() else float("nan")
            last = {"epoch": ep, "loss": loss.item(), "cell": l_cell.item(),
                    "rew": float(l_rew.detach()), "state": l_state.item(),
                    "acc_changed": acc_ch, "acc_click": acc_clk, "sacc": sacc}
            if (bi % log_every) == 0:
                print(f"  [cwm] ep{ep} step{bi}/{steps_per_epoch} "
                      f"loss={last['loss']:.3f} cell={last['cell']:.3f} "
                      f"rew={last['rew']:.4f} state={last['state']:.3f} "
                      f"acc_changed={acc_ch:.3f} acc_click={acc_clk:.3f} sacc={sacc:.3f}")
    return last


# =====================================================================
# DreamerV3-style RSSM training (change-weighted, for a fair head-to-head)
#
# Mirrors the RSSM eval path: observe(s) -> posterior (h0,z0); then step
# under action a WITH the posterior from s2, decode -> reconstruct s2,
# predict reward + state. KL(post||prior) trains the PRIOR to match the
# posterior, so at eval time imagine_step (prior-only, no s2) predicts well.
#
# We use the SAME change-weighted CE as train_cwm so the comparison
# isolates architecture (latent RSSM vs conv-skip), not the loss.
# =====================================================================
def train_dreamer(dm, buffer, epochs=6, batch=64, lr=3e-4, device="cpu",
                  change_weight=20.0, kl_scale=1.0, log_every=50):
    from dreamer import grids_to_long as d_g2l, kl_loss, N_ACTIONS as DNA, N_STATES as DNS
    dev = torch.device(device)
    dm.to(dev)
    opt = optim.Adam(dm.parameters(), lr=lr)
    items = [e for e in buffer
             if np.asarray(e["s"]).shape == (G, G)
             and np.asarray(e["s2"]).shape == (G, G)]
    n = len(items)
    if n < batch:
        raise ValueError(f"too few valid transitions ({n} < {batch})")
    steps_per_epoch = max(1, n // batch)
    last = {}
    for ep in range(epochs):
        order = np.random.permutation(n)
        for bi in range(steps_per_epoch):
            sel = order[bi * batch:(bi + 1) * batch]
            if len(sel) < batch:
                break
            b = [items[i] for i in sel]
            cur = d_g2l(np.stack([np.clip(np.asarray(e["s"], dtype=np.int64), 0, 15) for e in b])).to(dev)
            nxt = d_g2l(np.stack([np.clip(np.asarray(e["s2"], dtype=np.int64), 0, 15) for e in b])).to(dev)
            aid = torch.tensor([int(e.get("aid", e["a"] + 1)) for e in b],
                               dtype=torch.long, device=dev).clamp(0, DNA - 1)
            rew = torch.tensor([e["r"] for e in b], dtype=torch.float32, device=dev)
            st2 = torch.tensor([int(e.get("st2", 1)) for e in b],
                               dtype=torch.long, device=dev).clamp(0, DNS - 1)

            # t=0: fold current frame into posterior state (action RESET)
            h0, z0 = dm.observe(cur)
            # t=1: step under action a, posterior corrected by observing s2
            e_nxt = dm.enc(nxt)
            h1, z1, prior_l, post_l = dm.rssm.step(h0, z0, aid, embed=e_nxt)
            feat = dm._feat(h1, z1)
            cell_logits = dm.dec(feat)
            reward = dm.head_rew(feat).squeeze(-1)
            state_logits = dm.head_state(feat)

            ce = F.cross_entropy(cell_logits, nxt, reduction="none")      # [B,64,64]
            changed = (cur != nxt).float()
            w = 1.0 + change_weight * changed
            l_cell = (ce * w).sum() / w.sum()
            l_rew = F.mse_loss(reward, rew)
            l_state = F.cross_entropy(state_logits, st2)
            l_kl = kl_loss(post_l, prior_l)
            loss = l_cell + l_rew + l_state + kl_scale * l_kl
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(dm.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                # eval-path accuracy: predict from the PRIOR (no s2 observed)
                _h2, _z2, cl_im, _r, _sl = dm.imagine_step(h0, z0, aid)
                pred = cl_im.argmax(1)
                cm = changed.bool()
                acc_ch = (pred[cm] == nxt[cm]).float().mean().item() if cm.any() else 0.0
                sacc = (state_logits.argmax(-1) == st2).float().mean().item()
            last = {"epoch": ep, "loss": loss.item(), "cell": l_cell.item(),
                    "rew": float(l_rew.detach()), "state": l_state.item(),
                    "kl": float(l_kl.detach()), "acc_changed": acc_ch, "sacc": sacc}
            if (bi % log_every) == 0:
                print(f"  [dreamer] ep{ep} step{bi}/{steps_per_epoch} "
                      f"loss={last['loss']:.3f} cell={last['cell']:.3f} "
                      f"kl={last['kl']:.3f} rew={last['rew']:.4f} "
                      f"state={last['state']:.3f} acc_changed={acc_ch:.3f} sacc={sacc:.3f}")
    return last


# =====================================================================
# Engine-driven data collection  (needs arc_agi installed + game files)
# =====================================================================
def _grid_of(obs):
    arr = np.array(obs.frame)
    while arr.ndim > 2:
        arr = arr[-1]
    return np.clip(arr.astype(np.int64), 0, 15)


# TWM state head order, matches twm.STATE_NAMES / arcengine GameState
_STATE_ORDER = ["NOT_PLAYED", "NOT_FINISHED", "WIN", "GAME_OVER"]


def _state_idx(obs):
    """GameState on obs -> index into _STATE_ORDER (default NOT_FINISHED)."""
    st = getattr(obs, "state", None)
    name = getattr(st, "name", None) or str(st)
    try:
        return _STATE_ORDER.index(name)
    except ValueError:
        return 1                                   # NOT_FINISHED fallback


def _valid_grid(g):
    """True only for a real 64x64 frame (engine emits empty (0,) frames at
    some reset/boundary steps -- those must be skipped, not tokenized)."""
    return isinstance(g, np.ndarray) and g.shape == (G, G)


def collect(games=None, steps_per_game=4000, seed=0, env_dir="environment_files"):
    # The interactive engine is NOT the public `arc-agi` PyPI package (that
    # is the static ARC-1/2 puzzle dataset). It ships in the competition
    # wheels: arcengine + the arc_agi-0.9.x with Arcade/EnvironmentWrapper.
    # Install those wheels into a Python 3.12 env, e.g.:
    #   pip install --no-index --find-links arc_agi_3_wheels arc-agi
    try:
        from arc_agi import Arcade, OperationMode
        from arcengine import GameState
    except Exception as e:
        sys.exit(
            f"interactive engine not importable ({e}).\n"
            f"This needs the COMPETITION wheels (arcengine + arc_agi-0.9.x), "
            f"NOT `pip install arc-agi` (that's the static-puzzle library).\n"
            f"Install into a Python 3.12 env:\n"
            f"  pip install --no-index --find-links arc_agi_3_wheels arc-agi")

    rng = np.random.default_rng(seed)
    # OFFLINE: only the local environment_files games, no API/key.
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir)
    envs = arc.get_environments()
    all_ids = sorted({g.game_id.split("-", 1)[0] for g in envs})   # base ids
    if not all_ids:
        sys.exit(f"no offline games found under {env_dir}/. "
                 f"Run where environment_files/ lives.")
    target = games or all_ids
    print(f"offline games available ({len(all_ids)}): {all_ids}")
    print(f"collecting from: {target}")

    def progress(obs):
        # FrameDataRaw has NO score; progress is levels + win flag.
        lv = getattr(obs, "levels_completed", 0) or 0
        won = 1 if getattr(obs, "state", None) == GameState.WIN else 0
        return lv, won

    buf = []
    for gid in target:
        env = arc.make(gid)                       # no render_mode = fastest
        if env is None:
            print(f"  skip {gid}: make() returned None"); continue
        obs = env.observation_space               # make() already reset()s
        if obs is None:
            obs = env.reset()
        if obs is None:
            print(f"  skip {gid}: no initial frame"); continue
        cur = _grid_of(obs)
        prev_lv, _ = progress(obs)
        t0 = time.time()
        got = 0
        for _ in range(steps_per_game):
            space = list(env.action_space)        # only AVAILABLE actions
            if not space:
                obs = env.reset(); cur = _grid_of(obs); prev_lv, _ = progress(obs)
                continue
            act = space[rng.integers(len(space))]
            aid = act.value                        # GameAction enum value == id
            data = None
            if act.is_complex():                   # ACTION6 needs x,y
                data = {"x": int(rng.integers(0, G)), "y": int(rng.integers(0, G))}
            obs = env.step(act, data=data)
            if obs is None:
                obs = env.reset(); cur = _grid_of(obs); prev_lv, _ = progress(obs)
                continue
            nxt = _grid_of(obs)
            lv, won = progress(obs)
            st2 = _state_idx(obs)
            r = 5.0 * max(0, lv - prev_lv) + (10.0 if won else 0.0)
            if not np.array_equal(cur, nxt):
                r += 0.1                           # mild change bonus
            prev_lv = lv
            a_idx = (aid - 1) if 1 <= aid <= 5 else -1   # legacy latent-WM field (discrete only)
            cxy = (int(data["x"]), int(data["y"])) if data else None   # click coords (ACTION6)
            if _valid_grid(cur) and _valid_grid(nxt):    # skip empty boundary frames
                buf.append({"s": cur.astype(np.int8), "a": a_idx, "aid": int(aid), "r": r,
                            "s2": nxt.astype(np.int8), "st2": st2, "cxy": cxy})
                got += 1
            cur = nxt
            if won or getattr(obs, "state", None) == GameState.GAME_OVER:
                obs = env.reset(); cur = _grid_of(obs); prev_lv, _ = progress(obs)
        print(f"  {gid}: {got} transitions in {time.time()-t0:.1f}s")
    return buf


def save_buffer(buf, path):
    np.savez_compressed(
        path,
        s=np.stack([b["s"] for b in buf]),
        a=np.array([b["a"] for b in buf], dtype=np.int16),
        aid=np.array([b.get("aid", b["a"] + 1) for b in buf], dtype=np.int16),
        r=np.array([b["r"] for b in buf], dtype=np.float32),
        s2=np.stack([b["s2"] for b in buf]),
        st2=np.array([b.get("st2", 1) for b in buf], dtype=np.int16),
    )
    print(f"saved buffer ({len(buf)}) -> {path}")


def load_buffer(path):
    d = np.load(path)
    has_aid = "aid" in d
    has_st2 = "st2" in d
    out = []
    for i in range(len(d["a"])):
        a = int(d["a"][i])
        out.append({
            "s": d["s"][i], "a": a, "r": float(d["r"][i]), "s2": d["s2"][i],
            "aid": int(d["aid"][i]) if has_aid else a + 1,
            "st2": int(d["st2"][i]) if has_st2 else 1,
        })
    return out


def main():
    ap = argparse.ArgumentParser(description="Offline pretraining for the Dyna-WM agent.")
    ap.add_argument("--games", nargs="*", default=None, help="game ids (default: all offline)")
    ap.add_argument("--steps", type=int, default=4000, help="env steps per game")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="pretrained_weights.pt")
    ap.add_argument("--buffer", default=None, help="train from a saved .npz buffer (skip collection)")
    ap.add_argument("--save-buffer", default=None, help="dump collected buffer to this .npz")
    ap.add_argument("--seed", type=int, default=0)
    # IRIS-style token world model mode
    ap.add_argument("--twm", action="store_true",
                    help="train the IRIS-style token world model (tokenizer + GPT TWM) "
                         "instead of the legacy latent WM")
    ap.add_argument("--tok-epochs", type=int, default=4, help="tokenizer epochs (--twm)")
    ap.add_argument("--tok-batch", type=int, default=128, help="tokenizer batch (--twm)")
    ap.add_argument("--twm-epochs", type=int, default=4, help="TWM epochs (--twm)")
    ap.add_argument("--twm-batch", type=int, default=64, help="TWM batch (--twm)")
    ap.add_argument("--tok-out", default="tokenizer.pt", help="tokenizer weights out (--twm)")
    ap.add_argument("--twm-out", default="twm.pt", help="token world model weights out (--twm)")
    # conv grid->grid world model mode (no tokenizer)
    ap.add_argument("--cwm", action="store_true",
                    help="train the conv grid->grid world model (no tokenizer, "
                         "change-weighted loss) -- the recommended WM")
    ap.add_argument("--cwm-epochs", type=int, default=6, help="cwm epochs (--cwm)")
    ap.add_argument("--cwm-batch", type=int, default=64, help="cwm batch (--cwm)")
    ap.add_argument("--change-weight", type=float, default=20.0,
                    help="up-weight on changed cells in the cwm loss (--cwm)")
    ap.add_argument("--cwm-out", default="cwm.pt", help="conv world model weights out (--cwm)")
    # compact DreamerV3-style RSSM mode
    ap.add_argument("--dreamer", action="store_true",
                    help="train the compact DreamerV3-style RSSM world model "
                         "(latent dynamics) for head-to-head vs the cwm")
    ap.add_argument("--dreamer-epochs", type=int, default=6, help="dreamer epochs (--dreamer)")
    ap.add_argument("--dreamer-batch", type=int, default=64, help="dreamer batch (--dreamer)")
    ap.add_argument("--dreamer-out", default="dreamer.pt", help="dreamer weights out (--dreamer)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    if args.buffer:
        buf = load_buffer(args.buffer)
        print(f"loaded buffer ({len(buf)}) from {args.buffer}")
    else:
        buf = collect(args.games, args.steps, args.seed)
        if args.save_buffer:
            save_buffer(buf, args.save_buffer)
    if not buf:
        sys.exit("empty buffer, nothing to train")

    if args.cwm:
        from cwm import ConvWorldModel
        cwm = ConvWorldModel()
        clast = train_cwm(cwm, buf, epochs=args.cwm_epochs, batch=args.cwm_batch,
                          change_weight=args.change_weight, device=device)
        torch.save(cwm.state_dict(), args.cwm_out)
        print(f"cwm final {clast}")
        print(f"saved conv world model -> {args.cwm_out}")
        print(f"replay with: python replay_wm.py --recording <rec> --cwm {args.cwm_out}")
        return

    if args.dreamer:
        from dreamer import Dreamer
        dm = Dreamer()
        dlast = train_dreamer(dm, buf, epochs=args.dreamer_epochs,
                              batch=args.dreamer_batch,
                              change_weight=args.change_weight, device=device)
        torch.save(dm.state_dict(), args.dreamer_out)
        print(f"dreamer final {dlast}")
        print(f"saved dreamer world model -> {args.dreamer_out}")
        return

    if args.twm:
        # Stage 1: tokenizer
        tok = Tokenizer()
        tlast = train_tokenizer(tok, buf, epochs=args.tok_epochs,
                                batch=args.tok_batch, device=device)
        torch.save(tok.state_dict(), args.tok_out)
        print(f"tokenizer final {tlast}")
        print(f"saved tokenizer -> {args.tok_out}")
        # Stage 2: token world model (tokenizer frozen)
        twm = TokenWorldModel()
        wlast = train_twm(twm, tok, buf, epochs=args.twm_epochs,
                          batch=args.twm_batch, device=device)
        torch.save(twm.state_dict(), args.twm_out)
        print(f"twm final {wlast}")
        print(f"saved token world model -> {args.twm_out}")
        print(f"replay with: python replay_wm.py <recording> "
              f"--tokenizer {args.tok_out} --twm {args.twm_out}")
        return

    net = Net()
    last = train_on_buffer(net, buf, epochs=args.epochs, batch=args.batch, device=device)
    torch.save(net.state_dict(), args.out)
    print(f"final {last}")
    print(f"saved weights -> {args.out}  (load via ARC_PRETRAINED={args.out} or place next to my_agent.py)")


if __name__ == "__main__":
    main()
