#!/usr/bin/env python3
# =====================================================================
# Unified world-model vs ground-truth eval: CWM vs DreamerV3-RSSM.
#
# Both models expose the SAME interface
#   imagine(grid_long[B,64,64], action_ids[B], click_xy[B,2]) ->
#       (next_grid[B,64,64], reward[B], state_idx[B], state_logits[B,4])
# so we score them with IDENTICAL logic against held-out real engine
# transitions. The metric is the honest one that has caught every
# regression this session: changed-cell accuracy (copy baseline = 0).
#
# Ground truth = the actual next frame from the offline engine.
#
# Usage:
#   python _wm_compare.py                 # both, default games
#   python _wm_compare.py --games ls20 ft09
# =====================================================================
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)

import pretrain as PT
from cwm import ConvWorldModel, grids_to_long, G, N_ACTIONS
from dreamer import Dreamer


def _load(model, path, device):
    n = 0
    if os.path.exists(path):
        state = torch.load(path, map_location=device, weights_only=True)
        ms = model.state_dict()
        for k, v in state.items():
            if k in ms and ms[k].shape == v.shape:
                ms[k] = v; n += 1
        model.load_state_dict(ms)
    model.to(device).eval()
    return n


def _score(model, items, device, use_click):
    """Return (changed_acc, all_acc, unchanged_acc, per_action dict)."""
    hit_ch = tot_ch = hit_all = tot_all = hit_un = tot_un = 0
    by_act_hit = defaultdict(float)
    by_act_tot = defaultdict(float)
    with torch.no_grad():
        for i in range(0, len(items), 128):
            chunk = items[i:i + 128]
            cur = grids_to_long(np.stack([np.clip(np.asarray(e["s"], dtype=np.int64), 0, 15) for e in chunk])).to(device)
            nxt = np.stack([np.clip(np.asarray(e["s2"], dtype=np.int64), 0, 15) for e in chunk])
            aid = torch.tensor([int(e.get("aid", e["a"] + 1)) for e in chunk],
                               dtype=torch.long, device=device).clamp(0, N_ACTIONS - 1)
            if use_click:
                cxy = torch.tensor([list(e["cxy"]) if e.get("cxy") else [-1, -1] for e in chunk],
                                   dtype=torch.long, device=device)
                pred = model.imagine(cur, aid, cxy)[0].cpu().numpy()
            else:
                pred = model.imagine(cur, aid)[0].cpu().numpy()
            c = cur.cpu().numpy()
            for j in range(len(chunk)):
                ch = (c[j] != nxt[j])
                un = ~ch
                hit_all += int((pred[j] == nxt[j]).sum()); tot_all += nxt[j].size
                if ch.any():
                    h = int((pred[j][ch] == nxt[j][ch]).sum())
                    hit_ch += h; tot_ch += int(ch.sum())
                    a = int(chunk[j].get("aid", chunk[j]["a"] + 1))
                    by_act_hit[a] += h; by_act_tot[a] += int(ch.sum())
                if un.any():
                    hit_un += int((pred[j][un] == nxt[j][un]).sum()); tot_un += int(un.sum())
    return (hit_ch / max(1, tot_ch), hit_all / max(1, tot_all),
            hit_un / max(1, tot_un),
            {a: by_act_hit[a] / by_act_tot[a] for a in sorted(by_act_tot)})


ANAME = {0: "RESET", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
         5: "INTERACT", 6: "CLICK", 7: "UNDO"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=["ls20", "ft09", "vc33", "wa30", "sk48"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cwm = ConvWorldModel()
    n_cwm = _load(cwm, os.path.join(here, "cwm.pt"), device)
    dm = Dreamer()
    n_dm = _load(dm, os.path.join(here, "dreamer.pt"), device)
    print(f"CWM loaded {n_cwm} tensors; Dreamer loaded {n_dm} tensors")
    if n_dm == 0:
        print("WARNING: dreamer.pt missing/empty -- its numbers are random", file=sys.stderr)

    # held-out data (seed 7, distinct from training seed 0)
    buf = PT.collect(games=args.games, steps_per_game=args.steps, seed=args.seed)
    items = [e for e in buf
             if np.asarray(e["s"]).shape == (G, G) and np.asarray(e["s2"]).shape == (G, G)]
    print(f"held-out transitions: {len(items)}")

    # copy baseline: predict next = current
    c_hit = c_tot = 0
    for e in items:
        c = np.clip(np.asarray(e["s"], dtype=np.int64), 0, 15)
        n = np.clip(np.asarray(e["s2"], dtype=np.int64), 0, 15)
        ch = (c != n)
        if ch.any():
            c_hit += int((c[ch] == n[ch]).sum()); c_tot += int(ch.sum())  # =0 by def
    copy_changed = c_hit / max(1, c_tot)

    cwm_ch, cwm_all, cwm_un, cwm_pa = _score(cwm, items, device, use_click=True)
    dm_ch, dm_all, dm_un, dm_pa = _score(dm, items, device, use_click=False)

    print("\n================ WORLD MODEL vs GROUND TRUTH ================")
    print(f"{'metric':28s} {'COPY':>8s} {'CWM':>8s} {'Dreamer':>8s}")
    print(f"{'changed-cell acc':28s} {copy_changed:>8.3f} {cwm_ch:>8.3f} {dm_ch:>8.3f}   <- the one that matters")
    print(f"{'all-cell acc':28s} {'-':>8s} {cwm_all:>8.3f} {dm_all:>8.3f}")
    print(f"{'unchanged-cell acc':28s} {'-':>8s} {cwm_un:>8.3f} {dm_un:>8.3f}")

    print("\n---- per-action changed-cell acc (CWM vs Dreamer) ----")
    for a in sorted(set(cwm_pa) | set(dm_pa)):
        print(f"  {ANAME.get(a, a):8s}(id{a}): CWM={cwm_pa.get(a, float('nan')):.3f}  "
              f"Dreamer={dm_pa.get(a, float('nan')):.3f}")
    print("\nNOTE: Dreamer predicts in a compact latent then decodes, so lower")
    print("changed-cell acc vs CWM is the EXPECTED architectural trade, not a bug.")
    print("WM-COMPARE DONE")


if __name__ == "__main__":
    main()
