#!/usr/bin/env python3
# =====================================================================
# Conv world-model comparison replay for ARC-AGI-3 recordings.
#
# For each transition (frame i --action--> frame i+1) it renders a
# three-panel row:
#
#   [ actual next ]   [ CWM predicted ]   [ diff ]
#       (sharp)           (sharp)        (red = CWM got it wrong)
#
# The middle panel is the conv grid->grid world model imagining the next
# frame: feed frame_i + action through one forward pass -> per-cell
# next-frame logits -> argmax to a grid. No tokenizer, no autoregressive
# sampling. Predictions are SHARP and lossless-capable, because the U-Net
# skip connections carry exact spatial detail past the bottleneck instead
# of rounding it to a shared VQ code.
#
# The CWM models all 8 actions (RESET + ACTION1..7), so clicks/undo are
# predicted too -- no reconstruction fallback needed.
#
# It also predicts the resulting game STATE (NOT_PLAYED / NOT_FINISHED /
# WIN / GAME_OVER), shown under each row next to the recorded state.
#
# Usage:
#   python replay_wm.py --recording REC.jsonl [--cwm cwm.pt]
#   python replay_wm.py --recording DIR        # newest *.recording.jsonl
#   python replay_wm.py --recording REC.jsonl --scale 6 --fps 4 --png-dir frames/
# =====================================================================
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

from cwm import ConvWorldModel, grids_to_long, G, STATE_NAMES
from replay import (_load_font, _grid_from_entry, _palette_img,
                    _action_label, load_entries, resolve_path,
                    _ensure_parent, _default_replay_out)


# action name/id -> GameAction id 0..7 (RESET=0, ACTION1..7=1..7)
_ACTION_ID = {"RESET": 0, "ACTION1": 1, "ACTION2": 2, "ACTION3": 3,
              "ACTION4": 4, "ACTION5": 5, "ACTION6": 6, "ACTION7": 7}


def _find(explicit, env, default):
    for p in (explicit, os.getenv(env, ""), default):
        if p and os.path.exists(p):
            return p
    return None


def _load_model(cwm_path, device):
    cwm = ConvWorldModel().to(device)
    n = 0
    if cwm_path:
        state = torch.load(cwm_path, map_location=device, weights_only=True)
        ms = cwm.state_dict()
        for k, v in state.items():
            if k in ms and ms[k].shape == v.shape:
                ms[k] = v; n += 1
        cwm.load_state_dict(ms)
    cwm.eval()
    return cwm, n


def _action_id(ai):
    """GameAction id 0..7 for this entry's action, or None if unknown."""
    if not ai:
        return None
    aid = ai.get("id")
    if isinstance(aid, str):                       # recordings store the name
        return _ACTION_ID.get(aid)
    if isinstance(aid, int) and 0 <= aid <= 7:
        return aid
    return None


def _click_xy(ai):
    """Click (x,y) from this entry's action_input.data, or None if not a click."""
    if not ai:
        return None
    d = ai.get("data") or {}
    if "x" in d and "y" in d:
        return (int(d["x"]), int(d["y"]))
    return None


@torch.no_grad()
def _predict_next(cwm, grid_cur, a_id, click_xy, device):
    """Return (predicted_next_grid [64,64] int, predicted_state_name).
    One forward pass of the conv world model for the taken action + click."""
    cur = grids_to_long(grid_cur).to(device)                 # [1,64,64]
    a = torch.tensor([a_id if a_id is not None else 0],
                     dtype=torch.long, device=device)
    cxy = torch.tensor([list(click_xy) if click_xy else [-1, -1]],
                       dtype=torch.long, device=device)
    next_grid, _reward, state_idx, _ = cwm.imagine(cur, a, cxy)
    grid = next_grid[0].cpu().numpy().astype(int)
    state_name = STATE_NAMES[int(state_idx[0])]
    return grid, state_name


def _diff_img(actual, predicted, scale):
    """Grey where they agree, red where the TWM prediction missed."""
    h, w = actual.shape
    rgb = np.full((h, w, 3), 40, dtype=np.uint8)
    miss = actual != predicted
    rgb[~miss] = (60, 60, 60)
    rgb[miss] = (230, 40, 40)
    return Image.fromarray(rgb, "RGB").resize((w * scale, h * scale), Image.NEAREST)


def _label(draw, x, y, text, font, fill=(235, 235, 235)):
    draw.text((x, y), text, fill=fill, font=font)


def render_row(entry_prev, entry_cur, cwm, device, scale, font, font_sm):
    """One comparison row for the transition entry_prev --action--> entry_cur.
    entry_cur.action_input is the action that produced entry_cur from entry_prev."""
    grid_prev = _grid_from_entry(entry_prev.get("data", entry_prev))
    grid_cur = _grid_from_entry(entry_cur.get("data", entry_cur))
    if grid_prev is None or grid_cur is None:
        return None
    if grid_prev.shape != (G, G) or grid_cur.shape != (G, G):
        return None                                # CWM needs full 64x64 frames

    data_cur = entry_cur.get("data", entry_cur)
    ai = data_cur.get("action_input")
    a_id = _action_id(ai)
    cxy = _click_xy(ai)
    pred, pred_state = _predict_next(cwm, grid_prev, a_id, cxy, device)

    actual_img = _palette_img(grid_cur, scale)
    pred_img = _palette_img(pred, scale)
    diff_img = _diff_img(grid_cur, pred, scale)
    gw, gh = actual_img.size

    gap = 16
    header = 22
    panel = gw
    W = panel * 3 + gap * 2
    H = gh + header + 56
    canvas = Image.new("RGB", (W, H), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)

    name, _ = _action_label(ai)
    pct = 100.0 * float(np.mean(grid_cur == pred))

    for i, (img, lab) in enumerate([
        (actual_img, "actual next"),
        (pred_img, "CWM predicted"),
        (diff_img, f"diff  {pct:.0f}% match"),
    ]):
        x = i * (panel + gap)
        _label(draw, x, 4, lab, font_sm)
        canvas.paste(img, (x, header))

    step = entry_cur.get("_step", "?")
    actual_state = data_cur.get("state", "?")
    _label(draw, 0, header + gh + 6, f"step {step}   action {name}", font)
    ok = (str(actual_state) == pred_state)
    _label(draw, 0, header + gh + 28,
           f"state  actual={actual_state}  CWM={pred_state}", font_sm,
           fill=(150, 200, 150) if ok else (230, 120, 120))
    return canvas


def main():
    ap = argparse.ArgumentParser(description="Conv world-model comparison replay (actual vs CWM-predicted next frame + state).")
    ap.add_argument("--recording", help="path to .recording.jsonl (or a directory)")
    ap.add_argument("-o", "--out", help="output GIF path (default: <input>.wm.gif)")
    ap.add_argument("--cwm", default=None, help="conv world model weights (default: $ARC_CWM or cwm.pt)")
    ap.add_argument("--scale", type=int, default=6, help="pixels per grid cell")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--png-dir", help="also write each row as PNG here")
    ap.add_argument("--max-frames", type=int, default=0, help="cap rows (0 = all)")
    args = ap.parse_args()

    if not args.recording:
        sys.exit("pass --recording <path-or-dir>")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cwm_path = _find(args.cwm, "ARC_CWM", "cwm.pt")
    cwm, n_cwm = _load_model(cwm_path, device)
    if cwm_path:
        print(f"loaded CWM ({n_cwm} tensors) from {cwm_path}")
    else:
        print("WARNING: no CWM weights found -- predictions will be random "
              "(untrained). Pass --cwm or set ARC_CWM.", file=sys.stderr)

    path = resolve_path(args.recording)
    print(f"reading {path}")
    entries = load_entries(path)
    if len(entries) < 2:
        sys.exit("need at least 2 frames to show a transition")

    rows = []
    for i in range(1, len(entries)):
        row = render_row(entries[i - 1], entries[i], cwm, device,
                         args.scale, _load_font(15), _load_font(11))
        if row is not None:
            rows.append(row)
        if args.max_frames and len(rows) >= args.max_frames:
            break
    if not rows:
        sys.exit("no renderable transitions (need full 64x64 frames)")

    W = max(r.width for r in rows)
    H = max(r.height for r in rows)
    padded = []
    for r in rows:
        if r.size != (W, H):
            bg = Image.new("RGB", (W, H), (24, 24, 28))
            bg.paste(r, (0, 0))
            r = bg
        padded.append(r)

    out = args.out or _default_replay_out(path, ".wm.gif")
    _ensure_parent(out)
    dur = int(1000 / max(0.1, args.fps))
    padded[0].save(out, save_all=True, append_images=padded[1:],
                   duration=dur, loop=0, optimize=True)
    print(f"wrote {out}  ({len(padded)} rows, {args.fps} fps, scale {args.scale})")

    if args.png_dir:
        os.makedirs(args.png_dir, exist_ok=True)
        for i, r in enumerate(padded):
            r.save(os.path.join(args.png_dir, f"wm_{i:05d}.png"))
        print(f"wrote {len(padded)} PNGs to {args.png_dir}")


if __name__ == "__main__":
    main()
