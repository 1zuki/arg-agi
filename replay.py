#!/usr/bin/env python3
# =====================================================================
# Replay viewer for ARC-AGI-3 gameplay recordings.
#
# Reads the harness JSONL recordings (one frame per line) and renders
# them to an animated GIF (+ optional per-frame PNGs) using the ARC
# color palette, with a panel showing step / action / reasoning /
# score / state.
#
# Recording schema (one JSON object per line):
#   timestamp
#   data:
#     game_id, frame (grid or stack of grids), state, score,
#     action_input: { id, data:{x,y,...}, reasoning, guid }, full_reset
#
# Usage:
#   python replay.py RECORDING.jsonl [-o out.gif] [--scale 8] [--fps 8]
#   python replay.py DIR                 # picks newest *.recording.jsonl
#   python replay.py REC.jsonl --png-dir frames/
# =====================================================================
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ARC palette: 0-9 are the classic ARC colors, 10-15 extend it for
# ARC-AGI-3's 16-value grids. Distinct and roughly perceptually spread.
PALETTE = [
    (0, 0, 0),        # 0  black
    (0, 116, 217),    # 1  blue
    (255, 65, 54),    # 2  red
    (46, 204, 64),    # 3  green
    (255, 220, 0),    # 4  yellow
    (170, 170, 170),  # 5  grey
    (240, 18, 190),   # 6  magenta
    (255, 133, 27),   # 7  orange
    (127, 219, 255),  # 8  cyan
    (135, 12, 37),    # 9  maroon
    (1, 255, 196),    # 10 teal
    (177, 13, 201),   # 11 purple
    (133, 100, 4),    # 12 brown
    (255, 255, 255),  # 13 white
    (96, 96, 96),     # 14 dark grey
    (44, 60, 117),    # 15 navy
]

ACTION_NAMES = {
    0: "RESET", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
    5: "INTERACT", 6: "CLICK", 7: "UNDO",
}


def _load_font(size):
    for path in (
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    ):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _grid_from_entry(data):
    """Return the displayable 2D grid for one recording entry, or None."""
    frame = data.get("frame")
    if frame is None:
        return None
    arr = np.array(frame)
    while arr.ndim > 2:          # stack of grids -> take the last
        arr = arr[-1]
    if arr.ndim != 2 or arr.size == 0:
        return None
    return np.clip(arr.astype(int), 0, 15)


def _palette_img(grid, scale):
    h, w = grid.shape
    lut = np.array(PALETTE, dtype=np.uint8)        # [16,3]
    rgb = lut[grid]                                # [h,w,3]
    img = Image.fromarray(rgb, mode="RGB")
    return img.resize((w * scale, h * scale), Image.NEAREST)


def _action_label(ai):
    if not ai:
        return "(none)", ""
    aid = ai.get("id")
    name = ACTION_NAMES.get(aid, f"ACTION{aid}")
    d = ai.get("data") or {}
    if aid == 6 and "x" in d and "y" in d:
        name += f" ({d['x']},{d['y']})"
    return name, (ai.get("reasoning") or "")


def render_frame(entry, scale, panel_w, font, font_sm):
    data = entry.get("data", entry)
    grid = _grid_from_entry(data)
    if grid is None:
        return None
    grid_img = _palette_img(grid, scale)
    gw, gh = grid_img.size

    canvas = Image.new("RGB", (gw + panel_w, max(gh, 160)), (24, 24, 28))
    canvas.paste(grid_img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    name, reasoning = _action_label(data.get("action_input"))
    state = data.get("state", "?")
    score = data.get("score", 0)
    step = entry.get("_step", "?")

    x0 = gw + 12
    lines = [
        (f"step {step}", font),
        (f"state {state}", font_sm),
        (f"score {score}", font_sm),
        ("", font_sm),
        ("action", font_sm),
        (f"  {name}", font),
    ]
    y = 10
    for text, f in lines:
        draw.text((x0, y), text, fill=(235, 235, 235), font=f)
        y += (f.size + 6) if hasattr(f, "size") else 14

    # reasoning, wrapped
    if reasoning:
        draw.text((x0, y), "why", fill=(160, 160, 160), font=font_sm)
        y += (font_sm.size + 4) if hasattr(font_sm, "size") else 12
        wrap = max(8, panel_w // max(1, (font_sm.size // 2 if hasattr(font_sm, "size") else 6)))
        s = reasoning
        while s and y < canvas.height - 12:
            draw.text((x0, y), s[:wrap], fill=(150, 200, 150), font=font_sm)
            s = s[wrap:]
            y += (font_sm.size + 2) if hasattr(font_sm, "size") else 10

    return canvas


def load_entries(path):
    entries = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"  skip malformed line {ln}", file=sys.stderr)
                continue
            obj["_step"] = len(entries)
            entries.append(obj)
    return entries


def resolve_path(p):
    if os.path.isdir(p):
        cands = glob.glob(os.path.join(p, "**", "*.recording.jsonl"), recursive=True)
        if not cands:
            cands = glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True)
        if not cands:
            sys.exit(f"no recordings found under {p}")
        return max(cands, key=os.path.getmtime)
    return p


def main():
    ap = argparse.ArgumentParser(description="Replay an ARC-AGI-3 recording to GIF.")
    ap.add_argument("recording", help="path to .recording.jsonl (or a directory)")
    ap.add_argument("-o", "--out", help="output GIF path (default: alongside input)")
    ap.add_argument("--scale", type=int, default=8, help="pixels per grid cell")
    ap.add_argument("--fps", type=float, default=8.0, help="frames per second")
    ap.add_argument("--panel", type=int, default=240, help="info panel width px")
    ap.add_argument("--png-dir", help="also write each frame as PNG here")
    ap.add_argument("--max-frames", type=int, default=0, help="cap frames (0 = all)")
    args = ap.parse_args()

    path = resolve_path(args.recording)
    print(f"reading {path}")
    entries = load_entries(path)
    if not entries:
        sys.exit("no frames in recording")
    if args.max_frames:
        entries = entries[: args.max_frames]

    font = _load_font(16)
    font_sm = _load_font(12)

    frames = []
    for e in entries:
        img = render_frame(e, args.scale, args.panel, font, font_sm)
        if img is not None:
            frames.append(img)
    if not frames:
        sys.exit("no renderable frames (no grids found)")

    # pad all frames to a common size (grid dims can change between levels)
    W = max(f.width for f in frames)
    H = max(f.height for f in frames)
    padded = []
    for f in frames:
        if f.size != (W, H):
            bg = Image.new("RGB", (W, H), (24, 24, 28))
            bg.paste(f, (0, 0))
            f = bg
        padded.append(f)
    frames = padded

    out = args.out or os.path.splitext(path)[0].replace(".recording", "") + ".gif"
    dur_ms = int(1000 / max(0.1, args.fps))
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=dur_ms, loop=0, optimize=True)
    print(f"wrote {out}  ({len(frames)} frames, {args.fps} fps, scale {args.scale})")

    if args.png_dir:
        os.makedirs(args.png_dir, exist_ok=True)
        for i, f in enumerate(frames):
            f.save(os.path.join(args.png_dir, f"frame_{i:05d}.png"))
        print(f"wrote {len(frames)} PNGs to {args.png_dir}")


if __name__ == "__main__":
    main()
