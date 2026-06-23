#!/usr/bin/env python3
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch

from ls20_model import Ls20WorldModel
from ls20_wrapper import Ls20Wrapper


ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
STATE_ORDER = ["NOT_PLAYED", "NOT_FINISHED", "WIN", "GAME_OVER"]


def _state_idx(obs):
    st = getattr(obs, "state", None)
    name = getattr(st, "name", None) or str(st)
    try:
        return STATE_ORDER.index(name)
    except ValueError:
        return 1


def _load_model(path, device):
    if not os.path.exists(path):
        sys.exit(f"Missing model checkpoint: {path}")

    saved = torch.load(path, map_location=device, weights_only=False)
    model = Ls20WorldModel(num_tokens=saved["num_tokens"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved


def _attach_tokenizer(wrapper, saved):
    wrapper.tokenizer.id_to_block = saved["id_to_block"]
    wrapper.tokenizer.block_to_id = saved["block_to_id"]
    wrapper.tokenizer.next_id = saved["num_tokens"]


def _clip_tokens(tokens, num_tokens):
    return np.clip(tokens, 0, num_tokens - 1)


def _add_metric(hit_tot, pred, true, changed_mask):
    hit_tot["all_hit"] += int((pred == true).sum())
    hit_tot["all_tot"] += int(true.size)
    if changed_mask.any():
        hit_tot["changed_hit"] += int((pred[changed_mask] == true[changed_mask]).sum())
        hit_tot["changed_tot"] += int(changed_mask.sum())


def _ratio(hit, total):
    return hit / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ls20_model.pt")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--env-dir", default="environment_files")
    args = ap.parse_args()

    try:
        from arc_agi import Arcade, OperationMode
    except Exception as e:
        sys.exit(f"arc_agi is not importable: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, saved = _load_model(args.model, device)
    num_tokens = int(saved["num_tokens"])

    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=args.env_dir)
    env = arc.make("ls20")
    if env is None:
        sys.exit("Could not load ls20 from environment_files")

    wrapper = Ls20Wrapper(env)
    _attach_tokenizer(wrapper, saved)

    rng = np.random.default_rng(args.seed)
    obs = env.reset()
    board, gui = wrapper._process_frame(obs.frame)
    board = _clip_tokens(board, num_tokens)

    board_m = defaultdict(int)
    gui_m = defaultdict(int)
    per_action = defaultdict(lambda: defaultdict(int))
    exact_board = exact_gui = exact_both = total = 0
    state_hit = state_tot = 0
    unknown_targets = 0

    with torch.no_grad():
        while total < args.steps:
            act_id = int(rng.integers(1, 5))

            t_board = torch.tensor(board, dtype=torch.long, device=device).unsqueeze(0)
            t_gui = torch.tensor(gui, dtype=torch.long, device=device).unsqueeze(0)
            t_act = torch.tensor([act_id], dtype=torch.long, device=device)
            next_b_logits, next_g_logits, _, state_logits = model(t_board, t_gui, t_act)
            pred_b = next_b_logits.argmax(1).squeeze(0).cpu().numpy()
            pred_g = next_g_logits.argmax(1).squeeze(0).cpu().numpy()
            pred_state = int(state_logits.argmax(-1).item())

            b2, g2, _, _, _ = wrapper.step_dir(act_id)
            raw_obs = wrapper.env.observation_space
            if b2 is None:
                obs = env.reset()
                board, gui = wrapper._process_frame(obs.frame)
                board = _clip_tokens(board, num_tokens)
                continue

            unknown_targets += int((b2 >= num_tokens).sum())
            b2 = _clip_tokens(b2, num_tokens)

            ch_b = board != b2
            ch_g = gui != g2
            _add_metric(board_m, pred_b, b2, ch_b)
            _add_metric(gui_m, pred_g, g2, ch_g)
            _add_metric(per_action[act_id], pred_b, b2, ch_b)

            exact_board += int(np.array_equal(pred_b, b2))
            exact_gui += int(np.array_equal(pred_g, g2))
            exact_both += int(np.array_equal(pred_b, b2) and np.array_equal(pred_g, g2))
            state_hit += int(pred_state == _state_idx(raw_obs))
            state_tot += 1
            total += 1

            board, gui = b2, g2

    print("\n================ LS20 WM vs Ground Truth ================")
    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"steps: {total}")
    print(f"num_tokens: {num_tokens}")
    print(f"unknown target board tokens clipped: {unknown_targets}")
    print()
    print(f"{'metric':30s} {'board':>8s} {'gui':>8s}")
    print(f"{'all-cell acc':30s} {_ratio(board_m['all_hit'], board_m['all_tot']):8.3f} {_ratio(gui_m['all_hit'], gui_m['all_tot']):8.3f}")
    print(f"{'changed-cell acc':30s} {_ratio(board_m['changed_hit'], board_m['changed_tot']):8.3f} {_ratio(gui_m['changed_hit'], gui_m['changed_tot']):8.3f}")
    print(f"{'exact next-state acc':30s} {_ratio(exact_board, total):8.3f} {_ratio(exact_gui, total):8.3f}")
    print(f"{'exact board+gui acc':30s} {_ratio(exact_both, total):8.3f}")
    print(f"{'state acc':30s} {_ratio(state_hit, state_tot):8.3f}")
    print()
    print("per-action board changed-cell acc:")
    for act_id in sorted(per_action):
        m = per_action[act_id]
        print(f"  {ACTION_NAMES.get(act_id, str(act_id)):5s}: {_ratio(m['changed_hit'], m['changed_tot']):.3f}")


if __name__ == "__main__":
    main()
