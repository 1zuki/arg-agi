#!/usr/bin/env python3
import os
import sys
import time
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ls20_wrapper import Ls20Wrapper
from ls20_model import Ls20WorldModel
from pretrain_ls20 import train_ls20

ACTION_IDS = (1, 2, 3, 4)
ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
GAMMA = 0.97
WIN_BONUS = 100.0
GAME_OVER_PENALTY = 30.0
REWARD_SCALE = 4.0
FRONTIER_BONUS = 2.5
ENTROPY_BONUS = 0.05
STEP_COST = 0.03
STAY_PENALTY = 3.0
REVISIT_PENALTY = 1.25
WIN_THRESHOLD = 0.65
GAME_OVER_THRESHOLD = 0.75

class Node:
    def __init__(self, board, gui, action_path, score=0.0, reason="start", win_prob=0.0):
        self.board = board
        self.gui = gui
        self.action_path = action_path
        self.score = score
        self.reason = reason
        self.win_prob = win_prob

def state_hash(board, gui):
    # Tạo bản sao của GUI để che đi thanh đếm bước (nằm ở dòng 6, 7 trong GUI)
    # Các dòng này liên tục thay đổi, nếu không che đi BFS sẽ bị lặp
    gui_masked = gui.copy()
    gui_masked[6:8, :] = 0 
    return hash(board.tobytes() + gui_masked.tobytes())

@torch.no_grad()
def imagine_actions(model, board, gui, device):
    t_board = torch.tensor(board, dtype=torch.long, device=device).unsqueeze(0).repeat(len(ACTION_IDS), 1, 1)
    t_gui = torch.tensor(gui, dtype=torch.long, device=device).unsqueeze(0).repeat(len(ACTION_IDS), 1, 1)
    t_act = torch.tensor(ACTION_IDS, dtype=torch.long, device=device)
    next_b_logits, next_g_logits, reward, state_logits = model(t_board, t_gui, t_act)

    probs = F.softmax(next_b_logits, dim=1)
    entropy_map = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    entropy = entropy_map.flatten(1).max(dim=1).values.cpu().numpy()
    state_probs = F.softmax(state_logits, dim=-1).cpu().numpy()
    pred_b = next_b_logits.argmax(1).cpu().numpy()
    pred_g = next_g_logits.argmax(1).cpu().numpy()
    reward = reward.cpu().numpy()

    out = []
    for i, act_id in enumerate(ACTION_IDS):
        out.append({
            "act": act_id,
            "board": pred_b[i],
            "gui": pred_g[i],
            "reward": float(reward[i]),
            "entropy": float(entropy[i]),
            "win_prob": float(state_probs[i, 2]),
            "gameover_prob": float(state_probs[i, 3]),
        })
    return out

def transition_score(parent, tr, depth, real_visited, real_visit_counts=None):
    real_visit_counts = real_visit_counts or {}
    h = state_hash(tr["board"], tr["gui"])
    parent_h = state_hash(parent.board, parent.gui)
    frontier = h not in real_visited
    stayed = h == parent_h
    revisits = real_visit_counts.get(h, 0)
    score = (
        REWARD_SCALE * tr["reward"]
        + WIN_BONUS * tr["win_prob"]
        - GAME_OVER_PENALTY * tr["gameover_prob"]
        + (FRONTIER_BONUS if frontier else 0.0)
        + ENTROPY_BONUS * tr["entropy"]
        - STEP_COST * depth
        - (STAY_PENALTY if stayed else 0.0)
        - REVISIT_PENALTY * revisits
    )
    reason = "win" if tr["win_prob"] >= WIN_THRESHOLD else ("frontier" if frontier else "revisit")
    return score, reason

def _fallback_plan(rng):
    return [int(rng.integers(1, 5))], "random", 0.0

def bfs_plan(model, start_board, start_gui, depth, device, real_visited,
             real_visit_counts=None, beam=128, rng=None):
    """
    Beam-BFS over the learned world model.
    Returns (path, reason, score). The first action of path is executed in the real game.
    """
    rng = rng or np.random.default_rng()
    root = Node(start_board, start_gui, [])
    frontier = [root]
    visited_imagined = {state_hash(start_board, start_gui)}
    best = None

    for d in range(depth):
        candidates = []
        for node in frontier:
            for tr in imagine_actions(model, node.board, node.gui, device):
                h = state_hash(tr["board"], tr["gui"])
                step_score, reason = transition_score(node, tr, d + 1, real_visited, real_visit_counts)
                score = node.score + (GAMMA ** d) * step_score
                child = Node(
                    tr["board"],
                    tr["gui"],
                    node.action_path + [tr["act"]],
                    score=score,
                    reason=reason,
                    win_prob=tr["win_prob"],
                )

                if best is None or child.score > best.score:
                    best = child
                if tr["win_prob"] >= WIN_THRESHOLD:
                    return child.action_path, "win", child.score
                if tr["gameover_prob"] >= GAME_OVER_THRESHOLD:
                    continue
                if h in visited_imagined:
                    continue

                visited_imagined.add(h)
                candidates.append(child)

        if not candidates:
            break
        candidates.sort(key=lambda n: n.score, reverse=True)
        frontier = candidates[:beam]

    if best is not None and best.action_path:
        return best.action_path, best.reason, best.score
    return _fallback_plan(rng)

class MCTSNode:
    def __init__(self, board, gui, parent=None, action=None, immediate=0.0,
                 reason="root", depth=0, terminal=False, win_prob=0.0,
                 gameover_prob=0.0):
        self.board = board
        self.gui = gui
        self.parent = parent
        self.action = action
        self.immediate = immediate
        self.reason = reason
        self.depth = depth
        self.terminal = terminal
        self.win_prob = win_prob
        self.gameover_prob = gameover_prob
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.hash = state_hash(board, gui)

    @property
    def value(self):
        return self.value_sum / self.visits if self.visits else self.immediate

def _seen_in_path(node, h):
    while node is not None:
        if node.hash == h:
            return True
        node = node.parent
    return False

def _expand_mcts(node, model, depth_limit, device, real_visited, real_visit_counts):
    if node.terminal or node.depth >= depth_limit:
        return 0.0
    if node.children:
        return max(c.immediate for c in node.children.values())

    best_value = -1e9
    for tr in imagine_actions(model, node.board, node.gui, device):
        h = state_hash(tr["board"], tr["gui"])
        step_score, reason = transition_score(node, tr, node.depth + 1, real_visited, real_visit_counts)
        cycle = _seen_in_path(node, h)
        win = tr["win_prob"] >= WIN_THRESHOLD
        gameover = tr["gameover_prob"] >= GAME_OVER_THRESHOLD
        if cycle:
            step_score -= STAY_PENALTY
            reason = "cycle"
        elif win:
            reason = "win"
        elif gameover:
            reason = "gameover"

        terminal = (
            win
            or gameover
            or node.depth + 1 >= depth_limit
            or cycle
        )
        child = MCTSNode(
            tr["board"],
            tr["gui"],
            parent=node,
            action=tr["act"],
            immediate=step_score,
            reason=reason,
            depth=node.depth + 1,
            terminal=terminal,
            win_prob=tr["win_prob"],
            gameover_prob=tr["gameover_prob"],
        )
        node.children[tr["act"]] = child
        best_value = max(best_value, step_score)
    return best_value if node.children else 0.0

def _ucb(parent, child, cpuct):
    exploit = child.value
    explore = cpuct * math.sqrt(parent.visits + 1.0) / (1.0 + child.visits)
    return exploit + explore

def _best_mcts_path(root, depth):
    path = []
    node = root
    score = 0.0
    reason = "random"
    while node.children and len(path) < depth:
        node = max(
            node.children.values(),
            key=lambda c: (
                c.reason == "win",
                c.reason != "gameover",
                c.value,
                c.immediate,
                c.visits,
            ),
        )
        path.append(node.action)
        score = node.value
        reason = node.reason
        if node.terminal:
            break
    return path, reason, score

def mcts_plan(model, start_board, start_gui, depth, device, real_visited,
              real_visit_counts=None, sims=128, cpuct=1.4, rng=None):
    rng = rng or np.random.default_rng()
    real_visit_counts = real_visit_counts or {}
    root = MCTSNode(start_board, start_gui)

    for _ in range(max(1, sims)):
        node = root
        path = [root]
        while node.children and not node.terminal and node.depth < depth:
            node = max(node.children.values(), key=lambda c: _ucb(node, c, cpuct))
            path.append(node)

        future = _expand_mcts(node, model, depth, device, real_visited, real_visit_counts)

        for n in reversed(path[1:]):
            ret = n.immediate + GAMMA * future
            n.visits += 1
            n.value_sum += ret
            future = ret
        root.visits += 1
        root.value_sum += future

    path, reason, score = _best_mcts_path(root, depth)
    if path:
        return path, reason, score
    return _fallback_plan(rng)

def both_plan(model, start_board, start_gui, depth, device, real_visited,
              real_visit_counts=None, beam=128, sims=128, cpuct=1.4, rng=None):
    """Run both planners and choose the stronger current plan."""
    rng = rng or np.random.default_rng()
    bfs_path, bfs_reason, bfs_score = bfs_plan(
        model, start_board, start_gui, depth, device, real_visited,
        real_visit_counts=real_visit_counts, beam=beam, rng=rng,
    )
    mcts_path, mcts_reason, mcts_score = mcts_plan(
        model, start_board, start_gui, depth, device, real_visited,
        real_visit_counts=real_visit_counts, sims=sims, cpuct=cpuct, rng=rng,
    )

    if mcts_reason == "win" and bfs_reason != "win":
        chosen = ("MCTS", mcts_path, mcts_reason, mcts_score)
    elif bfs_reason == "win" and mcts_reason != "win":
        chosen = ("BFS", bfs_path, bfs_reason, bfs_score)
    elif mcts_score > bfs_score:
        chosen = ("MCTS", mcts_path, mcts_reason, mcts_score)
    else:
        chosen = ("BFS", bfs_path, bfs_reason, bfs_score)

    compare = (
        f"BFS={ACTION_NAMES.get(bfs_path[0]) if bfs_path else '?'}"
        f"/{bfs_reason}/{bfs_score:.2f}  "
        f"MCTS={ACTION_NAMES.get(mcts_path[0]) if mcts_path else '?'}"
        f"/{mcts_reason}/{mcts_score:.2f}"
    )
    return (*chosen, compare)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ls20_model.pt", help="Pretrained model weights")
    ap.add_argument("--depth", type=int, default=5, help="Độ sâu của BFS")
    ap.add_argument("--games", type=int, default=10, help="Số ván chơi")
    ap.add_argument("--max-steps", type=int, default=100, help="Số bước chơi tối đa mỗi ván")
    ap.add_argument("--out", default="recordings/agent_replay.npz", help="File lưu lại buffer")
    ap.add_argument("--train-freq", type=int, default=5, help="Cập nhật mô hình sau mỗi N bước")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--planner", choices=["bfs", "mcts", "both"], default="bfs",
                    help="Planner dùng world model: bfs, mcts, hoặc both để so sánh")
    ap.add_argument("--beam", type=int, default=128, help="Beam width cho BFS")
    ap.add_argument("--mcts-sims", type=int, default=128, help="Số simulation cho MCTS mỗi bước")
    ap.add_argument("--mcts-cpuct", type=float, default=1.4, help="Exploration constant cho MCTS")
    ap.add_argument("--commit-steps", type=int, default=3,
                    help="Số bước tiếp tục đi theo plan trước khi replan; win path sẽ commit hết")
    ap.add_argument("--online-train", action="store_true",
                    help="Bật fine-tuning world model trong lúc chơi; mặc định tắt để chạy rẻ và ổn định")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if not os.path.exists(args.model):
        sys.exit(f"Không tìm thấy model {args.model}")
        
    saved = torch.load(args.model, map_location=device, weights_only=False)
    num_tokens = saved["num_tokens"]
    
    model = Ls20WorldModel(num_tokens=num_tokens).to(device)
    model.load_state_dict(saved["model"])
    
    # Load Environment
    from arc_agi import Arcade, OperationMode
    from arcengine import GameState
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    env = arc.make("ls20")
    if env is None: sys.exit("Không tìm thấy ls20")
    
    wrapper = Ls20Wrapper(env)
    wrapper.tokenizer.id_to_block = saved["id_to_block"]
    wrapper.tokenizer.block_to_id = saved["block_to_id"]
    wrapper.tokenizer.next_id = num_tokens
    
    _STATE_ORDER = ["NOT_PLAYED", "NOT_FINISHED", "WIN", "GAME_OVER"]
    def _state_idx(obs):
        st = getattr(obs, "state", None)
        name = getattr(st, "name", None) or str(st)
        try:
            return _STATE_ORDER.index(name)
        except ValueError:
            return 1
            
    buf = []
    
    mode = "có online fine-tuning" if args.online_train else "frozen world model, không train online"
    print(f"Khởi động planner ({mode})...")
    
    rng = np.random.default_rng(42)
    t0 = time.time()
    
    total_steps = 0
    
    for game_id in range(args.games):
        print(f"\n=== Bắt đầu Game {game_id + 1}/{args.games} ===")
        obs = env.reset()
        board, gui = wrapper._process_frame(obs.frame)
        board = np.clip(board, 0, num_tokens - 1)
        lv = getattr(obs, "levels_completed", 0) or 0
        
        h0 = state_hash(board, gui)
        real_visited = {h0}
        real_visit_counts = {h0: 1}
        queued_plan = []
        queued_label = None
        
        for step in range(args.max_steps):
            model.eval()
            if queued_plan:
                act_id = queued_plan.pop(0)
                print(
                    f"G{game_id + 1}-S{step + 1}: tiếp tục plan {queued_label} -> "
                    f"{ACTION_NAMES.get(act_id)}"
                )
            else:
                compare = None
                if args.planner == "bfs":
                    path, reason, score = bfs_plan(
                        model, board, gui, args.depth, device, real_visited,
                        real_visit_counts=real_visit_counts, beam=args.beam, rng=rng,
                    )
                    planner_name = "BFS"
                elif args.planner == "mcts":
                    path, reason, score = mcts_plan(
                        model, board, gui, args.depth, device, real_visited,
                        real_visit_counts=real_visit_counts,
                        sims=args.mcts_sims, cpuct=args.mcts_cpuct, rng=rng,
                    )
                    planner_name = "MCTS"
                else:
                    planner_name, path, reason, score, compare = both_plan(
                        model, board, gui, args.depth, device, real_visited,
                        real_visit_counts=real_visit_counts,
                        beam=args.beam, sims=args.mcts_sims, cpuct=args.mcts_cpuct, rng=rng,
                    )
                    print(f"  compare: {compare}")

                act_id = path[0] if path else int(rng.integers(1, 5))
                commit_n = len(path) if reason == "win" else min(args.commit_steps, len(path))
                queued_plan = list(path[1:commit_n]) if commit_n > 1 else []
                queued_label = f"{planner_name}/{reason}"
                plan_txt = " ".join(ACTION_NAMES.get(a, str(a)) for a in path[:args.depth])
                print(
                    f"G{game_id + 1}-S{step + 1}: {planner_name} chọn "
                    f"{ACTION_NAMES.get(act_id)} reason={reason} score={score:.2f} "
                    f"commit={commit_n} path=[{plan_txt}]"
                )
                
            b2, g2, reward, done, info = wrapper.step_dir(act_id)
            total_steps += 1
            
            if b2 is None:
                obs = env.reset()
                board, gui = wrapper._process_frame(obs.frame)
                board = np.clip(board, 0, num_tokens - 1)
                lv = getattr(obs, "levels_completed", 0) or 0
                real_visited.clear()
                h0 = state_hash(board, gui)
                real_visited.add(h0)
                real_visit_counts.clear()
                real_visit_counts[h0] = 1
                queued_plan.clear()
                queued_label = None
                continue
                
            b2 = np.clip(b2, 0, num_tokens - 1)
            h2 = state_hash(b2, g2)
            real_visited.add(h2)
            real_visit_counts[h2] = real_visit_counts.get(h2, 0) + 1
            
            raw_obs = wrapper.env.observation_space
            nxt_lv = getattr(raw_obs, "levels_completed", 0) or 0
            won = 1 if getattr(raw_obs, "state", None) == GameState.WIN else 0
            st2 = _state_idx(raw_obs)
            
            is_transition = bool(nxt_lv > lv or won or getattr(raw_obs, "state", None) == GameState.GAME_OVER)
            if nxt_lv > lv:
                queued_plan.clear()
                queued_label = None
            
            r = 5.0 * max(0, nxt_lv - lv) + (10.0 if won else 0.0)
            if not np.array_equal(board, b2):
                r += 0.1
                
            buf.append({
                "game_id": game_id + 1,
                "step": step + 1,
                "b": board.copy(), "g": gui.copy(),
                "act_id": act_id, "r": r,
                "b2": b2.copy(), "g2": g2.copy(), "st2": st2,
                "is_trans": is_transition
            })
            
            board, gui = b2, g2
            lv = nxt_lv
            
            if args.online_train and len(buf) >= args.batch and total_steps % args.train_freq == 0:
                model.train()
                train_ls20(model, buf, epochs=1, batch=args.batch, lr=5e-5, device=device, log_every=1000)
                
            if won or getattr(raw_obs, "state", None) == GameState.GAME_OVER:
                print(f"Game {game_id + 1} kết thúc (WIN={won}) tại bước {step + 1}.")
                queued_plan.clear()
                break
            
    print(f"\nHoàn thành {args.games} ván chơi trong {time.time()-t0:.1f}s.")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if buf:
        np.savez_compressed(
            args.out,
            game_id=np.array([e["game_id"] for e in buf], dtype=np.int16),
            step=np.array([e["step"] for e in buf], dtype=np.int16),
            b=np.stack([e["b"] for e in buf]),
            g=np.stack([e["g"] for e in buf]),
            act_id=np.array([e["act_id"] for e in buf], dtype=np.int16),
            r=np.array([e["r"] for e in buf], dtype=np.float32),
            b2=np.stack([e["b2"] for e in buf]),
            g2=np.stack([e["g2"] for e in buf]),
            st2=np.array([e["st2"] for e in buf], dtype=np.int16),
            is_trans=np.array([e["is_trans"] for e in buf], dtype=np.bool_)
        )
        print(f"Đã lưu lịch sử chơi vào {args.out}")
    else:
        print("Không có transition hợp lệ để lưu replay buffer.")

if __name__ == "__main__":
    main()
