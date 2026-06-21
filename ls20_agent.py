#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ls20_wrapper import Ls20Wrapper
from ls20_model import Ls20WorldModel
from pretrain_ls20 import train_ls20

class Node:
    def __init__(self, board, gui, action_path, entropy):
        self.board = board
        self.gui = gui
        self.action_path = action_path
        self.entropy = entropy

def state_hash(board, gui):
    # Tạo bản sao của GUI để che đi thanh đếm bước (nằm ở dòng 6, 7 trong GUI)
    # Các dòng này liên tục thay đổi, nếu không che đi BFS sẽ bị lặp
    gui_masked = gui.copy()
    gui_masked[6:8, :] = 0 
    return hash(board.tobytes() + gui_masked.tobytes())

def bfs_plan(model, start_board, start_gui, depth, device, real_visited):
    """
    Tìm kiếm BFS.
    Trả về: win_path (nếu tìm thấy), hoặc frontier_path (chưa từng đi ngoài đời thực)
    """
    queue = [Node(start_board, start_gui, [], 0.0)]
    visited_imagined = {state_hash(start_board, start_gui)}
    
    best_win_path = None
    best_frontier_path = None
    max_frontier_score = -1.0
    
    max_uncertainty_node = None
    max_uncertainty = -1.0
    
    # Pre-allocate actions tensor
    t_act = torch.tensor([1, 2, 3, 4], dtype=torch.long, device=device)
    
    while queue:
        node = queue.pop(0)
        
        if len(node.action_path) >= depth:
            if node.entropy > max_uncertainty:
                max_uncertainty = node.entropy
                max_uncertainty_node = node
            continue
            
        t_board = torch.tensor(node.board, dtype=torch.long, device=device).unsqueeze(0).repeat(4, 1, 1)
        t_gui = torch.tensor(node.gui, dtype=torch.long, device=device).unsqueeze(0).repeat(4, 1, 1)
        
        with torch.no_grad():
            next_b_logits, next_g_logits, _, state_logits = model(t_board, t_gui, t_act)
            
            probs = F.softmax(next_b_logits, dim=1)
            # Dùng MAX thay vì MEAN để bắt được sự tò mò tại một vị trí cụ thể (khối mới)
            entropy_map = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            entropy = entropy_map.flatten(1).max(dim=1)[0]
            
            pred_b = next_b_logits.argmax(1).cpu().numpy()
            pred_g = next_g_logits.argmax(1).cpu().numpy()
            
            state_probs = F.softmax(state_logits, dim=-1)
            
        for i in range(4):
            act_id = i + 1
            nb = pred_b[i]
            ng = pred_g[i]
            h = state_hash(nb, ng)
            
            if h in visited_imagined:
                continue
                
            visited_imagined.add(h)
            ent = node.entropy + entropy[i].item()
            sp = state_probs[i]
            
            if sp[2].item() > 0.5:
                # WIN PATH
                best_win_path = node.action_path + [act_id]
                return best_win_path, None
                
            # Frontier: Trạng thái chưa từng được thấy ở thế giới thực
            if h not in real_visited:
                current_depth = len(node.action_path) + 1
                # Curiosity Score: Entropy tích luỹ chia cho số bước đi (ưu tiên đường ngắn mà độ bối rối cao nhất)
                score = ent / current_depth
                if score > max_frontier_score:
                    max_frontier_score = score
                    best_frontier_path = node.action_path + [act_id]
                    
            new_node = Node(nb, ng, node.action_path + [act_id], ent)
            queue.append(new_node)
            
            if ent > max_uncertainty:
                max_uncertainty = ent
                max_uncertainty_node = new_node
                
    if best_frontier_path:
        return None, best_frontier_path
        
    if max_uncertainty_node is not None and len(max_uncertainty_node.action_path) > 0:
        return None, max_uncertainty_node.action_path
        
    return None, [int(np.random.randint(1, 5))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ls20_model.pt", help="Pretrained model weights")
    ap.add_argument("--depth", type=int, default=5, help="Độ sâu của BFS")
    ap.add_argument("--games", type=int, default=10, help="Số ván chơi")
    ap.add_argument("--max-steps", type=int, default=100, help="Số bước chơi tối đa mỗi ván")
    ap.add_argument("--out", default="agent_replay.npz", help="File lưu lại buffer")
    ap.add_argument("--train-freq", type=int, default=5, help="Cập nhật mô hình sau mỗi N bước")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if not os.path.exists(args.model):
        sys.exit(f"Không tìm thấy model {args.model}")
        
    saved = torch.load(args.model, map_location=device)
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
    
    ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
    print("Khởi động Online Active Inference...")
    
    rng = np.random.default_rng(42)
    t0 = time.time()
    
    total_steps = 0
    
    for game_id in range(args.games):
        print(f"\n=== Bắt đầu Game {game_id + 1}/{args.games} ===")
        obs = env.reset()
        board, gui = wrapper._process_frame(obs.frame)
        board = np.clip(board, 0, num_tokens - 1)
        lv = getattr(obs, "levels_completed", 0) or 0
        
        real_visited = {state_hash(board, gui)}
        
        for step in range(args.max_steps):
            model.eval()
            win_path, exp_path = bfs_plan(model, board, gui, args.depth, device, real_visited)
            
            if win_path:
                act_id = win_path[0]
                print(f"G{game_id + 1}-S{step + 1}: Phát hiện WIN PATH! Chọn: {ACTION_NAMES.get(act_id)}")
            else:
                act_id = exp_path[0]
                print(f"G{game_id + 1}-S{step + 1}: Khám phá Frontier/Uncertainty. Chọn: {ACTION_NAMES.get(act_id)}")
                
            b2, g2, reward, done, info = wrapper.step_dir(act_id)
            total_steps += 1
            
            if b2 is None:
                obs = env.reset()
                board, gui = wrapper._process_frame(obs.frame)
                board = np.clip(board, 0, num_tokens - 1)
                lv = getattr(obs, "levels_completed", 0) or 0
                real_visited.clear()
                real_visited.add(state_hash(board, gui))
                continue
                
            b2 = np.clip(b2, 0, num_tokens - 1)
            real_visited.add(state_hash(b2, g2))
            
            raw_obs = wrapper.env.observation_space
            nxt_lv = getattr(raw_obs, "levels_completed", 0) or 0
            won = 1 if getattr(raw_obs, "state", None) == GameState.WIN else 0
            st2 = _state_idx(raw_obs)
            
            is_transition = bool(nxt_lv > lv or won or getattr(raw_obs, "state", None) == GameState.GAME_OVER)
            
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
            
            if len(buf) >= args.batch and total_steps % args.train_freq == 0:
                model.train()
                train_ls20(model, buf, epochs=1, batch=args.batch, lr=5e-5, device=device, log_every=1000)
                
            if won or getattr(raw_obs, "state", None) == GameState.GAME_OVER:
                print(f"Game {game_id + 1} kết thúc (WIN={won}) tại bước {step + 1}.")
                break
            
    print(f"\nHoàn thành {args.games} ván chơi trong {time.time()-t0:.1f}s.")
    
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

if __name__ == "__main__":
    main()
