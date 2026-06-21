#!/usr/bin/env python3
import argparse
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ls20_wrapper import Ls20Wrapper
from ls20_model import Ls20WorldModel, N_COLORS, N_STATES

def collect_ls20(steps_per_game=5000, seed=0, env_dir="environment_files"):
    try:
        from arc_agi import Arcade, OperationMode
        from arcengine import GameState
    except Exception as e:
        sys.exit("Hãy cài arc_agi từ arc_agi_3_wheels trước khi chạy.")

    rng = np.random.default_rng(seed)
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir)
    
    env = arc.make("ls20")
    if env is None:
        sys.exit("Không tìm thấy game ls20 trong environment_files.")
        
    wrapper = Ls20Wrapper(env)
    
    def progress(obs):
        lv = getattr(obs, "levels_completed", 0) or 0
        won = 1 if getattr(obs, "state", None) == GameState.WIN else 0
        return lv, won

    _STATE_ORDER = ["NOT_PLAYED", "NOT_FINISHED", "WIN", "GAME_OVER"]
    def _state_idx(obs):
        st = getattr(obs, "state", None)
        name = getattr(st, "name", None) or str(st)
        try:
            return _STATE_ORDER.index(name)
        except ValueError:
            return 1
            
    buf = []
    obs = env.reset()
    board, gui = wrapper._process_frame(obs.frame)
    prev_lv, _ = progress(obs)
    
    got = 0
    t0 = time.time()
    
    for _ in range(steps_per_game):
        # Random phím di chuyển: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
        act_id = int(rng.integers(1, 5))
        
        b2, g2, reward, done, info = wrapper.step_dir(act_id)
        
        if b2 is None:
            # Game was reset or invalid step
            obs = env.reset()
            board, gui = wrapper._process_frame(obs.frame)
            prev_lv, _ = progress(obs)
            continue
            
        raw_obs = wrapper.env.observation_space
        lv, won = progress(raw_obs)
        st2 = _state_idx(raw_obs)
        
        is_transition = bool(lv > prev_lv or won or getattr(raw_obs, "state", None) == GameState.GAME_OVER)
        
        r = 5.0 * max(0, lv - prev_lv) + (10.0 if won else 0.0)
        if not np.array_equal(board, b2):
            r += 0.1 # thưởng nhỏ nếu có sự thay đổi trên bàn
            
        prev_lv = lv
        
        # Chỉ lưu khi có dữ liệu board hợp lệ
        buf.append({
            "b": board.copy(), "g": gui.copy(),
            "act_id": act_id, "r": r,
            "b2": b2.copy(), "g2": g2.copy(), "st2": st2,
            "is_trans": is_transition
        })
        got += 1
        
        board, gui = b2, g2
        
        if won or getattr(raw_obs, "state", None) == GameState.GAME_OVER:
            obs = env.reset()
            board, gui = wrapper._process_frame(obs.frame)
            prev_lv, _ = progress(obs)
            
    print(f"Thu thập được {got} transitions. Có {wrapper.num_tokens} tokens riêng biệt được tìm thấy.")
    return buf, wrapper

def train_ls20(model, buffer, epochs=6, batch=64, lr=3e-4, device="cpu", change_weight=20.0, log_every=50):
    dev = torch.device(device)
    model.to(dev)
    opt = optim.Adam(model.parameters(), lr=lr)
    
    n = len(buffer)
    if n < batch:
        raise ValueError(f"Ít dữ liệu quá: {n} < {batch}")
        
    steps = max(1, n // batch)
    last = {}
    
    for ep in range(epochs):
        order = np.random.permutation(n)
        for bi in range(steps):
            sel = order[bi * batch:(bi + 1) * batch]
            if len(sel) < batch: break
            
            b = [buffer[i] for i in sel]
            cur_b = torch.tensor(np.stack([e["b"] for e in b]), dtype=torch.long, device=dev)
            cur_g = torch.tensor(np.stack([e["g"] for e in b]), dtype=torch.long, device=dev)
            nxt_b = torch.tensor(np.stack([e["b2"] for e in b]), dtype=torch.long, device=dev)
            nxt_g = torch.tensor(np.stack([e["g2"] for e in b]), dtype=torch.long, device=dev)
            
            act_id = torch.tensor([e["act_id"] for e in b], dtype=torch.long, device=dev)
            rew = torch.tensor([e["r"] for e in b], dtype=torch.float32, device=dev)
            st2 = torch.tensor([e["st2"] for e in b], dtype=torch.long, device=dev)
            is_trans = torch.tensor([e.get("is_trans", False) for e in b], dtype=torch.bool, device=dev)
            
            # Forward
            next_b_logits, next_g_logits, reward, state_logits = model(cur_b, cur_g, act_id)
            
            # Board Loss (Change-weighted)
            ce_b = F.cross_entropy(next_b_logits, nxt_b, reduction="none")
            changed_b = (cur_b != nxt_b).float()
            wb = 1.0 + change_weight * changed_b
            wb[is_trans] = 0.0 # Bỏ qua frame chuyển màn
            l_board = (ce_b * wb).sum() / (wb.sum() + 1e-8)
            
            # GUI Loss (Change-weighted)
            ce_g = F.cross_entropy(next_g_logits, nxt_g, reduction="none")
            changed_g = (cur_g != nxt_g).float()
            wg = 1.0 + change_weight * changed_g
            wg[is_trans] = 0.0 # Bỏ qua frame chuyển màn
            l_gui = (ce_g * wg).sum() / (wg.sum() + 1e-8)
            
            l_rew = F.mse_loss(reward, rew)
            l_state = F.cross_entropy(state_logits, st2)
            
            loss = l_board + l_gui + l_rew + l_state
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            
            with torch.no_grad():
                pred_b = next_b_logits.argmax(1)
                cm_b = changed_b.bool()
                acc_b = (pred_b[cm_b] == nxt_b[cm_b]).float().mean().item() if cm_b.any() else 0.0
                sacc = (state_logits.argmax(-1) == st2).float().mean().item()
                
            last = {
                "loss": loss.item(), "board": l_board.item(), "gui": l_gui.item(),
                "acc_b": acc_b, "sacc": sacc
            }
            if bi % log_every == 0:
                print(f"  ep{ep} step{bi}/{steps} loss={last['loss']:.3f} board={last['board']:.3f} gui={last['gui']:.3f} acc_ch_board={acc_b:.3f} sacc={sacc:.3f}")
                
    return last

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--out", default="ls20_model.pt")
    args = ap.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Thu thập dữ liệu game...")
    buf, wrapper = collect_ls20(steps_per_game=args.steps)
    
    num_tokens = wrapper.num_tokens
    print(f"Khởi tạo mô hình Ls20WorldModel với {num_tokens} tokens...")
    model = Ls20WorldModel(num_tokens=num_tokens)
    
    print("Bắt đầu huấn luyện...")
    train_ls20(model, buf, epochs=args.epochs, device=device)
    
    # Save the token dictionary along with the model weights
    save_data = {
        "model": model.state_dict(),
        "id_to_block": wrapper.tokenizer.id_to_block,
        "block_to_id": wrapper.tokenizer.block_to_id,
        "num_tokens": num_tokens
    }
    torch.save(save_data, args.out)
    print(f"Đã lưu mô hình và tokenizer vào {args.out}")

if __name__ == "__main__":
    main()
