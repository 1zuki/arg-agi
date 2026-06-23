#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw

from ls20_wrapper import Ls20Wrapper
from ls20_model import Ls20WorldModel

# ARC palette
PALETTE = [
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
    (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
    (127, 219, 255), (135, 12, 37), (1, 255, 196), (177, 13, 201),
    (133, 100, 4), (255, 255, 255), (96, 96, 96), (44, 60, 117),
]

def _ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def _palette_img(grid, scale):
    h, w = grid.shape
    lut = np.array(PALETTE, dtype=np.uint8)
    rgb = lut[grid]
    img = Image.fromarray(rgb, mode="RGB")
    return img.resize((w * scale, h * scale), Image.NEAREST)

def unwrap_board(board_tokens, id_to_block):
    """Chuyển ma trận token (11, 12) về lại mảng pixel (55, 60) bằng cách ghép các ô 5x5"""
    H, W = board_tokens.shape
    pixels = np.zeros((H * 5, W * 5), dtype=np.int8)
    for y in range(H):
        for x in range(W):
            tid = int(board_tokens[y, x])
            block = id_to_block.get(tid, np.zeros((5, 5), dtype=np.int8))
            pixels[y*5 : y*5+5, x*5 : x*5+5] = block
    return pixels

def combine_frame(board_pixels, gui_pixels):
    """Ghép board 55x60 và GUI 9x64 thành frame 64x64.
    Lưu ý: Board chỉ rộng 60 và bắt đầu từ cột 4."""
    frame = np.zeros((64, 64), dtype=np.int8)
    # Lắp board vào (y: 0->55, x: 4->64)
    frame[0:55, 4:64] = board_pixels
    # Lắp gui vào (y: 55->64, x: 0->64)
    frame[55:64, 0:64] = gui_pixels
    return frame

def render_comparison(true_frame, pred_frame, title_text, scale=6):
    """Vẽ 2 frame cạnh nhau (Ground Truth vs Predicted)"""
    img_true = _palette_img(true_frame, scale)
    img_pred = _palette_img(pred_frame, scale)
    
    gw, gh = img_true.size
    gap = 20
    panel_h = 40
    
    canvas = Image.new("RGB", (gw * 2 + gap, gh + panel_h), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    
    # Text
    draw.text((10, 5), title_text, fill=(255, 255, 255))
    draw.text((10, 25), "True Next State", fill=(46, 204, 64))
    draw.text((gw + gap + 10, 25), "Predicted Next State", fill=(255, 65, 54))
    
    # Dán ảnh
    canvas.paste(img_true, (0, panel_h))
    canvas.paste(img_pred, (gw + gap, panel_h))
    
    return canvas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ls20_model.pt", help="Đường dẫn đến file model đã train")
    ap.add_argument("--steps", type=int, default=50, help="Số bước muốn visualize")
    ap.add_argument("--out", default="replays/ls20_comparison.gif", help="Tên file GIF đầu ra")
    ap.add_argument("--scale", type=int, default=6, help="Scale pixel (to lên cho dễ nhìn)")
    ap.add_argument("--npz", default=None, help="Đường dẫn đến file agent_replay.npz để xem lại")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(args.model):
        sys.exit(f"Không tìm thấy {args.model}. Hãy chờ pretrain xong.")
        
    print(f"Load model từ {args.model}...")
    saved = torch.load(args.model, map_location=device, weights_only=False)
    num_tokens = saved["num_tokens"]
    id_to_block = saved["id_to_block"]
    
    model = Ls20WorldModel(num_tokens=num_tokens).to(device)
    model.load_state_dict(saved["model"])
    model.eval()

    if args.npz:
        print(f"Loading replay buffer từ {args.npz}...")
        if not os.path.exists(args.npz):
            sys.exit(f"Không tìm thấy {args.npz}")
            
        data = np.load(args.npz)
        buf_b = data["b"]
        buf_g = data["g"]
        buf_act = data["act_id"]
        buf_b2 = data["b2"]
        buf_g2 = data["g2"]
        
        frames = []
        n_steps = min(args.steps, len(buf_b))
        print(f"Đang vẽ {n_steps} bước từ buffer...")
        
        for step in range(n_steps):
            board = buf_b[step]
            gui = buf_g[step]
            act_id = int(buf_act[step])
            true_b = buf_b2[step]
            true_g = buf_g2[step]
            
            t_board = torch.tensor(board, dtype=torch.long, device=device).unsqueeze(0)
            t_gui = torch.tensor(gui, dtype=torch.long, device=device).unsqueeze(0)
            t_act = torch.tensor([act_id], dtype=torch.long, device=device)
            
            with torch.no_grad():
                next_b_logits, next_g_logits, _, _ = model(t_board, t_gui, t_act)
                pred_b = next_b_logits.argmax(1).squeeze(0).cpu().numpy()
                pred_g = next_g_logits.argmax(1).squeeze(0).cpu().numpy()
                
            true_b_pixels = unwrap_board(true_b, id_to_block)
            pred_b_pixels = unwrap_board(pred_b, id_to_block)
            
            true_frame = combine_frame(true_b_pixels, true_g)
            pred_frame = combine_frame(pred_b_pixels, pred_g)
            
            ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
            if "game_id" in data and "step" in data:
                gid = int(data["game_id"][step])
                sid = int(data["step"][step])
                title_text = f"Game {gid} | Step {sid} | Phím: {ACTION_NAMES.get(act_id, str(act_id))}"
            else:
                title_text = f"Step: {step} | Phím: {ACTION_NAMES.get(act_id, str(act_id))}"
            img = render_comparison(true_frame, pred_frame, title_text, scale=args.scale)
            frames.append(img)
            
        if frames:
            _ensure_parent(args.out)
            frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=300, loop=0)
            print(f"Đã lưu GIF so sánh tại {args.out}")
        return

    try:
        from arc_agi import Arcade, OperationMode
    except:
        sys.exit("Cần cài đặt arc_agi.")
        
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    env = arc.make("ls20")
    if env is None: sys.exit("Không tìm thấy ls20")
    
    wrapper = Ls20Wrapper(env)
    # Ghi đè từ điển để đảm bảo nhất quán với model
    wrapper.tokenizer.id_to_block = id_to_block
    wrapper.tokenizer.block_to_id = saved["block_to_id"]
    wrapper.tokenizer.next_id = num_tokens
    
    obs = env.reset()
    board, gui = wrapper._process_frame(obs.frame)
    rng = np.random.default_rng(42)
    
    frames = []
    print(f"Bắt đầu chạy random {args.steps} bước...")
    
    for step in range(args.steps):
        act_id = int(rng.integers(1, 5))
        
        # 1. Dự đoán bằng Model
        t_board = torch.tensor(board, dtype=torch.long, device=device).unsqueeze(0)
        t_gui = torch.tensor(gui, dtype=torch.long, device=device).unsqueeze(0)
        t_act = torch.tensor([act_id], dtype=torch.long, device=device)
        
        with torch.no_grad():
            next_b_logits, next_g_logits, _, _ = model(t_board, t_gui, t_act)
            pred_b = next_b_logits.argmax(1).squeeze(0).cpu().numpy()
            pred_g = next_g_logits.argmax(1).squeeze(0).cpu().numpy()
            
        # 2. Bước đi thực tế trong Environment
        true_b, true_g, _, _, _ = wrapper.step_dir(act_id)
        if true_b is None:
            # Game reset do win/lose
            obs = env.reset()
            board, gui = wrapper._process_frame(obs.frame)
            continue
            
        # 3. Unwrap Token -> Pixel Array
        true_b_pixels = unwrap_board(true_b, id_to_block)
        pred_b_pixels = unwrap_board(pred_b, id_to_block)
        
        # 4. Gắn thêm GUI
        true_frame = combine_frame(true_b_pixels, true_g)
        pred_frame = combine_frame(pred_b_pixels, pred_g)
        
        # 5. Vẽ ảnh
        ACTION_NAMES = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
        title_text = f"Step: {step} | Phím: {ACTION_NAMES.get(act_id, str(act_id))}"
        img = render_comparison(true_frame, pred_frame, title_text, scale=args.scale)
        frames.append(img)
        
        board, gui = true_b, true_g
        
    if frames:
        _ensure_parent(args.out)
        frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=300, loop=0)
        print(f"Đã lưu GIF so sánh tại {args.out}")
    else:
        print("Không tạo được frame nào.")

if __name__ == "__main__":
    main()
