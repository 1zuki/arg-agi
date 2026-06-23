#!/usr/bin/env python3
import sys
import os
import numpy as np
from pathlib import Path

# Thêm đường dẫn để có thể import package từ offline environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from arc_agi import Arcade, OperationMode
except ImportError:
    # Nếu chưa cài arc-agi, script có thể chạy từ trong docker, ta in ra lỗi
    print("Vui lòng chạy script này trong môi trường có cài arc_agi (như docker container vừa build).")
    sys.exit(1)

def analyze_ls20():
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir="environment_files")
    env = arc.make("ls20")
    if env is None:
        print("Không tìm thấy game ls20")
        return
    
    obs = env.reset()
    if obs is None:
        obs = env.observation_space
        
    arr = np.array(obs.frame)
    while arr.ndim > 2:
        arr = arr[-1]
    arr = np.clip(arr.astype(int), 0, 15)
    
    print("Kích thước lưới ban đầu:", arr.shape)
    
    out = Path("debug") / "ls20_frame.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    # In lưới ra file text để model có thể đọc
    with out.open("w") as f:
        for r in range(arr.shape[0]):
            f.write("".join([f"{c:x}" for c in arr[r, :]]) + "\n")
            
    print(f"Đã lưu lưới vào {out}")

if __name__ == "__main__":
    analyze_ls20()
