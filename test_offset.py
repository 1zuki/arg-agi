import numpy as np

with open('ls20_frame.txt') as f:
    lines = [list(line.strip()) for line in f if line.strip()]

arr = np.array(lines)
H, W = arr.shape

print(f"Frame shape: {H}x{W}")

# Tìm UI margin: hàng chứa 'b' hoặc 'c' (chữ) hoặc phần cố định
# Game board sẽ được cấu tạo bởi các ô 5x5
# Ta thử các offset_x từ 0 đến 10, offset_y từ 0 đến 10
best_offset = None
best_blocks = 0
best_w = 0
best_h = 0

for oy in range(15):
    for ox in range(15):
        valid_blocks = 0
        # Thử grid size WxH
        # Giả sử game board dài ít nhất 5 blocks (25px)
        max_by = (H - oy) // 5
        max_bx = (W - ox) // 5
        
        for by in range(max_by):
            for bx in range(max_bx):
                patch = arr[oy + by*5 : oy + by*5 + 5, ox + bx*5 : ox + bx*5 + 5]
                # Kiểm tra patch này có phải là 1 block hợp lệ không?
                # Giả sử block hợp lệ là block có 1 màu nền chiếm đa số (solid) hoặc viền
                # Hoặc chỉ đơn giản đếm số block khác background
                # Background là '4' (có vẻ vậy)
                pass
                
# Ta in thẳng ra các tọa độ mà có sự thay đổi pixel để nhìn bằng mắt
for y in range(H):
    row_str = "".join(arr[y])
    print(f"{y:2d}: {row_str}")
