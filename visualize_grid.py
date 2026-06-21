from PIL import Image, ImageDraw
import numpy as np
import os

PALETTE = [
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64),
    (255, 220, 0), (170, 170, 170), (240, 18, 190), (255, 133, 27),
    (127, 219, 255), (135, 12, 37), (1, 255, 196), (177, 13, 201),
    (133, 100, 4), (255, 255, 255), (96, 96, 96), (44, 60, 117)
]

def hex_to_int(c):
    try:
        return int(c, 16)
    except:
        return 0

with open('ls20_frame.txt', 'r') as f:
    lines = [list(line.strip()) for line in f if line.strip()]

H, W = len(lines), len(lines[0])
grid = np.zeros((H, W), dtype=int)
for y in range(H):
    for x in range(W):
        grid[y, x] = hex_to_int(lines[y][x])

# Phóng to ảnh để dễ nhìn (scale 10x)
SCALE = 10
img_w, img_h = W * SCALE, H * SCALE
img = Image.new("RGB", (img_w, img_h))
pixels = img.load()

for y in range(H):
    for x in range(W):
        color = PALETTE[grid[y, x] % 16]
        for dy in range(SCALE):
            for dx in range(SCALE):
                pixels[x*SCALE + dx, y*SCALE + dy] = color

draw = ImageDraw.Draw(img)

# Tọa độ dự đoán của lưới game
start_x = 4
end_x = 64   # 60 pixels = 12 blocks
start_y = 0  
end_y = 55   # 55 pixels = 11 blocks

# Vẽ đường viền tổng (Màu trắng dày)
draw.rectangle([start_x*SCALE, start_y*SCALE, end_x*SCALE, end_y*SCALE], outline="white", width=3)

# Vẽ các đường chia 5x5 (Màu đỏ mờ)
for x in range(start_x, end_x + 1, 5):
    draw.line([(x*SCALE, start_y*SCALE), (x*SCALE, end_y*SCALE)], fill="red", width=1)
for y in range(start_y, end_y + 1, 5):
    draw.line([(start_x*SCALE, y*SCALE), (end_x*SCALE, y*SCALE)], fill="red", width=1)

output_file = "ls20_visualized.png"
img.save(output_file)
print(f"Đã lưu ảnh trực quan hóa vào: {output_file}")
