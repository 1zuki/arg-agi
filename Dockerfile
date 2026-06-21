FROM python:3.12-slim

# Cài đặt các công cụ hệ thống cơ bản
RUN apt-get update && apt-get install -y \
    git \
    wget \
    nano \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt PyTorch hỗ trợ CUDA (GPU) để train model
# Code của bạn không dùng torchvision hay torchaudio, nên ta bỏ đi cho nhẹ
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Cài đặt các thư viện xử lý ma trận và hình ảnh được dùng trong code
RUN pip install --no-cache-dir numpy Pillow

# Thiết lập thư mục làm việc mặc định
WORKDIR /workspace

# Khởi chạy bash shell khi vào container
CMD ["/bin/bash"]
