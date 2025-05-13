# Sử dụng Python slim image để giảm kích thước
FROM python:3.10-slim

# Cài các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Sao chép requirements trước (tối ưu layer)
COPY requirements.txt .

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Chỉ định cổng mặc định cho Flask
ENV PORT=10000

# Biến môi trường Flask (vô hiệu debug)
ENV FLASK_ENV=production

# Chạy app Flask
CMD ["python", "main.py"]
