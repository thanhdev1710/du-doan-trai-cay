import os
import numpy as np
from glob import glob
from tqdm import tqdm
import shutil

# Danh sách các class cần xử lý
class_names = ['apple', 'banana', 'grape', 'mango', 'strawberry']

# Thư mục chứa dữ liệu gốc
image_base_path = "./dataset_original/train"

# Thư mục xuất kết quả
train_dir = "./data/train"
valid_dir = "./data/valid"
test_dir = "./data/test"

# Tạo thư mục train/valid/test và các thư mục con cho từng class
for directory in [train_dir, valid_dir, test_dir]:
    os.makedirs(directory, exist_ok=True)
    for name in class_names:
        os.makedirs(os.path.join(directory, name), exist_ok=True)

# Lấy danh sách ảnh cho từng class
all_class_paths = []
for name in class_names:
    class_path = os.path.join(image_base_path, name)
    image_paths = glob(f"{class_path}/*")
    np.random.shuffle(image_paths)
    all_class_paths.append(image_paths)

# Phân chia tỷ lệ
train_ratio = 0.70
valid_ratio = 0.20
test_ratio  = 0.10

# Duyệt qua từng class và chia ảnh
for class_name, image_paths in zip(class_names, all_class_paths):
    n_total = len(image_paths)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)

    train_images = image_paths[:n_train]
    valid_images = image_paths[n_train:n_train + n_valid]
    test_images  = image_paths[n_train + n_valid:]

    # Sao chép ảnh
    for path in tqdm(train_images, desc=f"[{class_name}] Train"):
        shutil.copy(path, os.path.join(train_dir, class_name, os.path.basename(path)))
    for path in tqdm(valid_images, desc=f"[{class_name}] Valid"):
        shutil.copy(path, os.path.join(valid_dir, class_name, os.path.basename(path)))
    for path in tqdm(test_images, desc=f"[{class_name}] Test"):
        shutil.copy(path, os.path.join(test_dir, class_name, os.path.basename(path)))

print("\n✅ Tách dữ liệu thành công! Kết quả nằm trong các thư mục: train/, valid/, test/")
