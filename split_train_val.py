import os
import shutil
import random
from tqdm import tqdm

def split_train_val(original_dir, output_root, val_ratio=0.2):
    train_dir = os.path.join(output_root, "train")
    val_dir = os.path.join(output_root, "val")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    class_names = sorted([d for d in os.listdir(original_dir) if os.path.isdir(os.path.join(original_dir, d))])

    for class_name in class_names:
        class_path = os.path.join(original_dir, class_name)
        images = os.listdir(class_path)
        random.shuffle(images)

        split_index = int(len(images) * (1 - val_ratio))
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Tạo thư mục đích
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        print(f"🔄 Splitting class {class_name} - Total: {len(images)}")

        for img in tqdm(train_images, desc=f"Train - {class_name}", ncols=80):
            src = os.path.join(class_path, img)
            dst = os.path.join(train_class_dir, img)
            if os.path.abspath(src) != os.path.abspath(dst):  # tránh copy chính nó
                shutil.copy2(src, dst)

        for img in tqdm(val_images, desc=f"Val - {class_name}", ncols=80):
            src = os.path.join(class_path, img)
            dst = os.path.join(val_class_dir, img)
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)

if __name__ == "__main__":
    original_train_dir = "dataset_original/train"  # 🔁 thư mục chứa dữ liệu gốc ban đầu
    output_root_dir = "dataset"                    # ✅ thư mục sẽ chứa train/val tách riêng

    split_train_val(original_train_dir, output_root_dir, val_ratio=0.2)
