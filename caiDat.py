import os
import torch

# ==== Cấu hình ====
# Định nghĩa các hằng số cấu hình cho quá trình huấn luyện.
SEED = 42
INPUT_RESIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 3
LR_SCHEDULER_PATIENCE = 2
MODEL_PATH = 'mobilenetv2_best_v3.pth'
PLOT_PATH = 'mobilenetv2_training_plot_v3.png'
CLASS_NAMES_PATH = 'class_names.txt'
DATASET_DIR = 'data'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'valid')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
NUM_WORKERS = min(8, os.cpu_count())

# ==== Seed & Thiết bị ====
# Đặt seed cho các thư viện để đảm bảo tính lặp lại của kết quả và chọn thiết bị (CPU/GPU).
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")