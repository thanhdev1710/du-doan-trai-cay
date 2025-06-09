from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from caiDat import INPUT_RESIZE, VAL_DIR, TRAIN_DIR, TEST_DIR, device, BATCH_SIZE, NUM_WORKERS

# ==== Biến đổi dữ liệu (Transforms) ====
# Định nghĩa các phép biến đổi ảnh cho tập huấn luyện và tập kiểm tra/thử nghiệm.
transform_train = transforms.Compose([
    transforms.Resize(INPUT_RESIZE),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
])

transform_val_test = transforms.Compose([
    transforms.Resize(INPUT_RESIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
])

# ==== Trình tải dữ liệu (Data Loaders) ====
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform_train)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform_val_test)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform_val_test)

# Use pin_memory if CUDA is available
pin_memory_flag = True if device.type == 'cuda' else False

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=pin_memory_flag)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=pin_memory_flag)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin_memory_flag)
