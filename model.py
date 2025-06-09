from torchvision import models
from torchvision.models import MobileNet_V2_Weights
import torch.nn as nn
from caiDat import device
from tienXuLyAnh import train_dataset

# ==== Mô hình ====
# Xây dựng và khởi tạo mô hình MobileNetV2 với bộ phân loại tùy chỉnh.
def build_model(num_classes):
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)
    # Custom classifier - Increased complexity and added Batch Normalization
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    return model.to(device)

model = build_model(len(train_dataset.classes))