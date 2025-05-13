import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
from PIL import Image

# ==== Config ====
MODEL_PATH = 'mobilenetv2_best.pth'
CLASS_NAMES_PATH = 'class_names.txt'
INPUT_SIZE = (96, 96)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load class names ====
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# ==== Transform ====
transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==== Model ====
def load_model():
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, len(class_names))
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ==== Predict Function ====
def predict_image(image: Image.Image):
    """
    Nhận ảnh PIL.Image, xử lý và trả về tên lớp dự đoán
    """
    image = image.convert('RGB')  # đảm bảo là ảnh RGB
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_idx = predicted.item()
        return class_names[class_idx]
