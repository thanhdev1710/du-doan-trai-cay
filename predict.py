import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
from PIL import Image

# ==== Config ====
MODEL_PATH = 'mobilenetv2_best_v3.pth'
CLASS_NAMES_PATH = 'class_names.txt'
INPUT_RESIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Load class names ====
if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"File '{CLASS_NAMES_PATH}' not found.")
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# ==== Define transforms ====
transform = transforms.Compose([
    transforms.Resize(INPUT_RESIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==== Build model ====
def build_model(num_classes):
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 512), # Increased output features from 256 to 512
        nn.ReLU(),
        nn.BatchNorm1d(512), # Added Batch Normalization for stability and better learning
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    return model

model = build_model(num_classes=len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ==== Predict function ====
def predict_image(image, topk=1):
    image = image.convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_probs, top_idxs = torch.topk(probabilities, k=topk)

    results = [(class_names[idx], round(prob.item() * 100, 2)) for prob, idx in zip(top_probs, top_idxs)]
    return results
