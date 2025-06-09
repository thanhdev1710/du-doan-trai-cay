import os
import torch
from model import build_model
from tienXuLyAnh import transform_val_test
from caiDat import CLASS_NAMES_PATH, MODEL_PATH, device

# ==== Load class names ====
if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"File '{CLASS_NAMES_PATH}' not found.")
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

model = build_model(num_classes=len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ==== Predict function ====
def predict_image(image, topk=1):
    image = image.convert('RGB')
    input_tensor = transform_val_test(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_probs, top_idxs = torch.topk(probabilities, k=topk)

    results = [(class_names[idx], round(prob.item() * 100, 2)) for prob, idx in zip(top_probs, top_idxs)]
    return results
