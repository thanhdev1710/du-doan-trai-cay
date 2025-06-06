import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==== Configuration ====
SEED = 42
INPUT_RESIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 4
LR_SCHEDULER_PATIENCE = 2
MODEL_PATH = 'mobilenetv2_best_v3.pth'
DATASET_DIR = 'data'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'valid')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
NUM_WORKERS = min(8, os.cpu_count())

# ==== Seed & Device ====
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Transforms ====
transform_train = transforms.Compose([
    transforms.Resize(INPUT_RESIZE),
    transforms.RandomRotation(15), # Increased rotation from 10 to 15 degrees
    transforms.RandomHorizontalFlip(p=0.5), # Increased probability to 0.5
    transforms.RandomVerticalFlip(p=0.2), # Kept as is, can adjust if needed
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Increased range
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10), # Added shear, adjusted scale/translate
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0) # Added RandomErasing
])

transform_val_test = transforms.Compose([
    transforms.Resize(INPUT_RESIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
])

# ==== Data Loaders ====
# Check if dataset directories exist
if not os.path.isdir(DATASET_DIR):
    print(f"❌ Error: Dataset directory '{DATASET_DIR}' not found.")
    print("Please ensure your dataset is structured as 'dataset/train', 'dataset/val', 'dataset/test'.")
    exit()
if not os.path.isdir(TRAIN_DIR):
    print(f"❌ Error: Training directory '{TRAIN_DIR}' not found.")
    exit()
if not os.path.isdir(VAL_DIR):
    print(f"❌ Error: Validation directory '{VAL_DIR}' not found.")
    exit()
if not os.path.isdir(TEST_DIR):
    print(f"❌ Error: Test directory '{TEST_DIR}' not found.")
    exit()

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

# ==== Model ====
def build_model(num_classes):
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights)
    # Custom classifier - Increased complexity and added Batch Normalization
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 512), # Increased output features from 256 to 512
        nn.ReLU(),
        nn.BatchNorm1d(512), # Added Batch Normalization for stability and better learning
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    return model.to(device)

model = build_model(len(train_dataset.classes))

# ==== Train Function ====
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_time_train = time.time()

    print("\n" + "="*60)
    print("🚀 STARTING TRAINING PROCESS 🚀")
    print("="*60)
    print(f"Device: {device}")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}, Weight Decay: {WEIGHT_DECAY}")
    print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"LR Scheduler Patience: {LR_SCHEDULER_PATIENCE}")
    print(f"Model will be saved to: {MODEL_PATH}")
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print("="*60 + "\n")

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        running_loss, correct_preds_train, total_samples_train = 0.0, 0, 0

        print(f"📅 Epoch {epoch+1}/{EPOCHS} | Current LR: {optimizer.param_groups[0]['lr']:.1e}")
        
        # Training loop with tqdm progress bar
        progress_bar_train = tqdm(train_loader, desc="Training", unit="batch", leave=False)
        for images, labels in progress_bar_train:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            correct_preds_train += (preds == labels).sum().item()
            total_samples_train += labels.size(0)
            
            # Update tqdm description with current batch loss and accuracy
            progress_bar_train.set_postfix(loss=loss.item(), acc= (preds == labels).sum().item() / images.size(0) )

        epoch_train_loss = running_loss / total_samples_train
        epoch_train_acc = correct_preds_train / total_samples_train
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        # Validation step
        model.eval()
        epoch_val_loss, epoch_val_acc = evaluate_model(model, val_loader, criterion, desc_prefix="Validating")
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        epoch_duration = time.time() - epoch_start_time

        print("-" * 60)
        print(f"✅ Epoch {epoch+1} Summary (took {epoch_duration:.2f}s):")
        print(f"   Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"   Val   Loss: {epoch_val_loss:.4f} | Val   Acc: {epoch_val_acc:.4f}")

        scheduler.step(epoch_val_loss) # Scheduler checks validation loss

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"   💾 Best model saved to {MODEL_PATH} (Val Loss: {best_val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"   📉 Val loss did not improve for {epochs_no_improve} epoch(s). Best: {best_val_loss:.4f}")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"   🛑 Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs with no improvement.")
                print("="*60)
                break
        print("="*60)

    total_training_time = time.time() - start_time_train
    print(f"\n🎉 TRAINING COMPLETE! 🎉")
    print(f"Total training time: {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s")
    
    if os.path.exists(MODEL_PATH):
        print(f"\n🔄 Loading best model from {MODEL_PATH} for final evaluation on test set...")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print(f"\n⚠️ Best model path {MODEL_PATH} not found. Using current model state for evaluation.")
        
    return history

# ==== Evaluation ====
def evaluate_model(model, loader, criterion, desc_prefix="Evaluating"):
    model.eval()
    running_loss, correct_preds, total_samples = 0.0, 0, 0
    
    progress_bar_eval = tqdm(loader, desc=desc_prefix, unit="batch", leave=False)
    with torch.no_grad():
        for images, labels in progress_bar_eval:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * images.size(0)
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)
            progress_bar_eval.set_postfix(loss=loss.item(), acc=(preds == labels).sum().item() / images.size(0))
            
    avg_loss = running_loss / total_samples
    avg_acc = correct_preds / total_samples
    return avg_loss, avg_acc

def evaluate_and_report_on_test_set(model, test_loader, criterion, class_names):
    print("\n" + "="*60)
    print("📊 EVALUATING ON TEST SET 📊")
    print("="*60)
    
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, desc_prefix="Testing")
    print(f"   Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    
    model.eval()
    y_true, y_pred = [], []
    progress_bar_report = tqdm(test_loader, desc="Generating Report", unit="batch", leave=False)
    with torch.no_grad():
        for images, labels in progress_bar_report:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\nClassification Report:\n")
    try:
        unique_labels_in_data = sorted(list(set(y_true) | set(y_pred)))
        
        report_labels = [l for l in unique_labels_in_data if l < len(class_names)]
        report_target_names = [class_names[l] for l in report_labels]

        if not report_target_names:
            print("Warning: No valid labels to report. Check consistency between dataset classes and predictions.")
        else:
            print(classification_report(
                y_true, y_pred,
                labels=report_labels, # Use filtered labels
                target_names=report_target_names, # Use filtered names
                zero_division=0
            ))
            # Calculate and print F1-score
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            print(f"\nWeighted F1-score: {f1:.4f}")
    except IndexError:
        print("Error: An issue occurred generating the classification report.")
        print("This might be due to a mismatch between predicted class indices and the length of 'class_names'.")
        print(f"Max predicted index: {max(y_pred) if y_pred else 'N/A'}, Max true index: {max(y_true) if y_true else 'N/A'}, Number of class names: {len(class_names)}")
    print("="*60)

# ==== Plot History ====
def plot_history(history):
    print("\n📈 Plotting Training History...")
    plt.figure(figsize=(18, 7)) # Adjusted figure size

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy', color='royalblue', marker='o', linestyle='-')
    plt.plot(history['val_acc'], label='Validation Accuracy', color='darkorange', marker='x', linestyle='--')
    plt.title('Training & Validation Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    # Dynamic Y-axis limits for accuracy
    min_acc = min(min(history['train_acc']), min(history['val_acc'])) if history['val_acc'] else min(history['train_acc'])
    plt.ylim(max(0, min_acc - 0.05), 1.01)


    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss', color='crimson', marker='o', linestyle='-')
    plt.plot(history['val_loss'], label='Validation Loss', color='forestgreen', marker='x', linestyle='--')
    plt.title('Training & Validation Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    max_loss = max(max(history['train_loss']), max(history['val_loss'])) if history['val_loss'] else max(history['train_loss'])
    min_loss_val = min(min(history['train_loss']), min(history['val_loss'])) if history['val_loss'] else min(history['train_loss'])
    plt.ylim(max(0, min_loss_val - 0.1), max_loss + 0.1)


    plt.tight_layout(pad=3.0)
    plot_filename = "mobilenetv2_training_plot_v3.png"
    plt.savefig(plot_filename)
    print(f"📊 Training plots saved to '{plot_filename}'")
    plt.show()

# ==== Run ====
if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         patience=LR_SCHEDULER_PATIENCE,
                                                         factor=0.5)

    # Start training
    training_history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
    
    # Evaluate the best loaded model on the test set
    evaluate_and_report_on_test_set(model, test_loader, criterion, train_dataset.classes)
    
    # Plot training history
    if training_history['train_loss']: # Check if history is not empty
        plot_history(training_history)
    else:
        print("No training history to plot (e.g., if training was interrupted immediately).")

    print("\n🏁 Script Finished 🏁")