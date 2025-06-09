import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from caiDat import BATCH_SIZE, EARLY_STOPPING_PATIENCE, EPOCHS, LEARNING_RATE, LR_SCHEDULER_PATIENCE, MODEL_PATH, WEIGHT_DECAY, device
from tienXuLyAnh import train_loader, test_loader, val_loader, train_dataset, val_dataset
from model import model
from veMoHinh import plot_history
from danhGiaMoHinh import evaluate_and_report_on_test_set, evaluate_model

# ==== Hàm huấn luyện (Train Function) ====
# Huấn luyện mô hình qua nhiều epoch, bao gồm các bước huấn luyện, kiểm tra, dừng sớm và lưu mô hình tốt nhất.
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
        print(f"    Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"    Val  Loss: {epoch_val_loss:.4f} | Val  Acc: {epoch_val_acc:.4f}")

        scheduler.step(epoch_val_loss) # Scheduler checks validation loss

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"    💾 Best model saved to {MODEL_PATH} (Val Loss: {best_val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"    📉 Val loss did not improve for {epochs_no_improve} epoch(s). Best: {best_val_loss:.4f}")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"    🛑 Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs with no improvement.")
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

# ==== Chạy chương trình ====
# Điểm bắt đầu thực thi chương trình: khởi tạo hàm mất mát, bộ tối ưu hóa, scheduler và bắt đầu quá trình huấn luyện.
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
    if training_history['train_loss']:
        plot_history(training_history)
    else:
        print("No training history to plot (e.g., if training was interrupted immediately).")

    print("\n🏁 Script Finished 🏁")