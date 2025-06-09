from sklearn.metrics import classification_report, f1_score
import torch
from tqdm import tqdm
from caiDat import device

# ==== Đánh giá (Evaluation) ====
# Đánh giá hiệu suất của mô hình trên tập dữ liệu đã cho (ví dụ: tập kiểm tra hoặc thử nghiệm).
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

# Đánh giá mô hình trên tập thử nghiệm và in báo cáo phân loại chi tiết.
def evaluate_and_report_on_test_set(model, test_loader, criterion, class_names):
    print("\n" + "="*60)
    print("📊 EVALUATING ON TEST SET 📊")
    print("="*60)
    
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, desc_prefix="Testing")
    print(f"    Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    
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
                labels=report_labels,
                target_names=report_target_names,
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