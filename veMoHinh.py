import matplotlib.pyplot as plt
from caiDat import PLOT_PATH

# ==== Vẽ lịch sử (Plot History) ====
# Vẽ biểu đồ hiển thị lịch sử loss và độ chính xác trong quá trình huấn luyện.
def plot_history(history):
    print("\n📈 Plotting Training History...")
    plt.figure(figsize=(18, 7))

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
    plt.savefig(PLOT_PATH)
    print(f"📊 Training plots saved to '{PLOT_PATH}'")
    plt.show()