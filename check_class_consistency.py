import os

def get_class_names(directory):
    return sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

def compare_class_names(train_dir, val_dir, test_dir):
    train_classes = get_class_names(train_dir)
    val_classes = get_class_names(val_dir)
    test_classes = get_class_names(test_dir)

    print("✅ Number of classes:")
    print(f"  - Train: {len(train_classes)}")
    print(f"  - Val:   {len(val_classes)}")
    print(f"  - Test:  {len(test_classes)}\n")

    print("📋 Comparing class names...")
    if train_classes == val_classes == test_classes:
        print("✅ All datasets have the SAME class names.\n")
    else:
        print("❌ Class name mismatch detected!\n")
        print(f"Train classes (not in val): {set(train_classes) - set(val_classes)}")
        print(f"Val classes (not in train): {set(val_classes) - set(train_classes)}\n")
        print(f"Train classes (not in test): {set(train_classes) - set(test_classes)}")
        print(f"Test classes (not in train): {set(test_classes) - set(train_classes)}")

if __name__ == '__main__':
    train_path = './dataset/train'
    val_path = './dataset/val'
    test_path = './dataset/test'

    compare_class_names(train_path, val_path, test_path)
