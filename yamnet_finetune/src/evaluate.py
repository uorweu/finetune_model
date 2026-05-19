import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import build_file_list, make_dataset
from sklearn.model_selection import train_test_split

CHECKPOINT_DIR = "checkpoints"
PROCESSED_DIR  = "data/processed"

def predict_file(model, filepath, class_names):
    """Average patch predictions → single file-level confidence."""
    import librosa
    y, _ = librosa.load(filepath, sr=16000, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-9)

    patch_size = 15360
    hop        = patch_size // 2
    patches = []
    for start in range(0, max(len(y) - patch_size + 1, 1), hop):
        chunk = y[start:start + patch_size]
        if len(chunk) < patch_size:
            chunk = np.pad(chunk, (0, patch_size - len(chunk)))
        patches.append(chunk)

    patches = np.array(patches, dtype=np.float32)
    probs   = model.predict(patches, verbose=0)  # (num_patches, num_classes)
    avg_prob = probs.mean(axis=0)               # aggregate across patches
    pred_class = class_names[np.argmax(avg_prob)]
    confidence = float(np.max(avg_prob))
    return pred_class, confidence

def evaluate():
    model = tf.keras.models.load_model(
        os.path.join(CHECKPOINT_DIR, "stage2_best.keras")
    )
    class_names = np.load(
        os.path.join(CHECKPOINT_DIR, "class_names.npy"), allow_pickle=True
    )

    filepaths, labels, _ = build_file_list(PROCESSED_DIR)
    _, test_fps, _, test_lbs = train_test_split(
        filepaths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    y_true, y_pred, confidences = [], [], []
    for fp, lbl in zip(test_fps, test_lbs):
        pred_class, conf = predict_file(model, fp, class_names)
        y_true.append(class_names[lbl])
        y_pred.append(pred_class)
        confidences.append(conf)

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print(f"\nMean confidence: {np.mean(confidences):.4f}")
    print(f"Files with conf ≥ 0.95: "
          f"{sum(c >= 0.95 for c in confidences)}/{len(confidences)}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names,
                yticklabels=class_names, cmap='Blues')
    plt.title("Confusion Matrix — Fine-tuned YAMNet")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved to confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
