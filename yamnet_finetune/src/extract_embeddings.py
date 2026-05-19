"""
One-time embedding extraction. Run AFTER preprocess.py, BEFORE train.py.

    python extract_embeddings.py

Runs YAMNet (frozen) over every audio patch exactly once and saves
the 1024-d embeddings to data/embeddings/. Stage 1 training then
operates on these saved embeddings — no YAMNet call per epoch.

Stage 1: seconds/epoch  (was hours/epoch)
Stage 2: still uses full waveforms since YAMNet is unfrozen there.
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from tqdm import tqdm

PROCESSED_DIR  = "data/processed"
EMBEDDINGS_DIR = "data/embeddings"
YAMNET_URL     = "https://tfhub.dev/google/yamnet/1"
PATCH_SAMPLES  = 15360   # 0.96s at 16kHz
HOP_SAMPLES    = 7680    # 50% overlap (matches dataset.py)


def load_and_patch(filepath):
    """Load a WAV file and slice into 0.96s patches. Returns list of np arrays."""
    y, _ = librosa.load(filepath, sr=16000, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-9)

    patches = []
    for start in range(0, max(len(y) - PATCH_SAMPLES + 1, 1), HOP_SAMPLES):
        chunk = y[start:start + PATCH_SAMPLES]
        if len(chunk) < PATCH_SAMPLES:
            chunk = np.pad(chunk, (0, PATCH_SAMPLES - len(chunk)))
        patches.append(chunk.astype(np.float32))
    return patches


def main():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    print("Loading YAMNet from TF Hub...")
    yamnet = hub.load(YAMNET_URL)
    print("YAMNet loaded.\n")

    # Scan class folders
    class_names = sorted(
        d for d in os.listdir(PROCESSED_DIR)
        if os.path.isdir(os.path.join(PROCESSED_DIR, d))
    )
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print(f"Found {len(class_names)} classes.\n")

    all_embeddings = []
    all_labels     = []
    errors         = 0

    for class_name in class_names:
        class_dir = os.path.join(PROCESSED_DIR, class_name)
        files = sorted(f for f in os.listdir(class_dir) if f.endswith(".wav"))

        for fname in tqdm(files, desc=f"{class_name:<30}"):
            filepath = os.path.join(class_dir, fname)
            try:
                patches = load_and_patch(filepath)
                for patch in patches:
                    waveform = tf.constant(patch, dtype=tf.float32)
                    _, emb, _ = yamnet(waveform)                  # (frames, 1024)
                    embedding = tf.reduce_mean(emb, axis=0).numpy()  # (1024,)
                    all_embeddings.append(embedding)
                    all_labels.append(class_to_idx[class_name])
            except Exception as e:
                errors += 1
                print(f"\n  ⚠  Skipped {fname}: {e}")

    all_embeddings = np.array(all_embeddings, dtype=np.float32)
    all_labels     = np.array(all_labels,     dtype=np.int32)

    np.save(os.path.join(EMBEDDINGS_DIR, "embeddings.npy"),  all_embeddings)
    np.save(os.path.join(EMBEDDINGS_DIR, "labels.npy"),      all_labels)
    np.save(os.path.join(EMBEDDINGS_DIR, "class_names.npy"), np.array(class_names))

    print(f"\nSaved {len(all_embeddings)} embeddings → {EMBEDDINGS_DIR}/")
    print(f"Embedding matrix shape: {all_embeddings.shape}")
    if errors:
        print(f"Skipped {errors} files due to errors.")


if __name__ == "__main__":
    main()
