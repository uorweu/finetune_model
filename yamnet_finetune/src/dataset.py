import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

TARGET_SR    = 16000
PATCH_WINDOW = 0.96      # YAMNet processes 0.96s patches
PATCH_HOP    = 0.48      # 50% overlap for data augmentation
NUM_FRAMES   = int(PATCH_WINDOW * TARGET_SR)  # 15360 samples

def load_and_slice_wav(filepath, label, augment=False):
    """Load WAV and slice into fixed-length patches."""
    audio = tf.io.read_file(filepath)
    waveform, _ = tf.audio.decode_wav(audio, desired_channels=1)
    waveform = tf.squeeze(waveform, axis=-1)  # (samples,)

    # Pad if shorter than one patch
    pad_length = tf.maximum(0, NUM_FRAMES - tf.shape(waveform)[0])
    waveform = tf.pad(waveform, [[0, pad_length]])

    # Slice into patches
    patches = tf.signal.frame(waveform, NUM_FRAMES,
                               int(PATCH_HOP * TARGET_SR))

    if augment:
        # Random gain augmentation
        gain = tf.random.uniform([], 0.8, 1.2)
        patches = patches * gain
        # Additive white noise
        noise = tf.random.normal(tf.shape(patches), stddev=0.005)
        patches = patches + noise

    labels = tf.fill([tf.shape(patches)[0]], label)
    return patches, labels

def build_file_list(processed_dir):
    """Scan directory, return (file_paths, labels, class_names)."""
    class_names = sorted(os.listdir(processed_dir))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    filepaths, labels = [], []

    for class_name in class_names:
        class_dir = os.path.join(processed_dir, class_name)
        for f in os.listdir(class_dir):
            if f.endswith('.wav'):
                filepaths.append(os.path.join(class_dir, f))
                labels.append(class_to_idx[class_name])

    return filepaths, labels, class_names

def make_dataset(filepaths, labels, augment=False, batch_size=64, shuffle=True):
    """Build a flat tf.data.Dataset of (waveform_patch, label) pairs."""
    def gen():
        for fp, lbl in zip(filepaths, labels):
            try:
                patches, patch_labels = load_and_slice_wav(fp, lbl, augment)
                for i in range(patches.shape[0]):
                    yield patches[i], patch_labels[i]
            except Exception:
                pass

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(NUM_FRAMES,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        )
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=5000)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
