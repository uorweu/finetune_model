import os
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

TARGET_SR = 16000  # YAMNet requires 16kHz mono
MIN_DURATION = 0.5  # seconds

def preprocess_audio(input_path, output_path):
    """Resample, convert to mono, normalize."""
    try:
        y, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)
        if len(y) / TARGET_SR < MIN_DURATION:
            return False  # skip too-short files
        # Peak normalize
        y = y / (np.max(np.abs(y)) + 1e-9)
        sf.write(output_path, y, TARGET_SR)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def preprocess_dataset(raw_dir, out_dir):
    """
    Expects raw_dir/class_name/*.wav structure.
    Outputs out_dir/class_name/*.wav
    """
    for class_name in os.listdir(raw_dir):
        class_in = os.path.join(raw_dir, class_name)
        class_out = os.path.join(out_dir, class_name)
        if not os.path.isdir(class_in):
            continue
        os.makedirs(class_out, exist_ok=True)

        files = [f for f in os.listdir(class_in) if f.endswith('.wav')]
        for f in tqdm(files, desc=class_name):
            preprocess_audio(
                os.path.join(class_in, f),
                os.path.join(class_out, f)
            )

if __name__ == "__main__":
    preprocess_dataset("data/raw", "data/processed")
