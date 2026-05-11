import os
import warnings

# 1. Tắt các log của TensorFlow (0: Tất cả, 1: Ẩn INFO, 2: Ẩn INFO & WARNING, 3: Ẩn tất cả kể cả ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 2. Tắt cảnh báo về oneDNN (Dòng chữ vàng đầu tiên trong hình của bạn)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 3. Tắt các cảnh báo từ thư viện Python (như DeprecationWarning)
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

def load_yamnet_from_hub():
    print("")
    print("--- Loading YAMNet from TensorFlow Hub ---")
    print("")
    try:
        # Official YAMNet model URL
        model_url = 'https://tfhub.dev/google/yamnet/1'
        
        # Load the model as a KerasLayer for fine-tuning
        model = hub.load(model_url)
        
        # YAMNet returns 3 outputs: scores, embeddings, and log_mel_spectrogram
        # Let's test with a dummy signal (1 second of silence at 16kHz)
        testing_signal = np.zeros(16000, dtype=np.float32)
        
        scores, embeddings, spectrogram = model(testing_signal)
        
        print(f"Model loaded successfully!")
        print(f"Output scores shape: {scores.shape}")
        print(f"Embeddings shape: {embeddings.shape}")
        print("\nStatus: SUCCESS - Ready to train.")
        
    except Exception as e:
        print(f"Status: FAILED - Error: {e}")

if __name__ == "__main__":
    load_yamnet_from_hub()
