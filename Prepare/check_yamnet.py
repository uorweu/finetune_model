import os
import logging
import warnings

# 1. Silences the C++ level (hardware/optimization logs)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. Silences Python's standard warning system (the Deprecation one you're seeing)
warnings.filterwarnings('ignore')

# 3. Silences TensorFlow's internal logger
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import tensorflow_hub as hub


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
