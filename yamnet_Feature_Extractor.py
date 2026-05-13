print('-----------------------------note-------------------------')
print('I added some lines to turn off warning from tensorflow libraries')
print('--------------------------------------------------------------')

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
import tensorflow as tf
import tensorflow_hub as hub

# # 4. Final 'old school' TF silencing for legacy Keras warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# To be extra sure, tell TensorFlow's logger to only show Errors
tf.get_logger().setLevel('ERROR')

print('Scores part')

# 1. Load the model
# Note: This requires an internet connection to download the ~30MB model
model = hub.load('https://tfhub.dev/google/yamnet/1')

# 2. Get the labels
class_map_path = model.class_map_path().numpy().decode('utf-8')
class_names = pd.read_csv(class_map_path)['display_name'].values

# 3. Generate sample data (This fixes the 'scores' error)
# YAMNet expects a 1-D float32 Tensor (mono audio at 16kHz)
# Here we create 1 second of silence as a placeholder
waveform = np.zeros(16000, dtype=np.float32)
scores, embeddings, spectrogram = model(waveform)

# 4. Process the scores
# scores.shape is (N, 521), where N is the number of audio frames
example_scores = scores[0].numpy() 

# 5. Print the results
print(f"{'Index':<10} | {'Label Name':<60} | {'Confidence Score'}")
print("-" * 60)

for i in range(len(example_scores)):
    label = class_names[i]
    score = example_scores[i]
    # No filter - show everything
    print(f"{i:<10} | {label:<60} | {score:.5f}")
print('--------------------------------------------------------------')
print('The output for this code represent:')
print('The score for each class that Yamnet get, which is 0 for all except Silence (494) - Because no audio data given')
print('So the scores of the model is stand for "The Score can be used for how many classes - which is 521"')

print('---------------------------------------')
print('Embeddings part')
# ... (include the suppression block from the previous step here for a clean output)

# 1. Run the model (using the same waveform of silence)
waveform = np.zeros(16000, dtype=np.float32)
scores, embeddings, spectrogram = model(waveform)

# 2. Convert the embeddings to a NumPy array
# The shape will be (N, 1024), where N is the number of 0.48s frames
embedding_values = embeddings.numpy()

print(f"Embedding Shape: {embedding_values.shape}")
print("-" * 30)

# 3. Look at the first 10 values of the first frame's embedding
print("First 10 values of the 1024-D embedding vector:")
print(embedding_values[0][:10])

print('-----------------------------------------')
print('Log Mel Spectrogram part')

import matplotlib.pyplot as plt
import numpy as np
# 2. Load a real .wav file
# Replace 'your_audio.wav' with the path to your file!
# CRITICAL: YAMNet expects mono (1-channel) audio at exactly 16,000 Hz.
file_path = './audio_dataset/classes/Laughter/dianasue500-male-deep-laugh-328480.wav' 
file_contents = tf.io.read_file(file_path)
waveform, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
waveform = tf.squeeze(waveform, axis=-1)  # Flatten to a 1D array

# 3. Run the audio through YAMNet
scores, embeddings, spectrogram = model(waveform)

# 4. Create the visual plots
plt.figure(figsize=(12, 10)) # Make a large window

# --- PLOT 1: Time Domain (Raw Waveform) ---
plt.subplot(3, 1, 1)
plt.plot(waveform.numpy())
plt.title('1. Time Domain (The Raw Sound Wave)')
plt.ylabel('Amplitude')
plt.xlabel('Time (in samples)')
plt.grid(True, alpha=0.3)

# --- PLOT 2: Frequency Domain (Spectrum/FFT) ---
plt.subplot(3, 1, 2)
# Matplotlib has a built-in function to calculate the frequency spectrum
plt.magnitude_spectrum(waveform.numpy(), Fs=16000, scale='dB', color='orange')
plt.title('2. Frequency Domain (Power Spectrum)')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
plt.grid(True, alpha=0.3)

# --- PLOT 3: YAMNet Log Mel Spectrogram ---
plt.subplot(3, 1, 3)
plt.imshow(spectrogram.numpy().T, aspect='auto', interpolation='nearest', origin='lower')
plt.title('3. YAMNet Log Mel Spectrogram (Frequency over Time)')
plt.ylabel('Mel Bin (Frequency)')
plt.xlabel('Time Frames')

# Adjust layout so labels don't overlap and show the plot
plt.tight_layout()
plt.show()

print('-----------------------------------')
print('Now assessing some audio file')

import audio_evaluator # Import file phụ vừa tạo

# ... (Phần code load model và class_names của bạn) ...

print("Running general assessments")
# Call the side file:
path_to_audio = './audio_dataset/classes/Laughter/dianasue500-male-deep-laugh-328480.wav'
result = audio_evaluator.evaluate_audio(model, class_names, path_to_audio)

print(f"\nFinal result: {result}")
