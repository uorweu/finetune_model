import numpy as np

def evaluate_audio(model, class_names, file_path):
    import tensorflow as tf
    
    # Load and run the audio
    file_contents = tf.io.read_file(file_path)
    waveform, _ = tf.audio.decode_wav(file_contents, desired_channels=1)
    waveform = tf.squeeze(waveform, axis=-1)

    # Run the model
    scores, _, _ = model(waveform)
    mean_scores = np.mean(scores, axis=0)
    
    # Take Top 5
    top_indices = np.argsort(mean_scores)[::-1][:5]
    
    print(f"\n--- Details assessments for: {file_path.split('/')[-1]} ---")
    for i in top_indices:
        print(f"{class_names[i]}: {mean_scores[i]:.4f}")
    
    return class_names[top_indices[0]] # Return to the highest scored class
