import numpy as np
import tensorflow as tf

def mixup_batch(x, y, num_classes, alpha=0.1):
    """
    MixUp augmentation — interpolates pairs of samples.
    alpha=0.1 was the optimal value per the sweep report.
    """
    batch_size = tf.shape(x)[0]
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # always keep dominant sample > 0.5

    indices = tf.random.shuffle(tf.range(batch_size))
    x_mix = lam * x + (1 - lam) * tf.gather(x, indices)

    y_onehot = tf.one_hot(y, num_classes)
    y_mix = lam * y_onehot + (1 - lam) * tf.gather(y_onehot, indices)

    return x_mix, y_mix


def add_background_noise(waveform, noise_waveform, snr_db=10.0):
    """
    Mix in real background noise at a given SNR.
    Critical for public study area environment — chairs, footsteps, HVAC, etc.
    """
    signal_power = tf.reduce_mean(tf.square(waveform))
    noise_power  = tf.reduce_mean(tf.square(noise_waveform))
    snr_linear   = 10.0 ** (snr_db / 10.0)
    scale = tf.sqrt(signal_power / (snr_linear * noise_power + 1e-9))
    return waveform + scale * noise_waveform
