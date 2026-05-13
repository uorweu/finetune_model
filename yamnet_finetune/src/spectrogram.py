import tensorflow as tf

# ── Optimized params from the sweep report ────────────────────────────────────
SAMPLE_RATE = 16000
N_FFT       = 4096   # ← KEY FINDING: 4x improvement in low-freq resolution
WIN_LENGTH  = int(0.025 * SAMPLE_RATE)   # 25ms frame (400 samples)
HOP_LENGTH  = int(0.010 * SAMPLE_RATE)   # 10ms hop   (160 samples) — matches YAMNet paper
N_MELS      = 128    # highest mean accuracy in sweep
F_MIN       = 0.0
F_MAX       = 8000.0 # YAMNet is capped at 16kHz/2 = 8kHz
# ──────────────────────────────────────────────────────────────────────────────

def waveform_to_log_mel(waveform):
    """
    Convert raw waveform to normalized log-Mel spectrogram.
    Normalization (mean=0, std=1) showed clear accuracy improvement per doc 2.
    """
    stfts = tf.signal.stft(
        waveform,
        frame_length=WIN_LENGTH,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,          # larger than win_length → interpolated bins
    )
    spectrograms = tf.abs(stfts) ** 2

    # Linear → Mel
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel = tf.signal.linear_to_mel_weight_matrix(
        N_MELS, num_spectrogram_bins, SAMPLE_RATE, F_MIN, F_MAX
    )
    mel = tf.tensordot(spectrograms, linear_to_mel, 1)
    mel.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel.shape[-1:]))

    log_mel = tf.math.log(mel + 1e-6)

    # Global normalization — mean=0, std=1
    mean, variance = tf.nn.moments(log_mel, axes=[0, 1])
    log_mel_norm = (log_mel - mean) / (tf.sqrt(variance) + 1e-9)

    return log_mel_norm  # shape: (time_frames, N_MELS)
