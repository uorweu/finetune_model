import tensorflow as tf
import tensorflow_hub as hub

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

def build_model(num_classes: int, dropout_rate=0.4):
    """
    Stage 1: Frozen YAMNet backbone + trainable head.
    Stage 2 (optional): Unfreeze top layers.
    """
    # Input: raw waveform patch
    waveform_input = tf.keras.Input(shape=(15360,), dtype=tf.float32,
                                    name="waveform")

    # Load YAMNet — extract embeddings (1024-d), not class scores
    yamnet_layer = hub.KerasLayer(YAMNET_URL, trainable=False, name="yamnet")
    _, embeddings, _ = yamnet_layer(waveform_input)  # (batch, 1024)

    # Classification head
    x = tf.keras.layers.Dense(512, activation='relu')(embeddings)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate / 2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax',
                                    name="predictions")(x)

    model = tf.keras.Model(inputs=waveform_input, outputs=outputs)
    return model

def unfreeze_yamnet_top(model, num_layers_to_unfreeze=20):
    """Stage 2: Gradually unfreeze top N layers of YAMNet for domain adaptation."""
    yamnet_layer = model.get_layer("yamnet")
    yamnet_layer.trainable = True
    # Freeze all but the last N layers inside YAMNet
    for layer in yamnet_layer.resolved_object.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
    return model
