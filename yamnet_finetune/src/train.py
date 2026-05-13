import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from augment import mixup_batch
from dataset import build_file_list, make_dataset

# ── Config — updated per research docs ────────────────────────────────────────
PROCESSED_DIR   = "data/processed"
CHECKPOINT_DIR  = "checkpoints"
BATCH_SIZE      = 128    # ← updated: paper used 128
STAGE1_LR       = 3e-4   # ← updated: paper used 0.0003 (NOT 1e-3)
STAGE2_LR       = 5e-5   # keep as-is
STAGE1_EPOCHS   = 15     # paper hit 95% in 15 epochs; use EarlyStopping as safety
STAGE2_EPOCHS   = 15
MIXUP_ALPHA     = 0.1    # ← new: optimal per sweep report
# ──────────────────────────────────────────────────────────────────────────────

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

def build_model(num_classes, dropout_rate=0.4):
    waveform_input = tf.keras.Input(shape=(15360,), dtype=tf.float32, name="waveform")
    yamnet_layer = hub.KerasLayer(YAMNET_URL, trainable=False, name="yamnet")
    _, embeddings, _ = yamnet_layer(waveform_input)  # 1024-d embeddings

    # Deeper head — justified by 15-class problem vs paper's 12
    x = tf.keras.layers.Dense(512, activation='relu')(embeddings)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rate / 2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax',
                                    name="predictions")(x)
    return tf.keras.Model(inputs=waveform_input, outputs=outputs)


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    filepaths, labels, class_names = build_file_list(PROCESSED_DIR)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # ── IMPORTANT: Add a 'background' pseudo-class if not already present ──────
    # The YAMNet paper explicitly adds background noise as a class.
    # In your study area context: add recordings of empty room / ambient HVAC.
    # ──────────────────────────────────────────────────────────────────────────

    train_fps, val_fps, train_lbs, val_lbs = train_test_split(
        filepaths, labels, test_size=0.2,   # paper used 80/20
        stratify=labels, random_state=42
    )

    train_ds = make_dataset(train_fps, train_lbs, augment=True,
                            batch_size=BATCH_SIZE)
    val_ds   = make_dataset(val_fps, val_lbs, augment=False,
                            batch_size=BATCH_SIZE, shuffle=False)

    weights_array = compute_class_weight('balanced',
                                         classes=np.arange(num_classes),
                                         y=train_lbs)
    class_weights = dict(enumerate(weights_array))

    model = build_model(num_classes)
    model.summary()

    # ── STAGE 1: Train head only ───────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(STAGE1_LR),  # 3e-4 per paper
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_s1 = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(CHECKPOINT_DIR, "stage1_best.keras"),
            monitor='val_accuracy', save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir="logs/stage1"),
    ]

    print("\n=== STAGE 1: Train classification head ===")
    model.fit(train_ds, validation_data=val_ds,
              epochs=STAGE1_EPOCHS, callbacks=callbacks_s1,
              class_weight=class_weights)

    # ── STAGE 2: Unfreeze top YAMNet layers ───────────────────────────────────
    yamnet_layer = model.get_layer("yamnet")
    yamnet_layer.trainable = True
    for layer in yamnet_layer.resolved_object.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(STAGE2_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_s2 = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(CHECKPOINT_DIR, "stage2_best.keras"),
            monitor='val_accuracy', save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.3, patience=4, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir="logs/stage2"),
    ]

    print("\n=== STAGE 2: Fine-tune top YAMNet layers ===")
    model.fit(train_ds, validation_data=val_ds,
              epochs=STAGE2_EPOCHS, callbacks=callbacks_s2,
              class_weight=class_weights)

    model.save(os.path.join(CHECKPOINT_DIR, "yamnet_final.keras"))
    np.save(os.path.join(CHECKPOINT_DIR, "class_names.npy"), class_names)
    print("Done.")

if __name__ == "__main__":
    train()
