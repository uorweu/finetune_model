import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from dataset import build_file_list, make_dataset

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR  = "data/processed"
EMBEDDINGS_DIR = "data/embeddings"
CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE_S1 = 512     # embeddings are tiny — large batches are fine
BATCH_SIZE_S2 = 64      # full waveforms through YAMNet — be conservative

STAGE1_LR     = 3e-4
STAGE2_LR     = 5e-5
STAGE1_EPOCHS = 30      # fast epochs so the head has room to fully converge
STAGE2_EPOCHS = 15

YAMNET_URL    = "https://tfhub.dev/google/yamnet/1"
# ─────────────────────────────────────────────────────────────────────────────


def setup_gpu():
    """Enable GPU memory growth + mixed precision if a GPU is available."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print(f"GPU detected ({len(gpus)} device(s)) — mixed_float16 enabled.")
        return True
    print("No GPU detected — running on CPU.")
    return False


def build_head_model(num_classes, dropout_rate=0.4):
    """
    Stage 1 model.
    Input: 1024-d pre-extracted YAMNet embeddings.
    No YAMNet call needed during Stage 1 training.
    """
    inp = tf.keras.Input(shape=(1024,), dtype=tf.float32, name="embedding")
    x = tf.keras.layers.Dense(512, activation='relu',  name='head_dense1')(inp)
    x = tf.keras.layers.BatchNormalization(name='head_bn1')(x)
    x = tf.keras.layers.Dropout(dropout_rate,          name='head_drop1')(x)
    x = tf.keras.layers.Dense(256, activation='relu',  name='head_dense2')(x)
    x = tf.keras.layers.BatchNormalization(name='head_bn2')(x)
    x = tf.keras.layers.Dropout(dropout_rate / 2,      name='head_drop2')(x)
    # dtype='float32' keeps softmax in full precision even under mixed precision
    out = tf.keras.layers.Dense(num_classes, activation='softmax',
                                dtype='float32', name='predictions')(x)
    return tf.keras.Model(inputs=inp, outputs=out, name='head_model')


def build_full_model(num_classes, dropout_rate=0.4):
    """
    Stage 2 model. Returns (model, yamnet_layer) so the caller can
    unfreeze YAMNet layers directly — get_layer('yamnet') won't work
    because yamnet is captured inside the Lambda, not a top-level layer.
    """
    inp = tf.keras.Input(shape=(15360,), dtype=tf.float32, name="waveform")

    # Keep a local reference — this is what we return for unfreezing
    yamnet_layer = hub.KerasLayer(YAMNET_URL, trainable=False, name="yamnet")

    def _embed(wave):
        _, emb, _ = yamnet_layer(wave)
        return tf.reduce_mean(emb, axis=0)   # (1024,)

    embeddings = tf.keras.layers.Lambda(
        lambda batch: tf.map_fn(_embed, batch, fn_output_signature=tf.float32),
        output_shape=(1024,),
        name="yamnet_pool",
    )(inp)

    x = tf.keras.layers.Dense(512, activation='relu',  name='head_dense1')(embeddings)
    x = tf.keras.layers.BatchNormalization(name='head_bn1')(x)
    x = tf.keras.layers.Dropout(dropout_rate,          name='head_drop1')(x)
    x = tf.keras.layers.Dense(256, activation='relu',  name='head_dense2')(x)
    x = tf.keras.layers.BatchNormalization(name='head_bn2')(x)
    x = tf.keras.layers.Dropout(dropout_rate / 2,      name='head_drop2')(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax',
                                dtype='float32', name='predictions')(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name='full_model')
    return model, yamnet_layer   # ← return both so caller can unfreeze


def transfer_head_weights(head_model, full_model):
    """Copy trained head weights from Stage 1 into the Stage 2 full model."""
    for name in ['head_dense1', 'head_bn1', 'head_dense2', 'head_bn2', 'predictions']:
        full_model.get_layer(name).set_weights(
            head_model.get_layer(name).get_weights()
        )
    print("Head weights transferred from Stage 1 → Stage 2 model.")


def make_embedding_dataset(embeddings, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((embeddings, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(embeddings), seed=42)
    return ds.batch(BATCH_SIZE_S1).prefetch(tf.data.AUTOTUNE)


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    setup_gpu()

    # ── Verify embeddings exist ───────────────────────────────────────────────
    emb_path = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
    if not os.path.exists(emb_path):
        raise SystemExit(
            f"'{emb_path}' not found.\n"
            "Run  python extract_embeddings.py  first."
        )

    class_names = np.load(
        os.path.join(EMBEDDINGS_DIR, "class_names.npy"), allow_pickle=True
    ).tolist()
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    embeddings = np.load(os.path.join(EMBEDDINGS_DIR, "embeddings.npy"))
    emb_labels = np.load(os.path.join(EMBEDDINGS_DIR, "labels.npy"))
    print(f"Embeddings loaded: {embeddings.shape}")

    train_emb, val_emb, train_lbl, val_lbl = train_test_split(
        embeddings, emb_labels,
        test_size=0.2, stratify=emb_labels, random_state=42
    )

    train_emb_ds = make_embedding_dataset(train_emb, train_lbl, shuffle=True)
    val_emb_ds   = make_embedding_dataset(val_emb,   val_lbl,   shuffle=False)

    weights_array = compute_class_weight(
        'balanced', classes=np.arange(num_classes), y=train_lbl
    )
    class_weights = dict(enumerate(weights_array))

    # ── STAGE 1: Train classification head on saved embeddings ────────────────
    print("\n=== STAGE 1: Train classification head (embedding mode) ===")
    head_model = build_head_model(num_classes)
    head_model.summary()

    head_model.compile(
        optimizer=tf.keras.optimizers.Adam(STAGE1_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_s1 = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(CHECKPOINT_DIR, "stage1_best.keras"),
            monitor='val_accuracy', save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
    ]

    head_model.fit(
        train_emb_ds, validation_data=val_emb_ds,
        epochs=STAGE1_EPOCHS, callbacks=callbacks_s1,
        class_weight=class_weights
    )

    # ── STAGE 2: Fine-tune top YAMNet layers on raw waveforms ────────────────
    print("\n=== STAGE 2: Fine-tune top YAMNet layers (waveform mode) ===")

    filepaths, labels, _ = build_file_list(PROCESSED_DIR)
    train_fps, val_fps, train_lbs, val_lbs = train_test_split(
        filepaths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_ds = make_dataset(train_fps, train_lbs, augment=True,
                            batch_size=BATCH_SIZE_S2)
    val_ds   = make_dataset(val_fps, val_lbs, augment=False,
                            batch_size=BATCH_SIZE_S2, shuffle=False)

    # build_full_model returns (model, yamnet_layer) — unpack both
    full_model, yamnet_layer = build_full_model(num_classes)
    transfer_head_weights(head_model, full_model)

    # Unfreeze top 20 YAMNet layers using the direct reference
    yamnet_layer.trainable = True

    full_model.compile(
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
    ]

    full_model.fit(
        train_ds, validation_data=val_ds,
        epochs=STAGE2_EPOCHS, callbacks=callbacks_s2,
        class_weight=class_weights
    )

    full_model.save(os.path.join(CHECKPOINT_DIR, "yamnet_final.keras"))
    np.save(os.path.join(CHECKPOINT_DIR, "class_names.npy"), np.array(class_names))
    print("\nDone. Final model saved to checkpoints/yamnet_final.keras")


if __name__ == "__main__":
    train()

