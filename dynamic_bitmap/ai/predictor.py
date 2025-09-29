import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score

from dynamic_bitmap.ai.features import hash_bits_vector, segment_density_features, stable_hash_int
from dynamic_bitmap.ai.dataset import generate_synthetic_dataset
from dynamic_bitmap.ai.model import build_model

MODEL_DIR = "models_tf"

def train_model(bitmap, n_samples=20000, hash_bits=64, reduce_to=8, batch_size=256, epochs=10):
    """Entrena un modelo IA para predecir segmentos en un SegmentedBitmap."""
    X, y, queries = generate_synthetic_dataset(bitmap, n_samples=n_samples,
                                               hash_bits=hash_bits, reduce_to=reduce_to)

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

    model = build_model(input_dim=X.shape[1], num_segments=bitmap.num_segments)

    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt_path = os.path.join(MODEL_DIR, "best_model.h5")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )

    # Evaluar
    preds = model.predict(X_test, batch_size=512)
    y_pred = np.argmax(preds, axis=1)
    acc = accuracy_score(y_test, y_pred)
    top3 = top_k_accuracy_score(y_test, preds, k=3)
    print(f"Accuracy test: {acc:.4f} | Top-3: {top3:.4f}")

    # Guardar modelo
    saved_path = os.path.join(MODEL_DIR, "modelo.keras")
    model.save(saved_path, include_optimizer=False)
    print("Modelo guardado en:", saved_path)

    return model

def load_model(path=os.path.join(MODEL_DIR, "modelo.keras")):
    """Carga un modelo previamente entrenado."""
    return tf.keras.models.load_model(path)

def query_with_model(query_text, model, bitmap, top_k=5, hash_bits=64, reduce_to=8):
    """Usa la IA para priorizar segmentos y verificar existencia de un valor."""
    hash_feat = hash_bits_vector(query_text, n_bits=hash_bits).reshape(1, -1)
    seg_stats = segment_density_features(bitmap, reduce_to=reduce_to).reshape(1, -1)
    Xq = np.concatenate([hash_feat, seg_stats], axis=1).astype(np.float32)

    probs = model.predict(Xq, verbose=0)[0]
    segs_order = np.argsort(-probs)

    nbits = bitmap.size
    seg_size = bitmap.segment_size

    for seg in segs_order[:top_k]:
        if bitmap.segments[seg].sum() == 0:
            continue
        pos = stable_hash_int(query_text) % nbits
        if seg * seg_size <= pos < (seg + 1) * seg_size:
            idx_local = pos - seg * seg_size
            if bitmap.segments[seg][idx_local] == 1:
                return True, pos, probs[segs_order[:top_k]]

    # fallback exacto
    pos = stable_hash_int(query_text) % nbits
    seg = pos // seg_size
    found = bool(bitmap.segments[seg][pos - seg * seg_size] == 1)
    return found, pos, probs[segs_order[:top_k]]
