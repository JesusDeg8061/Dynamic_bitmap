"""
ejemplos/ejemplo_ia_tensorflow.py
Pipeline completo de entrenamiento con TensorFlow (Keras) para predecir segmentos
a partir de patrones en un SegmentedBitmap.

Requisitos:
    pip install numpy tensorflow scikit-learn

Ejecutar:
    python ejemplos/ejemplo_ia_tensorflow.py
"""

import os
import numpy as np
import hashlib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from dynamic_bitmap.segmented_bitmap import SegmentedBitmap

# -----------------------
# Parámetros configurables
# -----------------------
N_BITS = 16384        # tamaño del bitmap global
S_SEGMENTS = 128      # número de segmentos
N_SAMPLES = 20000     # cantidad de ejemplos sintéticos para entrenar
HASH_BITS = 64        # número de bits para vectorizar hash (feature)
BATCH_SIZE = 256
EPOCHS = 10
MODEL_DIR = "models_tf"

# -----------------------
# Util: hash estable -> int
# -----------------------
def stable_hash_int(x: str) -> int:
    """Convierte string a entero usando SHA256."""
    return int(hashlib.sha256(str(x).encode()).hexdigest(), 16)

# -----------------------
# Feature builder
# -----------------------
def hash_bits_vector(query_text: str, n_bits=HASH_BITS):
    """Convierte hash SHA256 en vector binario de n_bits (LSB..MSB)."""
    h = stable_hash_int(query_text)
    bits = np.array([(h >> i) & 1 for i in range(n_bits)], dtype=np.float32)
    return bits  # shape (n_bits,)

def segment_density_features(bitmap: SegmentedBitmap, reduce_to=8):
    """
    Calcula densidad por segmento y lo reduce a 'reduce_to' estadísticas:
    (mean, var, top-k densidades) o una proyección simple.
    Retorna vector de tamaño reduce_to.
    """
    popcounts = np.array([int(seg.sum()) for seg in bitmap.segments], dtype=np.float32)
    densities = popcounts / (bitmap.segment_size + 1e-9)  # normalizar
    mean = densities.mean()
    var = densities.var()
    topk = np.sort(densities)[- (reduce_to - 2):] if reduce_to > 2 else []
    vec = np.concatenate(([mean, var], np.pad(topk, (0, max(0, (reduce_to - 2) - len(topk))), 'constant')))
    return vec.astype(np.float32)  # shape (reduce_to,)

# -----------------------
# Generador de dataset sintético
# -----------------------
def generate_synthetic_dataset(bitmap: SegmentedBitmap, n_samples=N_SAMPLES, hash_space_factor=5):
    """
    Crea X ejemplos sintéticos:
      - queries aleatorios
      - etiqueta: segmento objetivo donde caería el hash (0..S-1)
    Retorna X (features), y (labels int), y meta (queries list)
    """
    S = bitmap.num_segments
    n_bits = bitmap.size

    X_hash_bits = np.zeros((n_samples, HASH_BITS), dtype=np.float32)
    X_seg_stats = np.zeros((n_samples, 8), dtype=np.float32)  # reduce_to=8
    y = np.zeros((n_samples,), dtype=np.int32)
    queries = []

    for i in range(n_samples):
        val = np.random.randint(0, n_bits * hash_space_factor)
        q = f"q_{val}"
        queries.append(q)

        h = stable_hash_int(q)
        pos = h % n_bits
        seg = pos // bitmap.segment_size
        if seg >= S:
            seg = S - 1

        y[i] = int(seg)
        X_hash_bits[i, :] = hash_bits_vector(q, n_bits=HASH_BITS)
        X_seg_stats[i, :] = segment_density_features(bitmap, reduce_to=8)

    X = np.concatenate([X_hash_bits, X_seg_stats], axis=1)  # shape (n_samples, HASH_BITS + 8)
    return X, y, queries

# -----------------------
# Modelo Keras (MLP)
# -----------------------
def build_model(input_dim, s_segments):
    inputs = tf.keras.Input(shape=(input_dim,), name="input")
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(s_segments, activation="softmax", name="out")(x)

    # Métricas para labels enteros
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")]
    )
    return model

# -----------------------
# Entrenamiento / Evaluación
# -----------------------
def train_and_evaluate():
    sb = SegmentedBitmap(size=N_BITS, num_segments=S_SEGMENTS, stable_hash=True)

    # Hotspots: segmentos parcialmente llenos
    hotspots = [2, 5, 10, 20, 50]
    for seg in hotspots:
        base = seg * sb.segment_size
        for v in range(base, base + int(sb.segment_size * 0.15)):
            sb.segments[seg][v - base] = 1

    # Inserciones dispersas
    for _ in range(1000):
        val = np.random.randint(0, N_BITS)
        seg, idx = val // sb.segment_size, val % sb.segment_size
        sb.segments[seg][idx] = 1

    print("Generando dataset sintético...")
    X, y, queries = generate_synthetic_dataset(sb, n_samples=N_SAMPLES)

    # Split train / val / test
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

    print("Tamaños:", X_train.shape, X_val.shape, X_test.shape)

    model = build_model(input_dim=X.shape[1], s_segments=S_SEGMENTS)
    model.summary()

    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt_path = os.path.join(MODEL_DIR, "best_model.h5")
    cb = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    # Entrenar
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=cb,
        verbose=2
    )

    # Evaluar
    preds = model.predict(X_test, batch_size=512)
    y_pred = np.argmax(preds, axis=1)
    acc = accuracy_score(y_test, y_pred)
    top3 = top_k_accuracy_score(y_test, preds, k=3)
    print(f"Accuracy test: {acc:.4f}  Top-3: {top3:.4f}")

    # Guardar modelo en formato nativo Keras
    saved_path = os.path.join(MODEL_DIR, "modelo.keras")
    model.save(saved_path, include_optimizer=False)
    print("Modelo guardado en:", saved_path)

    return model, sb

# -----------------------
# Inferencia con modelo
# -----------------------
def query_with_model(query_text: str, model: tf.keras.Model, sb: SegmentedBitmap, top_k=5):
    """Prioriza segmentos con IA y verifica posición exacta en bitmap."""
    hash_feat = hash_bits_vector(query_text, n_bits=HASH_BITS).reshape(1, -1)
    seg_stats = segment_density_features(sb, reduce_to=8).reshape(1, -1)
    Xq = np.concatenate([hash_feat, seg_stats], axis=1).astype(np.float32)

    probs = model.predict(Xq, verbose=0)[0]
    segs_order = np.argsort(-probs)

    nbits = sb.size
    seg_size = sb.segment_size

    for i in range(min(top_k, len(segs_order))):
        seg = segs_order[i]
        if sb.segments[seg].sum() == 0:
            continue
        pos = stable_hash_int(query_text) % nbits
        if seg * seg_size <= pos < (seg + 1) * seg_size:
            idx_local = pos - seg * seg_size
            if sb.segments[seg][idx_local] == 1:
                return True, pos, probs[segs_order[:top_k]]

    pos = stable_hash_int(query_text) % nbits
    seg = pos // seg_size
    found = bool(sb.segments[seg][pos - seg * seg_size] == 1)
    return found, pos, probs[segs_order[:top_k]]

# -----------------------
# Ejecutar todo
# -----------------------
if __name__ == "__main__":
    model, sb = train_and_evaluate()

    tests = ["q_10", "q_99999", "q_1234567", "q_500"]
    for t in tests:
        found, pos, probs = query_with_model(t, model, sb, top_k=5)
        print(f"Query={t} -> found={found} pos={pos}  top_probs={probs[:5]}")
    print("Listo.")
