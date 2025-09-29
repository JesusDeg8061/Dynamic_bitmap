import numpy as np
from dynamic_bitmap.ai.features import stable_hash_int, hash_bits_vector, segment_density_features

def generate_synthetic_dataset(bitmap, n_samples=20000, hash_bits=64, reduce_to=8, hash_space_factor=5):
    """
    Genera dataset sintético: features (X), etiquetas (y) y queries originales.
    """
    S = bitmap.num_segments
    n_bits = bitmap.size

    X_hash_bits = np.zeros((n_samples, hash_bits), dtype=np.float32)
    X_seg_stats = np.zeros((n_samples, reduce_to), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.int32)
    queries = []

    for i in range(n_samples):
        val = np.random.randint(0, n_bits * hash_space_factor)
        q = f"q_{val}"
        queries.append(q)

        h = stable_hash_int(q)
        pos = h % n_bits
        seg = min(pos // bitmap.segment_size, S - 1)

        y[i] = int(seg)
        X_hash_bits[i, :] = hash_bits_vector(q, n_bits=hash_bits)
        X_seg_stats[i, :] = segment_density_features(bitmap, reduce_to=reduce_to)

    X = np.concatenate([X_hash_bits, X_seg_stats], axis=1)
    return X, y, queries
