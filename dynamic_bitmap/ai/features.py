import numpy as np
import hashlib
from dynamic_bitmap.segmented_bitmap import SegmentedBitmap

def stable_hash_int(x: str) -> int:
    """Convierte string a entero usando SHA256."""
    return int(hashlib.sha256(str(x).encode()).hexdigest(), 16)

def hash_bits_vector(query_text: str, n_bits=64):
    """Convierte hash SHA256 en vector binario de n_bits (LSB..MSB)."""
    h = stable_hash_int(query_text)
    return np.array([(h >> i) & 1 for i in range(n_bits)], dtype=np.float32)

def segment_density_features(bitmap: SegmentedBitmap, reduce_to=8):
    """
    Calcula densidad por segmento y lo reduce a un vector de longitud reduce_to.
    Incluye mean, var y top-k densidades.
    """
    popcounts = np.array([int(seg.sum()) for seg in bitmap.segments], dtype=np.float32)
    densities = popcounts / (bitmap.segment_size + 1e-9)
    mean, var = densities.mean(), densities.var()
    topk = np.sort(densities)[-(reduce_to - 2):] if reduce_to > 2 else []
    vec = np.concatenate(([mean, var],
                          np.pad(topk, (0, max(0, (reduce_to - 2) - len(topk))), 'constant')))
    return vec.astype(np.float32)
