import numpy as np
import hashlib
import zlib
class SegmentedBitmap:
    """
    Bitmap segmentado: divide el mapa principal en sub-mapas más pequeños,
    mejorando la búsqueda localizada y facilitando análisis por IA.
    """

    def __init__(self, size, num_segments=4, stable_hash=True):
        self.size = size
        self.num_segments = num_segments
        self.segment_size = size // num_segments
        self.stable_hash = stable_hash
        self.segments = [np.zeros(self.segment_size, dtype=np.uint8) for _ in range(num_segments)]

    def _hash(self, value):
        if self.stable_hash:
            return int(hashlib.sha256(str(value).encode()).hexdigest(), 16) % self.size
        else:
            return hash(value) % self.size

    def _segment_index(self, value):
        h = self._hash(value)
        seg = h // self.segment_size
        idx = h % self.segment_size
        return seg, idx

    def insert(self, value):
        seg, idx = self._segment_index(value)
        self.segments[seg][idx] = 1

    def delete(self, value):
        seg, idx = self._segment_index(value)
        self.segments[seg][idx] = 0

    def search(self, value):
        seg, idx = self._segment_index(value)
        return self.segments[seg][idx] == 1

    def to_array(self):
        """Convierte los segmentos en un único array concatenado"""
        return np.concatenate(self.segments)

    def segment_popcounts(self):
        """Cuenta los bits activos (1s) por segmento."""
        return np.array([seg.sum() for seg in self.segments])

    def to_bytes_for_segments(self, segs):
        """Serializa segmentos en formato comprimido para sincronización P2P."""
        return {i: zlib.compress(self.segments[i].tobytes()) for i in segs}

    def update_from_serialized_segments(self, data):
        """Deserializa y actualiza los segmentos a partir de datos comprimidos."""
        for i, blob in data.items():
            try:
                arr = np.frombuffer(zlib.decompress(blob), dtype=np.uint8)
                self.segments[i][:len(arr)] |= arr
            except Exception as e:
                print(f"Error al actualizar segmento {i}: {e}")