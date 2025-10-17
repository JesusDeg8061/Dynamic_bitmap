"""
dpb_net_q.py — DPB-NET-Q (Extensión lógica 'qbit')
Versión 0.3.4-qStable — Jesús A. Degollado López
"""

from __future__ import annotations
import time
import threading
import pickle
import zlib
import hashlib
import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

from ..segmented_bitmap import SegmentedBitmap

logger = logging.getLogger("dpb_net_q")

# ===========================================================
# Qbit lógico (simulación)
# ===========================================================
class LogicalQBit:
    """Representa un 'qbit' lógico con score (valor) y conf (confianza)."""
    __slots__ = ("score", "conf")

    def __init__(self, score: float = 0.0, conf: float = 0.0):
        self.score = float(np.clip(score, 0.0, 1.0))
        self.conf = float(np.clip(conf, 0.0, 1.0))

    def measure(self, threshold: float = 0.5) -> bool:
        return self.score >= threshold

    def update(self, delta_score: float, delta_conf: float):
        self.score = float(np.clip(self.score + delta_score, 0.0, 1.0))
        self.conf = float(np.clip(self.conf + delta_conf, 0.0, 1.0))


# ===========================================================
# Nodo DPB-NET-Q
# ===========================================================
class DPBNetQNode:
    """Nodo de red cuántico lógico que sincroniza segmentos y estados entre nodos."""
    PROTO_V = 1

    def __init__(self, bind_addr: Tuple[str, int],
                 size: int = 1024, num_segments: int = 64,
                 predictor=None, top_k: int = 4,
                 sync_interval: float = 5.0):
        self.bind_addr = bind_addr
        self.size = size
        self.num_segments = num_segments
        self.segment_size = size // num_segments
        self.segments = [np.zeros(self.segment_size, dtype=bool) for _ in range(num_segments)]
        self.qbits = [LogicalQBit(score=0.01, conf=0.01) for _ in range(num_segments)]
        self.predictor = predictor
        self.top_k = top_k
        self.peers: List[Tuple[str, int]] = []
        self.sync_interval = sync_interval
        self._stop_flag = False
        self._last_touched_segment: Optional[int] = None  # 👈 nuevo

        self._auto_assign_predictor()

    def _hash(self, value) -> int:
        return int(hashlib.sha256(str(value).encode()).hexdigest(), 16) % self.size

    def _segment_index(self, value) -> Tuple[int, int]:
        h = self._hash(value)
        return h // self.segment_size, h % self.segment_size

    def insert(self, value):
        seg, idx = self._segment_index(value)
        self.segments[seg][idx] = True
        self.qbits[seg].update(0.25, 0.2)
        self._last_touched_segment = seg  # 👈 guardamos segmento activo

    def delete(self, value):
        seg, idx = self._segment_index(value)
        self.segments[seg][idx] = False
        self.qbits[seg].update(-0.05, -0.03)
        self._last_touched_segment = seg

    def search(self, value) -> bool:
        seg, idx = self._segment_index(value)
        return bool(self.segments[seg][idx])

    def predict_topk(self, top_k: Optional[int] = None) -> List[int]:
        k = top_k or self.top_k
        if self.predictor is not None:
            try:
                return self.predictor.predict_topk(self, k=k)
            except Exception as e:
                logger.warning("Predictor falló: %s", e)
        scores = np.array([qb.score * qb.conf for qb in self.qbits])
        return np.argsort(-scores)[:k].tolist()

    def serialize_segments(self, segs: List[int]) -> Dict[int, bytes]:
        return {i: zlib.compress(self.segments[i].tobytes()) for i in segs}

    def update_from_serialized(self, data: Dict[int, bytes], entangle: bool = True):
        recv_segments = set()
        for i, blob in data.items():
            try:
                arr = np.frombuffer(zlib.decompress(blob), dtype=np.bool_)
                minlen = min(len(arr), len(self.segments[i]))
                before = np.copy(self.segments[i])
                self.segments[i][:minlen] = np.logical_or(before[:minlen], arr[:minlen])
                if not np.array_equal(before, self.segments[i]):
                    self.qbits[i].update(0.45, 0.35)
                recv_segments.add(i)
            except Exception as e:
                logger.warning("Error al decodificar segmento %s: %s", i, e)

        if entangle and recv_segments:
            self._entangle_logic(recv_segments)

    def _entangle_logic(self, recv_segments):
        top_local = set(self.predict_topk(self.top_k))
        common = top_local.intersection(recv_segments)
        for i in common:
            self.qbits[i].update(0.9, 0.9)
        for i in common:
            for nb in (i - 1, i + 1):
                if 0 <= nb < self.num_segments:
                    self.qbits[nb].update(0.6, 0.5)
        scores = np.array([qb.score for qb in self.qbits])
        mean_score = float(np.mean(scores))
        for qb in self.qbits:
            qb.update((mean_score - qb.score) * 1.0, 0.3)

    def prepare_payload(self) -> bytes:
        segs = set(self.predict_topk(self.top_k))
        if self._last_touched_segment is not None:
            segs.add(self._last_touched_segment)  # 👈 forzamos inclusión del segmento activo
        payload = {"v": self.PROTO_V, "segments": self.serialize_segments(list(segs)), "ts": time.time()}
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    def integrate_payload(self, raw: bytes):
        try:
            payload = pickle.loads(raw)
            if payload.get("v") == self.PROTO_V:
                self.update_from_serialized(payload.get("segments", {}), entangle=True)
        except Exception as e:
            logger.warning("Payload inválido: %s", e)

    def broadcast_update(self, sender_callable):
        raw = self.prepare_payload()
        for peer in list(self.peers):
            try:
                sender_callable(raw, peer)
            except Exception as e:
                logger.warning(f"Error enviando actualización a {peer}: {e}")

    def start_sync_loop(self, sender_callable):
        def _loop():
            while not self._stop_flag:
                self.broadcast_update(sender_callable)
                time.sleep(self.sync_interval)
        threading.Thread(target=_loop, daemon=True).start()

    def stop_sync_loop(self):
        self._stop_flag = True

    def _auto_assign_predictor(self):
        try:
            from dynamic_bitmap.ai.predictor import IAPredictor
            self.predictor = IAPredictor()
            logger.info("DPBNetQNode usando IAPredictor (modelo IA).")
        except Exception as e:
            logger.warning(f"No se pudo usar IAPredictor, fallback a RandomPredictor: {e}")
            self.predictor = RandomPredictor()


# ===========================================================
# Predictors de compatibilidad
# ===========================================================
class RandomPredictor:
    """Predictor aleatorio compatible con DPB-NET-Q."""
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def predict_topk(self, bitmap_like, k=None, top_k=None, **kwargs):
        if k is None:
            k = top_k if top_k is not None else kwargs.get("k", 3)
        num_segments = getattr(bitmap_like, "num_segments", 1)
        k = min(int(k), num_segments)
        return self.rng.choice(num_segments, size=k, replace=False).tolist()
