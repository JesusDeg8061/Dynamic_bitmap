"""
dpb_net.py — DPB-NET (Dynamic Parallel Bitmap Network)
Extensión P2P descentralizada del Dynamic Parallel Bitmap.

Versión: 0.3.3-Stable
Autor: Jesús Alberto Degollado López
Descripción:
    Permite la conexión de nodos que funcionan como peers, compartiendo segmentos
    de bitmaps mediante sincronización incremental, predicción IA y compresión.

Características:
    - Comunicación TCP peer-to-peer
    - Compresión zlib y merge idempotente
    - Predictor configurable (IA o heurístico)
    - Sincronización periódica y top-K segment sharing
"""

from __future__ import annotations
import socket
import threading
import pickle
import time
import logging
import zlib
import hashlib
from typing import Dict, List, Tuple, Optional, Protocol
import numpy as np
from dynamic_bitmap.segmented_bitmap import SegmentedBitmap

# =============================
# Logging
# =============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dpb_net")

# =============================
# Predictor Interface
# =============================
class SegmentPredictor(Protocol):
    def predict_topk(self, bitmap: "SegmentedBitmap", top_k: int) -> List[int]:
        ...


# =============================
# Predictors
# =============================
class RandomPredictor:
    """Predictor aleatorio o ponderado por densidad.
    Acepta `predict_topk(bitmap, k=5)` o `predict_topk(bitmap, top_k=5)` (compatibilidad total).
    """
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def predict_topk(self, bitmap: SegmentedBitmap, k=None, top_k=None, **kwargs) -> List[int]:
        if k is None:
            k = top_k if top_k is not None else kwargs.get("k", 3)
        num_segments = getattr(bitmap, "num_segments", 1)
        k = min(int(k), num_segments)

        pops = bitmap.segment_popcounts()
        weights = pops + 1e-8
        total = weights.sum()
        probs = (weights / total) if total > 0 else None

        segs = self.rng.choice(num_segments, size=k, replace=False, p=probs if probs is not None else None)
        return list(map(int, segs))


# =============================
# DPBNet Node
# =============================
class DPBNetNode:
    """Nodo DPB-NET — sincronización P2P de bitmaps dinámicos con IA opcional."""
    PROTOCOL_VERSION = 1

    def __init__(self, bind_addr: Tuple[str, int],
                 size: int = 1024,
                 num_segments: int = 64,
                 predictor: Optional[SegmentPredictor] = None,
                 top_k: int = 4,
                 sync_interval: float = 5.0):
        self.bind_addr = bind_addr
        self.bitmap = SegmentedBitmap(size=size, num_segments=num_segments)
        self.predictor = predictor
        self.top_k = top_k
        self.sync_interval = sync_interval
        self.peers: List[Tuple[str, int]] = []
        self._stop = threading.Event()

        # Asignar predictor IA automáticamente (fallback a RandomPredictor)
        self._auto_assign_predictor()

    # ============================
    # Predictor automático
    # ============================
    def _auto_assign_predictor(self):
        """Usa el predictor IA si está disponible; si no, recurre a RandomPredictor."""
        if self.predictor is not None:
            return
        try:
            from dynamic_bitmap.ai.predictor import IAPredictor
            self.predictor = IAPredictor()
            logger.info("DPBNetNode usando IAPredictor (modelo IA).")
        except Exception as e:
            logger.warning(f"No se pudo usar IAPredictor, fallback a RandomPredictor: {e}")
            self.predictor = RandomPredictor()

    # ============================
    # Control de nodo
    # ============================
    def start(self):
        threading.Thread(target=self._server_loop, daemon=True).start()
        threading.Thread(target=self._sync_loop, daemon=True).start()
        logger.info("Nodo iniciado en %s", self.bind_addr)

    def stop(self):
        self._stop.set()
        logger.info("Nodo detenido en %s", self.bind_addr)

    def add_peer(self, addr: Tuple[str, int]):
        if addr not in self.peers and addr != self.bind_addr:
            self.peers.append(addr)
            logger.info("Peer agregado: %s", addr)

    # ============================
    # Server loop
    # ============================
    def _server_loop(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(self.bind_addr)
        s.listen(8)
        s.settimeout(1.0)

        while not self._stop.is_set():
            try:
                conn, addr = s.accept()
                threading.Thread(target=self._handle_conn, args=(conn, addr), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                logger.warning("Error en server loop: %s", e)
        s.close()

    def _handle_conn(self, conn: socket.socket, addr):
        data = b""
        try:
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
        finally:
            conn.close()

        try:
            payload = pickle.loads(data)
            if payload.get("v") == self.PROTOCOL_VERSION:
                self.bitmap.update_from_serialized_segments(payload["segments"])
                logger.info("Actualizados %d segmentos desde %s", len(payload["segments"]), addr)
        except Exception as e:
            logger.warning("Error recibiendo de %s: %s", addr, e)

    # ============================
    # Sincronización / Broadcast
    # ============================
    def _sync_loop(self):
        while not self._stop.is_set():
            time.sleep(self.sync_interval)
            if self.peers:
                self.broadcast_update()

    def prepare_payload(self) -> bytes:
        segs = self.predictor.predict_topk(self.bitmap, top_k=self.top_k)
        payload = {
            "v": self.PROTOCOL_VERSION,
            "segments": self.bitmap.to_bytes_for_segments(segs),
            "ts": time.time()
        }
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    def broadcast_update(self):
        """Envía actualización comprimida a todos los peers."""
        raw = self.prepare_payload()
        for peer in list(self.peers):
            try:
                with socket.create_connection(peer, timeout=3) as s:
                    s.sendall(raw)
                logger.info("Sync enviado a %s", peer)
            except Exception as e:
                logger.warning("Fallo al enviar a %s: %s", peer, e)

    # ============================
    # Operaciones utilitarias
    # ============================
    def insert(self, value):
        self.bitmap.insert(value)

    def search(self, value) -> bool:
        return self.bitmap.search(value)
