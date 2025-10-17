"""
dynamic_bitmap.p2p
===================
Módulo de conexión punto a punto (P2P) para Dynamic Parallel Bitmap.

Contiene:
    - DPBNetNode: nodo principal de red que maneja sincronización entre peers.
    - DPBNetQNode: versión experimental con colas de mensajes y predicción IA.
    - RandomPredictor: predictor simple para priorizar segmentos.
"""

from .dpb_net import DPBNetNode, RandomPredictor

# Soporte alternativo (versión cuántica o extendida)
try:
    from .dpb_net_q import DPBNetQNode
except ImportError:
    DPBNetQNode = None

__all__ = ["DPBNetNode", "RandomPredictor", "DPBNetQNode"]
