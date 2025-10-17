from .bitmap import DynamicParallelBitmap
from .segmented_bitmap import SegmentedBitmap
from . import utils
from . import ai

# Integrar módulo P2P si está disponible
try:
    from .p2p.dpb_net import DPBNetNode, RandomPredictor
    from .p2p.dpb_net_q import DPBNetQNode
except ImportError:
    DPBNetNode = None
    RandomPredictor = None
    DPBNetQNode = None

__all__ = [
    "DynamicParallelBitmap",
    "SegmentedBitmap",
    "utils",
    "ai",
    "DPBNetNode",
    "RandomPredictor",
    "DPBNetQNode",
]
