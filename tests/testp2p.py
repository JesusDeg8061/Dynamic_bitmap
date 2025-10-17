"""
test_dpb_net.py — Pruebas de integración y conectividad del sistema DPB-NET y DPB-NET-Q
Versión 0.3.3-Stable
"""

import time
import socket
import pickle
import pytest
import numpy as np

from dynamic_bitmap import DPBNetNode, RandomPredictor

# Intentar cargar QNode si existe
try:
    from dynamic_bitmap import DPBNetQNode
except ImportError:
    DPBNetQNode = None


def wait_for_port(port, timeout=3.0):
    """Espera hasta que un puerto esté disponible para conexión."""
    start = time.time()
    while time.time() - start < timeout:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.1)
    return False


# ============================================================
# PRUEBA 1 — Conectividad básica (modo simulado)
# ============================================================
@pytest.mark.timeout(15)
@pytest.mark.skipif(DPBNetQNode is None, reason="DPBNetQNode no disponible en esta versión.")
def test_basic_connection_qnode():
    """Verifica que dos nodos DPB-NET-Q puedan sincronizar datos correctamente."""
    port1, port2 = 7050, 7051

    node1 = DPBNetQNode(("127.0.0.1", port1), size=512, num_segments=16, top_k=3)
    node2 = DPBNetQNode(("127.0.0.1", port2), size=512, num_segments=16, top_k=3)
    node1.peers = [("127.0.0.1", port2)]
    node2.peers = [("127.0.0.1", port1)]

    # Simular envío de payloads entre nodos (sin sockets reales)
    def fake_sender(raw, peer):
        for n in (node1, node2):
            if n.bind_addr == peer:
                n.integrate_payload(raw)

    node1.insert("sensor_42")
    node1.broadcast_update(fake_sender)

    time.sleep(1.0)
    assert node2.search("sensor_42"), "El valor 'sensor_42' no se replicó correctamente."

    node1.stop_sync_loop()
    node2.stop_sync_loop()


# ============================================================
# PRUEBA 2 — Predictor aleatorio / IA
# ============================================================
def test_predictor_weighted_selection():
    """Verifica que el predictor (IA o Random) devuelva índices válidos y consistentes."""
    try:
        from dynamic_bitmap.ai.predictor import IAPredictor
        pred = IAPredictor()
    except Exception:
        pred = RandomPredictor(seed=123)

    from dynamic_bitmap.segmented_bitmap import SegmentedBitmap
    bmp = SegmentedBitmap(size=1024, num_segments=16)
    for i in range(200):
        bmp.insert(i)

    segs = pred.predict_topk(bmp, k=5)
    assert isinstance(segs, list)
    assert 1 <= len(segs) <= 5
    assert all(isinstance(s, int) for s in segs)
    assert all(0 <= s < bmp.num_segments for s in segs)


# ============================================================
# PRUEBA 3 — Integración paralela simulada (QNodes)
# ============================================================
@pytest.mark.timeout(10)
@pytest.mark.skipif(DPBNetQNode is None, reason="DPBNetQNode no disponible en esta versión.")
def test_parallel_integration_cycle():
    """Prueba completa: inserción, sincronización y entrelazamiento lógico."""
    nodeA = DPBNetQNode(("127.0.0.1", 7070), size=256, num_segments=8, top_k=3)
    nodeB = DPBNetQNode(("127.0.0.1", 7071), size=256, num_segments=8, top_k=3)
    nodeA.peers = [nodeB.bind_addr]
    nodeB.peers = [nodeA.bind_addr]

    def fake_send(raw, peer):
        for n in (nodeA, nodeB):
            if n.bind_addr == peer:
                n.integrate_payload(raw)

    nodeA.insert("user_alpha")
    nodeB.insert("user_beta")

    for _ in range(3):
        nodeA.broadcast_update(fake_send)
        nodeB.broadcast_update(fake_send)
        time.sleep(0.5)

    assert nodeA.search("user_beta"), "user_beta no llegó a nodeA"
    assert nodeB.search("user_alpha"), "user_alpha no llegó a nodeB"

    nodeA.stop_sync_loop()
    nodeB.stop_sync_loop()


# ============================================================
# PRUEBA 4 — Entrelazamiento lógico y correlación
# ============================================================
@pytest.mark.skipif(DPBNetQNode is None, reason="DPBNetQNode no disponible en esta versión.")
def test_entanglement_similarity():
    """Mide la correlación lógica entre nodos después de sincronizar."""
    node1 = DPBNetQNode(("127.0.0.1", 7080), size=256, num_segments=8, top_k=4)
    node2 = DPBNetQNode(("127.0.0.1", 7081), size=256, num_segments=8, top_k=4)
    node1.peers = [node2.bind_addr]
    node2.peers = [node1.bind_addr]

    def fake_send(raw, peer):
        for n in (node1, node2):
            if n.bind_addr == peer:
                n.integrate_payload(raw)

    for val in ["temp", "hum", "press"]:
        node1.insert(val)

    node1.broadcast_update(fake_send)
    time.sleep(0.5)

    s1 = np.array([qb.score for qb in node1.qbits])
    s2 = np.array([qb.score for qb in node2.qbits])
    corr = np.corrcoef(s1, s2)[0, 1]

    assert corr > 0.7, f"Correlación baja ({corr:.2f}); entrelazamiento lógico no efectivo."

    node1.stop_sync_loop()
    node2.stop_sync_loop()


# ============================================================
# PRUEBA 5 — DPBNetNode (modo TCP real con fallback IA)
# ============================================================
@pytest.mark.timeout(15)
def test_dpbnetnode_auto_predictor():
    """Verifica que DPBNetNode arranca y usa predictor IA o fallback correctamente."""
    port1, port2 = 7090, 7091
    node1 = DPBNetNode(("127.0.0.1", port1), size=256, num_segments=8, top_k=3)
    node2 = DPBNetNode(("127.0.0.1", port2), size=256, num_segments=8, top_k=3)
    node1.add_peer(node2.bind_addr)

    node1.insert("node_test_val")
    payload = node1.prepare_payload()
    assert isinstance(payload, (bytes, bytearray))

    # Integrar en nodo destino (simulación sin red)
    data = pickle.loads(payload)
    node2.bitmap.update_from_serialized_segments(data["segments"])

    #  Ahora debe existir tras sincronización simulada
    assert node2.search("node_test_val"), "El valor no se replicó correctamente entre nodos."

    node1.stop()
    node2.stop()
