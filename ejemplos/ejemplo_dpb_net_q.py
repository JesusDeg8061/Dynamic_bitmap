from dynamic_bitmap.p2p.dpb_net_q import DPBNetQNode
import numpy as np
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # opcional, elimina advertencias de TensorFlow

# Crear dos nodos
nodeA = DPBNetQNode(("127.0.0.1", 7070), size=256, num_segments=8, top_k=3)
nodeB = DPBNetQNode(("127.0.0.1", 7071), size=256, num_segments=8, top_k=3)

# Conectarlos virtualmente
nodeA.peers = [nodeB.bind_addr]
nodeB.peers = [nodeA.bind_addr]

# Función simulada de red
def fake_send(raw, peer):
    for n in (nodeA, nodeB):
        if n.bind_addr == peer:
            n.integrate_payload(raw)

# Insertar dato en nodeA
nodeA.insert("sensor_42")
nodeA.broadcast_update(fake_send)

# Verificar replicación
print("¿Sensor replicado?", nodeB.search("sensor_42"))  # ✅ True

# Calcular correlación segura
s1 = np.array([qb.score for qb in nodeA.qbits])
s2 = np.array([qb.score for qb in nodeB.qbits])
if np.std(s1) == 0 or np.std(s2) == 0:
    corr = 1.0
else:
    corr = np.corrcoef(s1, s2)[0, 1]
print("Correlación entre nodos:", round(corr, 3))
