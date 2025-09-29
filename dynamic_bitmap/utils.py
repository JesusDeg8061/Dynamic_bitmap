import hashlib

def stable_hash(value, size):
    """
    Devuelve un hash estable usando SHA-256 limitado al tamaño dado.
    """
    return int(hashlib.sha256(str(value).encode()).hexdigest(), 16) % size
