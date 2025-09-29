import numpy as np
import multiprocessing
from multiprocessing import shared_memory
import hashlib

# ==============================
# Workers
# ==============================
def _join_worker(start, end, shm_name, size, dtype, queue, idx, num_bitmaps):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray((num_bitmaps, size), dtype=dtype, buffer=shm.buf)
    result = np.bitwise_and.reduce(arr[:, start:end], axis=0)
    shm.close()
    queue.put((idx, result))

def _search_worker(start, end, shm_name, size, dtype, value, queue, idx):
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(size, dtype=dtype, buffer=shm.buf)
    h = int(hashlib.sha256(str(value).encode()).hexdigest(), 16)
    idx_val = h % size
    found = start <= idx_val < end and arr[idx_val] == 1
    shm.close()
    queue.put((idx, found))

# ==============================
# Clase principal
# ==============================
class DynamicParallelBitmap:
    """
    Dynamic Parallel Bitmap optimizado con NumPy y multiprocessing.shared_memory.
    """

    def __init__(self, size, num_processes=4, stable_hash=True):
        self.size = size
        self.num_processes = num_processes
        self.map = np.zeros(size, dtype=np.uint8)
        self.stable_hash = stable_hash

    def _hash(self, value):
        if self.stable_hash:
            return int(hashlib.sha256(str(value).encode()).hexdigest(), 16) % self.size
        else:
            return hash(value) % self.size

    def insert(self, value):
        self.map[self._hash(value)] = 1

    def delete(self, value):
        self.map[self._hash(value)] = 0

    def parallel_search(self, value):
        segment_size = self.size // self.num_processes
        segments = [(i * segment_size,
                     (i + 1) * segment_size if i != self.num_processes - 1 else self.size, i)
                    for i in range(self.num_processes)]

        shm = shared_memory.SharedMemory(create=True, size=self.map.nbytes)
        shm_arr = np.ndarray(self.map.shape, dtype=self.map.dtype, buffer=shm.buf)
        np.copyto(shm_arr, self.map)

        queue = multiprocessing.Queue()
        processes = []

        for start, end, idx in segments:
            p = multiprocessing.Process(target=_search_worker,
                                        args=(start, end, shm.name, self.size, self.map.dtype, value, queue, idx))
            processes.append(p)
            p.start()

        results = [queue.get() for _ in processes]
        for p in processes:
            p.join()

        shm.close()
        shm.unlink()

        return any(found for _, found in results)

    @staticmethod
    def parallel_join(bitmaps, num_processes=4):
        size = bitmaps[0].size
        dtype = bitmaps[0].map.dtype
        num_bitmaps = len(bitmaps)
        arr = np.stack([bmp.map for bmp in bitmaps])

        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        shm_arr = np.ndarray(arr.shape, dtype=dtype, buffer=shm.buf)
        np.copyto(shm_arr, arr)

        segment_size = size // num_processes
        segments = [(i * segment_size,
                     (i + 1) * segment_size if i != num_processes - 1 else size, i)
                    for i in range(num_processes)]

        queue = multiprocessing.Queue()
        processes = []

        for start, end, idx in segments:
            p = multiprocessing.Process(target=_join_worker,
                                        args=(start, end, shm.name, size, dtype, queue, idx, num_bitmaps))
            processes.append(p)
            p.start()

        results = sorted([queue.get() for _ in processes], key=lambda x: x[0])
        final_result = np.concatenate([r for _, r in results])

        for p in processes:
            p.join()

        shm.close()
        shm.unlink()

        return np.flatnonzero(final_result).tolist()
