import numpy as np
import multiprocessing

# ==============================
# Workers globales para Windows
# ==============================
def _join_worker(start, end, bitmaps, queue):
    result = np.ones(end-start, dtype=np.uint8)
    for bmp in bitmaps:
        result &= bmp.map[start:end]
    queue.put(result)

def _search_worker(map_array, start, end, value, queue):
    idx = hash(value) % len(map_array)
    found = False
    if start <= idx < end:
        found = map_array[idx] == 1
    queue.put(found)

# ==============================
# Clase principal
# ==============================
class DynamicParallelBitmap:
    """
    Dynamic Parallel Bitmap optimizado con NumPy y multiprocessing.
    """

    def __init__(self, size, num_processes=4):
        self.size = size
        self.num_processes = num_processes
        self.map = np.zeros(size, dtype=np.uint8)

    def insert(self, value):
        self.map[hash(value) % self.size] = 1

    def delete(self, value):
        self.map[hash(value) % self.size] = 0

    def parallel_search(self, value):
        segment_size = self.size // self.num_processes
        segments = [(i*segment_size, (i+1)*segment_size if i != self.num_processes-1 else self.size)
                    for i in range(self.num_processes)]

        queue = multiprocessing.Queue()
        processes = []

        for start, end in segments:
            p = multiprocessing.Process(target=_search_worker, args=(self.map, start, end, value, queue))
            processes.append(p)
            p.start()

        results = [queue.get() for _ in processes]
        for p in processes:
            p.join()

        return any(results)

    @staticmethod
    def parallel_join(bitmaps, num_processes=4):
        size = bitmaps[0].size
        segment_size = size // num_processes
        segments = [(i*segment_size, (i+1)*segment_size if i != num_processes-1 else size)
                    for i in range(num_processes)]

        queue = multiprocessing.Queue()
        processes = []

        for start, end in segments:
            p = multiprocessing.Process(target=_join_worker, args=(start, end, bitmaps, queue))
            processes.append(p)
            p.start()

        final_result = np.concatenate([queue.get() for _ in processes])
        for p in processes:
            p.join()

        return np.flatnonzero(final_result).tolist()
