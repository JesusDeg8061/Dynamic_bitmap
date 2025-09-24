from dynamic_bitmap import DynamicParallelBitmap
import random

if __name__ == "__main__":
    N = 10**6
    NUM_TABLAS = 3

    bitmaps = [DynamicParallelBitmap(size=N, num_processes=8) for _ in range(NUM_TABLAS)]

    for i in range(0, N, 10):
        for bmp in bitmaps:
            if random.random() < 0.3:
                bmp.insert(i)

    test_value = 12345
    print(f"¿Existe {test_value}? -> {bitmaps[0].parallel_search(test_value)}")

    resultado_join = DynamicParallelBitmap.parallel_join(bitmaps, num_processes=8)
    print(f"Coincidencias en JOIN: {len(resultado_join)}")
