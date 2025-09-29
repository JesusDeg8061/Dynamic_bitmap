import unittest
from dynamic_bitmap import DynamicParallelBitmap

class TestDynamicParallelBitmap(unittest.TestCase):
    def test_insert_search_delete(self):
        bmp = DynamicParallelBitmap(size=1000, num_processes=4)
        bmp.insert(10)
        self.assertTrue(bmp.parallel_search(10))
        bmp.delete(10)
        self.assertFalse(bmp.parallel_search(10))

    def test_parallel_join(self):
        bmp1 = DynamicParallelBitmap(size=100, num_processes=2)
        bmp2 = DynamicParallelBitmap(size=100, num_processes=2)
        bmp1.insert(5)
        bmp2.insert(5)

        result = DynamicParallelBitmap.parallel_join([bmp1, bmp2], num_processes=2)

        # calcular la posición real que le corresponde a "5"
        hashed = hash(5) % 100
        self.assertIn(hashed, result)

if __name__ == '__main__':
    unittest.main()
