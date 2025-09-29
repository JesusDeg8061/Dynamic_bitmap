import unittest
import numpy as np
from dynamic_bitmap.bitmap import DynamicParallelBitmap
from dynamic_bitmap.segmented_bitmap import SegmentedBitmap
from dynamic_bitmap.ai.predictor import train_model, query_with_model

class TestIntegration(unittest.TestCase):

    def test_dynamic_bitmap(self):
        bmp = DynamicParallelBitmap(size=1000)
        bmp.insert(10)
        bmp.insert(20)
        bmp.insert(30)

        self.assertTrue(bmp.parallel_search(10))
        self.assertFalse(bmp.parallel_search(999))

        bmp.delete(20)
        self.assertFalse(bmp.parallel_search(20))

    def test_segmented_bitmap(self):
        seg_bmp = SegmentedBitmap(size=1000, num_segments=10)

        seg_bmp.insert(15)
        seg_bmp.insert(250)
        seg_bmp.insert(999)

        self.assertTrue(seg_bmp.search(15))
        self.assertTrue(seg_bmp.search(250))
        self.assertFalse(seg_bmp.search(1234))

    def test_ai_prediction(self):
        seg_bmp = SegmentedBitmap(size=800, num_segments=8)

        # Entrenamiento desde bitmap (no pasar X e y)
        model = train_model(bitmap=seg_bmp, n_samples=200, epochs=5)

        # Predicción con la IA
        sample_val = "q_42"
        found, pos, probs = query_with_model(sample_val, model, seg_bmp, top_k=3)

        # Verificar que devuelve booleano
        self.assertIsInstance(found, bool)
        self.assertIsInstance(pos, int)
        self.assertEqual(probs.shape[0], 3)

if __name__ == "__main__":
    unittest.main()
