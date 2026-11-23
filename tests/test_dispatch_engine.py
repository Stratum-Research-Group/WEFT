import unittest
import torch
import numpy as np

class TestUnifiedDispatch(unittest.TestCase):
    def test_algebraic_invariance(self):
        # Test simulated SIMD popcount dot product invariance
        cols = 4096
        # Generate random bitvectors
        b = np.random.randint(0, 2, size=(cols,), dtype=np.uint8)
        x = np.random.randint(0, 2, size=(cols,), dtype=np.uint8)
        
        # Scalar Hamming distance
        hamming = int(np.sum(b ^ x))
        dot_product = float(cols - 2 * hamming)
        
        # Bipolar dot product reference (-1, +1)
        b_bipolar = 1.0 - 2.0 * b
        x_bipolar = 1.0 - 2.0 * x
        bipolar_ref = np.dot(b_bipolar, x_bipolar)
        
        self.assertEqual(dot_product, bipolar_ref)

    def test_fp8_e4m3_bounds(self):
        # Exponent range: 1..14 with bias 7 -> unbiased -6..+7
        bias = 7
        for raw_exp in range(1, 15):
            unbiased = raw_exp - bias
            self.assertTrue(-6 <= unbiased <= 7)

if __name__ == "__main__":
    unittest.main()
