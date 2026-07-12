import unittest

from atomic_tofu.bridge_r_math import centered_uniform_tail


class BridgeRMathTests(unittest.TestCase):
    def test_centered_tail_is_shift_invariant(self):
        base = centered_uniform_tail([0.0, 1.0, 3.0], 0.5)
        shifted = centered_uniform_tail([10.0, 11.0, 13.0], 0.5)
        self.assertAlmostEqual(base, shifted)

    def test_uniform_values_have_zero_centered_tail(self):
        self.assertAlmostEqual(centered_uniform_tail([2.0, 2.0, 2.0], 1.0), 0.0)

    def test_nonuniform_tail_is_positive(self):
        self.assertGreater(centered_uniform_tail([0.0, 0.0, 2.0], 1.0), 0.0)


if __name__ == "__main__":
    unittest.main()

