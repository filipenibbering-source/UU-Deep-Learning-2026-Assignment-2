import unittest

import numpy as np

from meg_decode.preprocessing import (
    compute_window_starts,
    extract_window,
    preprocess_recording,
    window_geometry,
)


class PreprocessingTests(unittest.TestCase):
    def test_window_geometry(self):
        window_size, stride_size, effective_rate = window_geometry(
            sample_rate=2034.0,
            downsample_factor=8,
            window_seconds=2.0,
            stride_seconds=1.0,
        )
        self.assertEqual(window_size, 508)
        self.assertEqual(stride_size, 254)
        self.assertAlmostEqual(effective_rate, 254.25)

    def test_preprocess_shape_and_dtype(self):
        x = np.random.default_rng(0).normal(size=(4, 64))
        y = preprocess_recording(x, downsample_factor=2, normalization="zscore", clip_value=6.0)
        self.assertEqual(y.shape, (4, 32))
        self.assertEqual(y.dtype, np.float32)

    def test_extract_window_pads_short_tail(self):
        x = np.ones((2, 5), dtype=np.float32)
        window = extract_window(x, start=3, window_size=4)
        self.assertEqual(window.shape, (2, 4))
        self.assertTrue(np.all(window[:, :2] == 1))
        self.assertTrue(np.all(window[:, 2:] == 0))

    def test_compute_window_starts(self):
        self.assertEqual(compute_window_starts(10, 4, 3), [0, 3, 6])


if __name__ == "__main__":
    unittest.main()

