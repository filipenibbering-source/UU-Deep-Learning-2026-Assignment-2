import unittest

import torch

from meg_decode.models import EEGNetClassifier, EEGNetTCNClassifier


class ModelTests(unittest.TestCase):
    def test_eegnet_forward(self):
        model = EEGNetClassifier(
            n_classes=4,
            n_channels=8,
            temporal_filters=4,
            depth_multiplier=2,
            separable_filters=8,
            temporal_kernel=15,
            separable_kernel=7,
            pool1=2,
            pool2=2,
            dropout=0.1,
        )
        logits = model(torch.randn(3, 1, 8, 64))
        self.assertEqual(tuple(logits.shape), (3, 4))

    def test_tcn_forward(self):
        model = EEGNetTCNClassifier(
            n_classes=4,
            n_channels=8,
            temporal_filters=4,
            depth_multiplier=2,
            separable_filters=8,
            temporal_kernel=15,
            separable_kernel=7,
            pool1=2,
            pool2=2,
            dropout=0.1,
            tcn_dilations=[1, 2],
        )
        logits = model(torch.randn(3, 1, 8, 64))
        self.assertEqual(tuple(logits.shape), (3, 4))


if __name__ == "__main__":
    unittest.main()

