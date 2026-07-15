import sys
import unittest
from unittest import mock

import numpy as np
import torch

from model.v2_debias.network import DebiasNetV2
from model.v2_debias.train import get_args
from utils import eval_xauc


class PublicContractTest(unittest.TestCase):
    def test_xauc_on_strict_orders(self):
        labels = np.array([1.0, 2.0, 3.0])
        self.assertEqual(eval_xauc(labels, labels.copy()), 1.0)
        self.assertEqual(eval_xauc(labels, labels[::-1].copy()), 0.0)

    def test_xauc_ties_receive_zero_credit(self):
        self.assertAlmostEqual(
            eval_xauc(np.array([1.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])),
            2.0 / 3.0,
        )
        self.assertEqual(
            eval_xauc(np.array([1.0, 2.0]), np.array([1.0, 1.0])),
            0.0,
        )
        self.assertEqual(
            eval_xauc(np.array([2.0, 1.0]), np.array([1.0, 1.0])),
            0.0,
        )

    def test_xauc_matches_brute_force_with_ties(self):
        rng = np.random.default_rng(2027)
        for sample_count in range(2, 12):
            for _ in range(50):
                labels = rng.integers(0, 4, size=sample_count).astype(float)
                predictions = rng.integers(0, 4, size=sample_count).astype(float)
                agreements = sum(
                    (predictions[i] - predictions[j]) * (labels[i] - labels[j]) > 0
                    for i in range(sample_count)
                    for j in range(i + 1, sample_count)
                )
                expected = agreements / (sample_count * (sample_count - 1) // 2)
                self.assertEqual(eval_xauc(labels, predictions), expected)

    def test_xauc_validates_input(self):
        with self.assertRaises(ValueError):
            eval_xauc(np.array([1.0]), np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            eval_xauc(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
        self.assertTrue(np.isnan(eval_xauc(np.array([1.0]), np.array([1.0]))))

    def test_manuscript_aligned_defaults(self):
        with mock.patch.object(sys, "argv", ["train.py"]):
            args = get_args()
        self.assertTrue(args.hard_routing)
        self.assertTrue(args.freeze_base)
        self.assertTrue(args.use_aux_targets)
        self.assertEqual(args.duration_thresh_mode, "quantile")
        self.assertEqual(args.nr_weight, 0.05)
        self.assertEqual(args.aux_target_weight, 0.10)
        self.assertEqual(args.nr_pred_weight, 0.0)
        self.assertEqual(args.lambda_smooth_weight, 0.0)
        self.assertEqual(args.kurtosis_weight, 0.0)
        self.assertFalse(args.bucket_reweighting)
        self.assertFalse(args.backbone_autotune)

    def test_default_gate_is_hard_duration_routing(self):
        network = DebiasNetV2(
            user_vocab_size=2,
            video_vocab_size=2,
            bucket_num=3,
            hard_routing=True,
        )
        weights = network._routing_weights(
            torch.tensor([[0.1], [0.5], [0.9]]),
            thresholds=[0.3, 0.7],
        )
        torch.testing.assert_close(weights, torch.eye(3))
        with torch.no_grad():
            network.lambda_params.copy_(torch.tensor([-0.2, 0.1, 0.4]))
        routed_lambda = network.get_routed_lambda(
            torch.tensor([[0.1], [0.5], [0.9]]),
            thresholds=[0.3, 0.7],
        )
        torch.testing.assert_close(
            routed_lambda,
            torch.tensor([[-0.2], [0.1], [0.4]]),
        )


if __name__ == "__main__":
    unittest.main()
