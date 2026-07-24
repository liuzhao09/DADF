import sys
import unittest
from unittest import mock

import numpy as np
import torch

from model.dadf.network import DADF
from model.dadf.train import get_args
from model.dadf.transforms import boxcox_inverse, boxcox_transform
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
        self.assertFalse(args.shared_correction)
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

    def test_default_routing_is_duration_indexed(self):
        network = DADF(
            user_vocab_size=2,
            video_vocab_size=2,
            bucket_num=3,
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
    def test_boxcox_round_trip(self):
        values = torch.tensor([[0.2], [1.0], [3.0], [8.0]])
        lambdas = torch.tensor([[-0.2], [0.0], [0.3], [0.7]])
        recovered = boxcox_inverse(boxcox_transform(values, lambdas), lambdas)
        torch.testing.assert_close(recovered, values, rtol=1e-5, atol=1e-6)

    def test_dadf_forward_contract(self):
        network = DADF(
            user_vocab_size=4,
            video_vocab_size=5,
            bucket_num=3,
            aux_target_names=("short_view", "completion"),
        )
        transformed, auxiliary_logits = network(
            base_pred=torch.tensor([[0.2], [0.5], [0.8]]),
            user_id=torch.tensor([[0], [1], [2]]),
            video_id=torch.tensor([[0], [1], [2]]),
            duration=torch.tensor([[0.1], [0.5], [0.9]]),
            thresholds=[0.3, 0.7],
            proxy=torch.tensor([[0.2], [0.5], [0.8]]),
            return_aux=True,
        )
        self.assertEqual(transformed.shape, (3, 1))
        self.assertTrue(torch.isfinite(transformed).all())
        self.assertEqual(set(auxiliary_logits), {"short_view", "completion"})

    def test_correction_loss_does_not_update_auxiliary_heads(self):
        network = DADF(
            user_vocab_size=4,
            video_vocab_size=5,
            bucket_num=2,
            aux_target_names=("short_view",),
        )
        transformed = network(
            base_pred=torch.tensor([[0.2], [0.8]]),
            user_id=torch.tensor([[0], [1]]),
            video_id=torch.tensor([[0], [1]]),
            duration=torch.tensor([[0.1], [0.9]]),
            thresholds=[0.5],
            proxy=torch.tensor([[0.2], [0.8]]),
        )
        transformed.sum().backward()
        self.assertTrue(all(
            parameter.grad is None for parameter in network.aux_heads.parameters()
        ))

    def test_shared_correction_ablation_uses_one_expert(self):
        network = DADF(
            user_vocab_size=2,
            video_vocab_size=2,
            bucket_num=3,
            shared_correction=True,
        )
        self.assertEqual(len(network.regime_experts), 1)


if __name__ == "__main__":
    unittest.main()
