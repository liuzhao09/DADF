import sys
import unittest
from unittest import mock

import numpy as np
import torch

from model import Cread, D2Q, EGMN, TPM, WideAndDeep
from model.dadf.network import DADF
from model.dadf.train import _parameter_counts, get_args
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
        self.assertFalse(args.base_only)
        self.assertEqual(args.base_mlp_dims, [256, 128, 64])

    def test_dense_parameter_count_excludes_embeddings(self):
        network = torch.nn.Sequential(
            torch.nn.Embedding(10, 4),
            torch.nn.Linear(4, 2),
        )
        total, dense = _parameter_counts(network)
        self.assertEqual(total, 50)
        self.assertEqual(dense, 10)

    def test_kuairec_default_dense_parameter_contract(self):
        description = [
            ("play_time", -1, "label"),
            ("duration", -1, "ctn"),
            ("user_id", 4, "spr"),
            ("video_id", 5, "spr"),
            ("duration_bucket", 50, "spr"),
        ]
        description.extend(
            ("feature_{}".format(index), 3, "spr")
            for index in range(32)
        )

        default_dims = (256, 128, 64)
        models = {
            "vr": WideAndDeep(description, 16, default_dims, 0.0),
            "wlr": WideAndDeep(description, 16, default_dims, 0.0),
            "tpm": TPM(description, 31, 16, default_dims, 0.0),
            "d2q": D2Q(description, 16, default_dims, 0.0),
            "cread": Cread(description, 16, default_dims, (32,), 50, 0.0),
            "d2co": WideAndDeep(description, 16, default_dims, 0.0),
            "egmn": EGMN(description, 16, default_dims, 0.2),
        }
        expected_backbone_dense = {
            "vr": 185730,
            "wlr": 185730,
            "tpm": 187936,
            "d2q": 185986,
            "cread": 294771,
            "d2co": 185730,
            "egmn": 188001,
        }
        expected_combined_dense = {
            "vr": 253649,
            "wlr": 253649,
            "tpm": 247663,
            "d2q": 245713,
            "cread": 354498,
            "d2co": 253649,
            "egmn": 255920,
        }
        hidden_backbones = {"vr", "wlr", "d2co", "egmn"}
        auxiliary_targets = tuple("aux_{}".format(index) for index in range(7))

        for name, backbone in models.items():
            dadf = DADF(
                user_vocab_size=4,
                video_vocab_size=5,
                bucket_num=4,
                hidden_dim_base=64 if name in hidden_backbones else 0,
                aux_target_names=auxiliary_targets,
            )
            with self.subTest(backbone=name):
                self.assertEqual(
                    _parameter_counts(backbone)[1],
                    expected_backbone_dense[name],
                )
                self.assertEqual(
                    _parameter_counts(backbone, dadf)[1],
                    expected_combined_dense[name],
                )

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
