"""Smoke tests for the public T2exture configuration interfaces."""

from __future__ import annotations

import unittest

from config import LossWeights, TrainConfig, passive_context_ids


class TestPublicConfiguration(unittest.TestCase):
    """Cover configuration that does not require a CUDA or PyTorch runtime."""

    def test_passive_context_uses_two_frames_on_each_side(self) -> None:
        """The target passive frame is excluded from a four-frame context."""
        self.assertEqual(passive_context_ids(center_id=10, context_size=4), (8, 9, 11, 12))

    def test_train_config_uses_required_defaults(self) -> None:
        """The public defaults preserve the approved two-stage setup."""
        config = TrainConfig()
        self.assertEqual(config.passive_context, 4)
        self.assertEqual(config.adapter_iterations, 10_000)
        self.assertEqual(config.finetune_iterations, 5_000)

    def test_loss_weights_are_fixed(self) -> None:
        """The three supervised objectives have the requested coefficients."""
        self.assertEqual(LossWeights().as_dict(), {'charbonnier': 1.0, 'css': 0.1, 'flow': 0.001})
