"""Smoke tests for the public T2exture configuration interfaces."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import LossWeights, TrainConfig, passive_context_ids
from flow_generation import load_pseudo_flow, missing_pseudo_flow_paths, pseudo_flow_path


class TestPublicConfiguration(unittest.TestCase):
    """Cover configuration that does not require a CUDA or PyTorch runtime."""

    def test_passive_context_uses_two_frames_on_each_side(self) -> None:
        """The target passive frame is excluded from a four-frame context."""
        self.assertEqual(passive_context_ids(center_id=10, context_size=4), (8, 9, 11, 12))

    def test_passive_context_can_be_disabled(self) -> None:
        """No-passive ablations use an empty context instead of special dataset logic."""
        self.assertEqual(passive_context_ids(center_id=10, context_size=0), ())

    def test_train_config_uses_required_defaults(self) -> None:
        """The public defaults preserve the approved two-stage setup."""
        config = TrainConfig()
        self.assertEqual(config.passive_context, 4)
        self.assertEqual(config.active_stride, 10)
        self.assertEqual(config.adapter_iterations, 10_000)
        self.assertEqual(config.finetune_iterations, 5_000)

    def test_loss_weights_are_fixed(self) -> None:
        """The three supervised objectives have the requested coefficients."""
        self.assertEqual(LossWeights().as_dict(), {'charbonnier': 1.0, 'css': 0.1, 'flow': 0.001})

    def test_loss_weights_can_be_overridden_from_config(self) -> None:
        """YAML-provided loss weights override only the requested keys."""
        weights = LossWeights.from_mapping({'flow': 0.002})
        self.assertEqual(weights.as_dict(), {'charbonnier': 1.0, 'css': 0.1, 'flow': 0.002})

    def test_pseudo_flow_index_matches_dataset_contract(self) -> None:
        """Flow labels are addressed inside each scene by endpoint-target IDs."""
        root = Path('dataset')
        expected = Path('dataset/flow/s10/scene/001_002_011.npz')
        self.assertEqual(pseudo_flow_path(root, 'scene', 1, 2, 11), expected)

    def test_relative_pseudo_flow_root_is_dataset_relative(self) -> None:
        """Experiment-specific flow sets are addressed under the dataset root."""
        root = Path('dataset')
        expected = Path('dataset/flow/s05/scene/001_002_006.npz')
        self.assertEqual(pseudo_flow_path(root, 'scene', 1, 2, 6, Path('flow/s05')), expected)

    def test_missing_pseudo_flow_uses_active_stride(self) -> None:
        """Active-sparsity experiments look for flow files matching their endpoint stride."""
        root = Path('dataset')
        missing = missing_pseudo_flow_paths(root, [('scene', 1, 1)], active_stride=5)
        self.assertEqual(missing, [Path('dataset/flow/s10/scene/001_002_006.npz')])

    def test_load_pseudo_flow_concatenates_target_to_endpoint_flows(self) -> None:
        """The flow loader returns AMT's expected [flow_t0, flow_t1] channel layout."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flow_dir = root / 'flow' / 's10' / 'scene'
            flow_dir.mkdir(parents=True)
            np.savez_compressed(
                flow_dir / '001_002_011.npz',
                flow0=np.ones((2, 3, 4), dtype=np.float32),
                flow1=np.full((2, 3, 4), 2.0, dtype=np.float32),
            )
            flow = load_pseudo_flow(root, 'scene', 1, 2, 11)
        self.assertEqual(flow.shape, (4, 3, 4))
        self.assertTrue(np.allclose(flow[:2], 1.0))
        self.assertTrue(np.allclose(flow[2:], 2.0))

    def test_load_pseudo_flow_falls_back_to_old_scene_flow_for_default_set(self) -> None:
        """Existing sim/<scene>/flow caches remain usable until the dataset is migrated."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flow_dir = root / 'sim' / 'scene' / 'flow'
            flow_dir.mkdir(parents=True)
            np.savez_compressed(
                flow_dir / '001_002_011.npz',
                flow0=np.ones((2, 3, 4), dtype=np.float32),
                flow1=np.full((2, 3, 4), 2.0, dtype=np.float32),
            )
            flow = load_pseudo_flow(root, 'scene', 1, 2, 11)
        self.assertEqual(flow.shape, (4, 3, 4))

    def test_batch_psnr_is_per_sample(self) -> None:
        """Training validation can aggregate PSNR over batches."""
        try:
            import torch
            from metrics import batch_psnr
        except ModuleNotFoundError as exc:
            self.skipTest(f'PyTorch is not available in this Python environment: {exc}')
        pred = torch.zeros(2, 1, 2, 2)
        target = torch.zeros(2, 1, 2, 2)
        target[1] = 1.0
        psnr = batch_psnr(pred, target)
        self.assertEqual(tuple(psnr.shape), (2,))
        self.assertGreater(float(psnr[0]), float(psnr[1]))

    def test_main_metric_keys_match_experiment_plan(self) -> None:
        """Metric output keys must match the table headers in the synthetic-only plan."""
        try:
            import torch
            from metrics import MAIN_METRIC_KEYS, compute_main_metrics
        except ModuleNotFoundError as exc:
            self.skipTest(f'PyTorch is not available in this Python environment: {exc}')
        pred = torch.zeros(2, 1, 8, 8)
        target = torch.zeros(2, 1, 8, 8)
        target[1, :, 2:6, 2:6] = 1.0
        metrics = compute_main_metrics(pred, target)
        self.assertEqual(tuple(metrics), MAIN_METRIC_KEYS)
        for value in metrics.values():
            self.assertEqual(tuple(value.shape), (2,))

    def test_deployment_metric_keys_match_experiment_plan(self) -> None:
        """Deployment table helpers expose the columns requested by the plan."""
        try:
            import torch.nn as nn
            from metrics import DEPLOYMENT_METRIC_KEYS, count_parameters, count_trainable_parameters
        except ModuleNotFoundError as exc:
            self.skipTest(f'PyTorch is not available in this Python environment: {exc}')
        model = nn.Linear(3, 2)
        self.assertEqual(DEPLOYMENT_METRIC_KEYS, ('Latency', 'FLOPs', 'Params', 'Trainable Params'))
        self.assertEqual(count_parameters(model), 8)
        self.assertEqual(count_trainable_parameters(model), 8)
