"""Smoke tests for the public T2exture configuration interfaces."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import LossWeights, TrainConfig, expected_flow_set, passive_context_ids, resolve_sample_passive_context, validate_runtime_config
from flow_generation import load_pseudo_flow, missing_pseudo_flow_paths, pseudo_flow_path


class TestPublicConfiguration(unittest.TestCase):
    """Cover configuration that does not require a CUDA or PyTorch runtime."""

    def test_passive_context_is_centered_on_target_time(self) -> None:
        """The default structural context includes the target-time passive frame."""
        self.assertEqual(passive_context_ids(center_id=10, context_size=5), (8, 9, 10, 11, 12))

    def test_passive_context_rejects_even_sizes(self) -> None:
        """Eq. (10) uses a center frame, so enabled contexts must have odd size."""
        with self.assertRaises(ValueError):
            passive_context_ids(center_id=10, context_size=4)

    def test_passive_context_can_be_disabled(self) -> None:
        """No-passive ablations use an empty context instead of special dataset logic."""
        self.assertEqual(passive_context_ids(center_id=10, context_size=0), ())

    def test_train_config_uses_required_defaults(self) -> None:
        """The public defaults preserve the approved two-stage setup."""
        config = TrainConfig()
        self.assertEqual(config.passive_context, 5)
        self.assertEqual(config.active_stride, 10)
        self.assertIsNone(config.sample_passive_context)
        self.assertTrue(config.require_datasets_root)
        self.assertEqual(config.adapter_iterations, 10_000)
        self.assertEqual(config.finetune_iterations, 5_000)

    def test_expected_flow_set_matches_active_stride(self) -> None:
        """The flow-set naming convention prevents s05/s10 mixups."""
        self.assertEqual(expected_flow_set(5), 's05')

    def test_sample_passive_context_keeps_sampling_window_fixed(self) -> None:
        """Context ablations can share the same largest passive sampling window."""
        self.assertEqual(resolve_sample_passive_context({'sample_passive_context': 11}, passive_context=0), 11)

    def test_public_runs_require_datasets_root(self) -> None:
        """Public runs fail early when a command accidentally points at a source cache."""
        with self.assertRaises(ValueError):
            validate_runtime_config({'require_datasets_root': True}, Path('dataset'))

    def test_flow_set_must_match_active_stride(self) -> None:
        """Stride-specific flow caches must match the active-frame stride."""
        with self.assertRaises(ValueError):
            validate_runtime_config({'require_datasets_root': False, 'active_stride': 5, 'pseudo_flow_dir': 'flow/s10'}, Path('dataset'))

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
        """Existing sim/<scene>/flow caches remain usable during local migration."""
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

    def test_main_metric_keys_match_public_outputs(self) -> None:
        """Metric output keys stay stable for downstream reporting."""
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

    def test_edge_fi_uses_canny_boundary_matching(self) -> None:
        """Canny Edge-FI rewards identical edges and penalizes missing ones."""
        try:
            import torch
            from metrics import batch_edge_fi
        except ModuleNotFoundError as exc:
            self.skipTest(f'Required metric dependencies are not available: {exc}')
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 8:24, 8:24] = 1.0
        pred = target.clone()
        pred[1] = 0.0
        scores = batch_edge_fi(pred, target, tolerance_px=2)
        self.assertAlmostEqual(float(scores[0]), 1.0, places=6)
        self.assertLess(float(scores[1]), 0.5)

    def test_checkpoint_state_extraction_accepts_public_formats(self) -> None:
        """Evaluation and inference can read raw, training, and legacy states."""
        try:
            import torch
            from utils.checkpoint import extract_model_state
        except ModuleNotFoundError as exc:
            self.skipTest(f'PyTorch is not available in this Python environment: {exc}')

        tensor = torch.ones(1)
        self.assertIs(extract_model_state({'weight': tensor})['weight'], tensor)
        self.assertIs(extract_model_state({'model': {'weight': tensor}})['weight'], tensor)
        self.assertIs(extract_model_state({'state_dict': {'weight': tensor}})['weight'], tensor)

    def test_main_figure_conditioning_shapes(self) -> None:
        """T2V, FourierConv2d, and Conv-P pyramid match the public architecture."""
        try:
            import torch
            from model.conditioning import FourierConv2d, PassiveGuidancePyramid, T2VAdapter
        except ModuleNotFoundError as exc:
            self.skipTest(f'PyTorch is not available in this Python environment: {exc}')

        time = torch.tensor([0.25, 0.75], dtype=torch.float32)
        adapter = T2VAdapter()
        texture = torch.rand(2, 1, 32, 48)
        self.assertEqual(tuple(adapter(texture).shape), (2, 3, 32, 48))
        self.assertEqual(tuple(FourierConv2d(20)(time).shape), (2, 20, 1, 1))

        pyramid = PassiveGuidancePyramid(context_size=5, decoder_channels=(20, 32, 44))
        guidance = pyramid(torch.rand(2, 5, 32, 48), time)
        self.assertEqual([tuple(item.shape) for item in guidance], [(2, 20, 16, 24), (2, 32, 8, 12), (2, 44, 4, 6)])
