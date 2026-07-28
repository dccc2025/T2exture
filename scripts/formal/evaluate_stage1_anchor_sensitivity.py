"""Appendix diagnostic for Stage-1 passive-anchor error propagation.

This script keeps the trained Ours checkpoint fixed and evaluates two input
conditions on the formal test split:

1. oracle texture anchors from the prepared dataset;
2. texture anchors reconstructed after replacing the anchor passive state with
   a complete LiteFlowNet Stage-1 passive estimate.

It is intentionally separate from eval.py so the formal main-table evaluation
code stays unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import resolve_sample_passive_context, validate_runtime_config
from data import TextureDataset, _load_frame, _load_pseudo_flow, _normalise
from eval import _batch_ints, _batch_strings, _relative, _sample_name, _save_png, _state_dict
from metrics import MAIN_METRIC_KEYS, batch_ie, compute_main_metrics
from model import build_t2texture_model
from train import move_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('datasets'))
    parser.add_argument('--pretrained', type=Path, default=Path('pretrained/amt-l.pth'))
    parser.add_argument('--checkpoint', type=Path, default=Path('outputs/final/table03_passive_context/context-per-side-02/best.pt'))
    parser.add_argument('--config', type=Path, default=Path('outputs/final/table03_passive_context/context-per-side-02/config.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/final/appendix_stage1_error'))
    parser.add_argument('--backbone', choices=['amt-s', 'amt-l', 'amt-g'], default='amt-l')
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='test')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, default=None, help='Optional debug limit on evaluated samples.')
    return parser.parse_args()


def torch_load(path: Path, device: torch.device) -> Any:
    """Load checkpoints saved with either POSIX or Windows Path objects."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except NotImplementedError as exc:
        if 'PosixPath' not in str(exc):
            raise
        original_posix_path = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        try:
            return torch.load(path, map_location=device, weights_only=False)
        finally:
            pathlib.PosixPath = original_posix_path


def loader_kwargs(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    workers = int(config.get('num_workers', 0))
    kwargs: dict[str, Any] = {
        'num_workers': workers,
        'pin_memory': bool(config.get('pin_memory', False)) and device.type == 'cuda',
    }
    if workers > 0:
        kwargs['persistent_workers'] = bool(config.get('persistent_workers', False))
        kwargs['prefetch_factor'] = int(config.get('prefetch_factor', 2))
    return kwargs


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() == '.json':
        return json.loads(text)
    return yaml.safe_load(text)


def as_tensor_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image.astype(np.float32, copy=True)).to(device=device).unsqueeze(0)


def frame_tensor(data_root: Path, scene: str, modality: str, frame_id: int, device: torch.device, normalise: bool = True) -> torch.Tensor:
    image = _load_frame(data_root, scene, modality, frame_id)
    if normalise:
        image = _normalise(image)
    return as_tensor_image(image, device)


def forward_splat(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Forward-splat a single-channel image with target-to-anchor pixel flow."""
    if image.ndim != 3 or image.shape[0] != 1:
        raise ValueError(f'Expected image [1,H,W], got {tuple(image.shape)}')
    if flow.shape != (2, image.shape[-2], image.shape[-1]):
        raise ValueError(f'Expected flow [2,H,W], got {tuple(flow.shape)} for image {tuple(image.shape)}')

    _, height, width = image.shape
    dtype = image.dtype
    device = image.device
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing='ij',
    )
    x = xx + flow[0]
    y = yy + flow[1]
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    dx = x - x0
    dy = y - y0

    flat_image = image[0].reshape(-1)
    accum = torch.zeros(height * width, device=device, dtype=dtype)
    weight_sum = torch.zeros_like(accum)

    for x_offset, y_offset, weight in (
        (0, 0, (1.0 - dx) * (1.0 - dy)),
        (1, 0, dx * (1.0 - dy)),
        (0, 1, (1.0 - dx) * dy),
        (1, 1, dx * dy),
    ):
        xi = (x0 + x_offset).long()
        yi = (y0 + y_offset).long()
        valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        if not bool(valid.any()):
            continue
        index = (yi[valid] * width + xi[valid]).reshape(-1)
        w = weight[valid].reshape(-1).clamp_min(0.0)
        values = flat_image[valid.reshape(-1)] * w
        accum.scatter_add_(0, index, values)
        weight_sum.scatter_add_(0, index, w)

    splatted = accum / weight_sum.clamp_min(1e-8)
    fallback = image[0].reshape(-1)
    splatted = torch.where(weight_sum > 1e-6, splatted, fallback)
    return splatted.reshape(1, height, width).clamp(0.0, 1.0)


class LiteFlowNetStage1:
    """Complete Stage-1 passive-key estimator backed by cached LiteFlowNet flows."""

    def __init__(
        self,
        data_root: Path,
        output_dir: Path,
        pseudo_flow_root: Path | None,
        active_stride: int,
        device: torch.device,
    ) -> None:
        self.data_root = data_root
        self.output_dir = output_dir
        self.pseudo_flow_root = pseudo_flow_root
        self.active_stride = active_stride
        self.device = device
        self._memory: dict[tuple[str, int], np.ndarray] = {}
        (self.output_dir / 'pred').mkdir(parents=True, exist_ok=True)

    def _path(self, scene: str, frame_id: int) -> Path:
        return self.output_dir / 'pred' / scene / f'{frame_id:03d}_passive_pred.npy'

    def _source_to_anchor_flow(self, scene: str, source_id: int, anchor_id: int) -> torch.Tensor | None:
        if source_id < 1 or source_id > 180:
            return None
        if source_id < anchor_id:
            left_id = anchor_id - self.active_stride
            right_id = anchor_id
            if left_id < 1 or not (left_id < source_id < right_id):
                return None
            flow = _load_pseudo_flow(self.data_root, scene, left_id, source_id, right_id, self.pseudo_flow_root)[2:4]
        else:
            left_id = anchor_id
            right_id = anchor_id + self.active_stride
            if right_id > 181 or not (left_id < source_id < right_id):
                return None
            flow = _load_pseudo_flow(self.data_root, scene, left_id, source_id, right_id, self.pseudo_flow_root)[0:2]
        return torch.from_numpy(flow.copy()).to(device=self.device)

    def predict(self, scene: str, frame_id: int) -> torch.Tensor:
        key = (scene, frame_id)
        if key in self._memory:
            return as_tensor_image(self._memory[key], self.device)
        path = self._path(scene, frame_id)
        if path.is_file():
            pred = np.load(path).astype(np.float32)
            self._memory[key] = pred
            return as_tensor_image(pred, self.device)

        warped_sources: list[torch.Tensor] = []
        for source_id in (frame_id - 1, frame_id + 1):
            flow = self._source_to_anchor_flow(scene, source_id, frame_id)
            if flow is None:
                continue
            source = _load_frame(self.data_root, scene, 'passive', source_id).astype(np.float32)
            warped_sources.append(forward_splat(as_tensor_image(source, self.device), flow))
        if not warped_sources:
            raise FileNotFoundError(f'No cached LiteFlowNet source-to-anchor flow for {scene} frame {frame_id:03d}')
        pred = torch.stack(warped_sources, dim=0).mean(dim=0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, pred)
        self._memory[key] = pred
        return as_tensor_image(pred, self.device)


def derived_anchor(
    data_root: Path,
    scene: str,
    anchor_id: int,
    stage1: LiteFlowNetStage1,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Stage-1-derived anchor, oracle anchor, and passive IE."""
    texture_raw = frame_tensor(data_root, scene, 'texture', anchor_id, device, normalise=False)
    passive_anchor_raw = frame_tensor(data_root, scene, 'passive', anchor_id, device, normalise=False)
    passive_est_raw = stage1.predict(scene, anchor_id)

    active_raw = texture_raw + passive_anchor_raw
    derived_raw = (active_raw - passive_est_raw).clamp_min(0.0)

    derived = as_tensor_image(_normalise(derived_raw.squeeze(0).detach().cpu().numpy()), device)
    oracle = as_tensor_image(_normalise(texture_raw.squeeze(0).detach().cpu().numpy()), device)
    stage1_ie = batch_ie(
        as_tensor_image(_normalise(passive_est_raw.squeeze(0).detach().cpu().numpy()), device).unsqueeze(0),
        as_tensor_image(_normalise(passive_anchor_raw.squeeze(0).detach().cpu().numpy()), device).unsqueeze(0),
    )[0]
    return derived, oracle, stage1_ie


def prepare_stage1_batch(
    batch: dict[str, Any],
    data_root: Path,
    stage1: LiteFlowNetStage1,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    scenes = _batch_strings(batch['scene'])
    left_ids = _batch_ints(batch['left_id'])
    right_ids = _batch_ints(batch['right_id'])
    derived_left: list[torch.Tensor] = []
    derived_right: list[torch.Tensor] = []
    oracle_left: list[torch.Tensor] = []
    oracle_right: list[torch.Tensor] = []
    stage1_ies: list[torch.Tensor] = []
    for index, scene in enumerate(scenes):
        left_id = left_ids[index]
        right_id = right_ids[index]
        left, oracle0, stage1_ie0 = derived_anchor(data_root, scene, left_id, stage1, device)
        right, oracle1, stage1_ie1 = derived_anchor(data_root, scene, right_id, stage1, device)
        derived_left.append(left)
        derived_right.append(right)
        oracle_left.append(oracle0)
        oracle_right.append(oracle1)
        stage1_ies.append(0.5 * (stage1_ie0 + stage1_ie1))

    updated = dict(batch)
    updated['texture0'] = torch.stack(derived_left, dim=0)
    updated['texture1'] = torch.stack(derived_right, dim=0)
    anchor_ie = 0.5 * (
        batch_ie(torch.stack(derived_left, dim=0), torch.stack(oracle_left, dim=0))
        + batch_ie(torch.stack(derived_right, dim=0), torch.stack(oracle_right, dim=0))
    )
    return updated, anchor_ie, torch.stack(stage1_ies)


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError('Cannot average an empty metric group')
    keys = [*MAIN_METRIC_KEYS, 'Stage1-IE', 'Anchor-IE']
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


@torch.no_grad()
def evaluate_setting(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    data_root: Path,
    stage1: LiteFlowNetStage1 | None,
    output_dir: Path,
    setting: str,
    limit: int | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, Any]] = []
    scene_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    model.eval()

    for batch in loader:
        batch = move_batch(batch, device)
        if setting == 'liteflownet_stage1_passive':
            if stage1 is None:
                raise ValueError('Stage-1 estimator is required for liteflownet_stage1_passive')
            batch, anchor_ie, stage1_ie = prepare_stage1_batch(batch, data_root, stage1, device)
        elif setting == 'oracle_passive':
            anchor_ie = torch.zeros(int(batch['target'].shape[0]), device=device, dtype=batch['target'].dtype)
            stage1_ie = torch.zeros_like(anchor_ie)
        else:
            raise ValueError(f'Unknown setting {setting!r}')

        result = model(batch['texture0'], batch['texture1'], batch['time'], batch['passive_context'], return_flow=False)
        prediction = result['prediction'].clamp(0.0, 1.0)
        target = batch['target'].clamp(0.0, 1.0)
        metric_tensors = compute_main_metrics(prediction, target)

        scenes = _batch_strings(batch['scene'])
        left_ids = _batch_ints(batch['left_id'])
        target_ids = _batch_ints(batch['target_id'])
        right_ids = _batch_ints(batch['right_id'])
        batch_size = int(prediction.shape[0])

        for index in range(batch_size):
            scene = scenes[index]
            left_id = left_ids[index]
            target_id = target_ids[index]
            right_id = right_ids[index]
            name = _sample_name(left_id, target_id, right_id)
            pred_path = output_dir / 'pred' / scene / f'{name}.png'
            err_path = output_dir / 'err' / scene / f'{name}.png'
            _save_png(prediction[index], pred_path)
            _save_png((prediction[index] - target[index]).abs(), err_path)
            metrics = {key: float(metric_tensors[key][index].detach().cpu()) for key in MAIN_METRIC_KEYS}
            metrics['Anchor-IE'] = float(anchor_ie[index].detach().cpu())
            metrics['Stage1-IE'] = float(stage1_ie[index].detach().cpu())
            row = {
                'scene': scene,
                'left_id': left_id,
                'target_id': target_id,
                'right_id': right_id,
                **metrics,
                'pred_path': _relative(pred_path, output_dir),
                'err_path': _relative(err_path, output_dir),
            }
            frame_rows.append(row)
            scene_rows[scene].append(metrics)
            if limit is not None and len(frame_rows) >= limit:
                break
        if limit is not None and len(frame_rows) >= limit:
            break

    overall = mean_rows([{key: float(row[key]) for key in [*MAIN_METRIC_KEYS, 'Anchor-IE', 'Stage1-IE']} for row in frame_rows])
    scene_summary = [{'scene': scene, 'frames': len(rows), **mean_rows(rows)} for scene, rows in sorted(scene_rows.items())]

    with (output_dir / 'frame.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['scene', 'left_id', 'target_id', 'right_id', 'Stage1-IE', 'Anchor-IE', *MAIN_METRIC_KEYS, 'pred_path', 'err_path'])
        writer.writeheader()
        writer.writerows(frame_rows)
    with (output_dir / 'scene.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['scene', 'frames', 'Stage1-IE', 'Anchor-IE', *MAIN_METRIC_KEYS])
        writer.writeheader()
        writer.writerows(scene_summary)

    metrics = {
        'setting': setting,
        'split': 'test',
        'frames': len(frame_rows),
        'overall': overall,
        'by_scene': scene_summary,
    }
    manifest = {
        **metrics,
        'diagnostic': 'stage1_passive_anchor_error_propagation',
        'passive_estimator': 'Cached LiteFlowNet source-to-anchor flows from neighboring observed passive frames',
        'data_root': str(data_root.resolve()),
        'artifacts': {
            'pred': 'pred/',
            'err': 'err/',
            'frame_csv': 'frame.csv',
            'scene_csv': 'scene.csv',
            'metrics_json': 'metrics.json',
        },
    }
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_runtime_config(config, args.data_root)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

    passive_context = int(config.get('passive_context', 4))
    active_stride = int(config.get('active_stride', 10))
    sample_passive_context = resolve_sample_passive_context(config, passive_context)
    pseudo_flow_root = Path(config['pseudo_flow_dir']) if config.get('pseudo_flow_dir') else None
    dataset = TextureDataset(
        args.data_root,
        args.data_root / f'{args.split}.txt',
        passive_context,
        crop_size=None,
        random_crop=False,
        pseudo_flow_root=pseudo_flow_root,
        active_stride=active_stride,
        sample_passive_context=sample_passive_context,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs(config, device))

    model = build_t2texture_model(args.backbone, args.pretrained, passive_context).to(device)
    checkpoint = torch_load(args.checkpoint, device)
    model.load_state_dict(_state_dict(checkpoint), strict=True)

    stage1 = LiteFlowNetStage1(args.data_root, args.output_dir / 'stage1_passive', pseudo_flow_root, active_stride, device)

    results = []
    for setting in ('oracle_passive', 'liteflownet_stage1_passive'):
        metrics = evaluate_setting(
            model,
            loader,
            device,
            args.data_root,
            stage1 if setting == 'liteflownet_stage1_passive' else None,
            args.output_dir / setting / args.split,
            setting,
            args.limit,
        )
        results.append(metrics)

    summary_rows = []
    oracle_psnr = float(results[0]['overall']['PSNR'])
    for metrics in results:
        row = {'setting': metrics['setting'], 'frames': metrics['frames'], **metrics['overall']}
        row['Delta-PSNR'] = float(row['PSNR']) - oracle_psnr
        summary_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['setting', 'frames', 'Stage1-IE', 'Anchor-IE', *MAIN_METRIC_KEYS, 'Delta-PSNR'])
        writer.writeheader()
        writer.writerows(summary_rows)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary_rows, indent=2) + '\n')
    print(json.dumps(summary_rows, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
