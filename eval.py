"""Evaluate T2exture checkpoints and export metrics plus prediction frames."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from config import resolve_sample_passive_context, validate_runtime_config
from data import TextureDataset
from metrics import MAIN_METRIC_KEYS, compute_main_metrics
from model import build_t2texture_model


def parse_args() -> argparse.Namespace:
    """Read the checkpoint, dataset split, and output location for evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--backbone', choices=['amt-s', 'amt-l', 'amt-g'], default='amt-l')
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='test')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('train.yaml'))
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def _torch_load(path: Path, device: torch.device) -> Any:
    """Load checkpoints saved on Linux or Windows hosts."""
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


def _state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract a model state dict from a training checkpoint or raw state dict."""
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f'Unsupported checkpoint format: {type(checkpoint)!r}')


def _sample_name(left_id: int, target_id: int, right_id: int) -> str:
    """Return the stable endpoint-target filename stem used by eval and vis."""
    return f'{left_id:03d}_{target_id:03d}_{right_id:03d}'


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert one normalized image tensor to a grayscale uint8 array."""
    array = image.detach().float().clamp(0.0, 1.0).cpu().numpy()
    array = np.squeeze(array)
    return np.round(array * 255.0).astype(np.uint8)


def _save_png(image: torch.Tensor, path: Path) -> None:
    """Save one normalized grayscale image tensor as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_to_uint8(image), mode='L').save(path)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values to the evaluation device without touching metadata fields."""
    non_blocking = device.type == 'cuda'
    return {key: value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value for key, value in batch.items()}


def _loader_kwargs(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Return DataLoader options shared with training."""
    workers = int(config.get('num_workers', 0))
    kwargs: dict[str, Any] = {
        'num_workers': workers,
        'pin_memory': bool(config.get('pin_memory', False)) and device.type == 'cuda',
    }
    if workers > 0:
        kwargs['persistent_workers'] = bool(config.get('persistent_workers', False))
        kwargs['prefetch_factor'] = int(config.get('prefetch_factor', 2))
    return kwargs


def _resolve_data_path(root: Path, value: str | Path) -> Path:
    """Resolve a config path relative to the dataset root unless it is absolute."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def _batch_ints(value: Any) -> list[int]:
    """Convert a collated integer metadata field to a Python list."""
    if torch.is_tensor(value):
        return [int(item) for item in value.detach().cpu().flatten().tolist()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _batch_strings(value: Any) -> list[str]:
    """Convert a collated string metadata field to a Python list."""
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _relative(path: Path, root: Path) -> str:
    """Return a portable relative path for CSV and JSON records."""
    return str(path.relative_to(root)).replace('\\', '/')


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average the main metric keys over a list of frame-level rows."""
    if not rows:
        raise ValueError('Cannot average an empty metric group')
    return {key: float(np.mean([row[key] for row in rows])) for key in MAIN_METRIC_KEYS}


def main() -> None:
    """Run evaluation and write frame, scene, and overall metric artifacts."""
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    validate_runtime_config(config, args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    pseudo_flow_root = Path(config['pseudo_flow_dir']) if config.get('pseudo_flow_dir') else None
    passive_context = int(config.get('passive_context', 4))
    active_stride = int(config.get('active_stride', 10))
    sample_passive_context = resolve_sample_passive_context(config, passive_context)
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
    batch_size = args.batch_size or int(config.get('eval_batch_size', config['batch_size']))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **_loader_kwargs(config, device))

    model = build_t2texture_model(args.backbone, args.pretrained, passive_context).to(device)
    checkpoint = _torch_load(args.checkpoint, device)
    model.load_state_dict(_state_dict(checkpoint), strict=True)
    model.eval()

    frame_rows: list[dict[str, Any]] = []
    scene_rows: dict[str, list[dict[str, float]]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
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
                pred_path = args.output_dir / 'pred' / scene / f'{name}.png'
                err_path = args.output_dir / 'err' / scene / f'{name}.png'
                _save_png(prediction[index], pred_path)
                _save_png((prediction[index] - target[index]).abs(), err_path)

                metrics = {key: float(metric_tensors[key][index].detach().cpu()) for key in MAIN_METRIC_KEYS}
                metric_row = {
                    'scene': scene,
                    'left_id': left_id,
                    'target_id': target_id,
                    'right_id': right_id,
                    **metrics,
                    'pred_path': _relative(pred_path, args.output_dir),
                    'err_path': _relative(err_path, args.output_dir),
                }
                frame_rows.append(metric_row)
                scene_rows[scene].append(metrics)

    if not frame_rows:
        raise ValueError(f'No samples were evaluated for split {args.split!r}')

    overall = _mean_rows([{key: float(row[key]) for key in MAIN_METRIC_KEYS} for row in frame_rows])
    scene_summary = [{'scene': scene, 'frames': len(rows), **_mean_rows(rows)} for scene, rows in sorted(scene_rows.items())]
    frame_csv = args.output_dir / 'frame.csv'
    scene_csv = args.output_dir / 'scene.csv'
    with frame_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['scene', 'left_id', 'target_id', 'right_id', *MAIN_METRIC_KEYS, 'pred_path', 'err_path'])
        writer.writeheader()
        writer.writerows(frame_rows)
    with scene_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['scene', 'frames', *MAIN_METRIC_KEYS])
        writer.writeheader()
        writer.writerows(scene_summary)

    metrics = {
        'backbone': args.backbone,
        'split': args.split,
        'frames': len(frame_rows),
        'overall': overall,
        'by_scene': scene_summary,
    }
    manifest = {
        **metrics,
        'data_root': str(args.data_root.resolve()),
        'checkpoint': str(args.checkpoint.resolve()),
        'pretrained': str(args.pretrained.resolve()),
        'config': config,
        'artifacts': {
            'pred': 'pred/',
            'err': 'err/',
            'frame_csv': 'frame.csv',
            'scene_csv': 'scene.csv',
            'metrics_json': 'metrics.json',
        },
    }
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(metrics['overall'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
