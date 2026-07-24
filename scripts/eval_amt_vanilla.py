"""Evaluate official AMT-S/L/G checkpoints on the T2exture ROI split."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import resolve_sample_passive_context, validate_runtime_config
from data import TextureDataset
from eval import _batch_ints, _batch_strings, _mean_rows, _move_batch, _relative, _sample_name, _save_png, _state_dict, _torch_load
from metrics import MAIN_METRIC_KEYS, compute_main_metrics


BACKBONE_FILES = {
    'amt-s': 'AMT-S.py',
    'amt-l': 'AMT-L.py',
    'amt-g': 'AMT-G.py',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--backbone', choices=sorted(BACKBONE_FILES), required=True)
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='test')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('train.yaml'))
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def load_backbone_class(backbone: str) -> type[nn.Module]:
    package_root = ROOT / 'third_party' / 'AMT_official'
    path = package_root / 'networks' / BACKBONE_FILES[backbone]
    spec = importlib.util.spec_from_file_location(f't2texture_vanilla_{backbone}', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load {backbone} from {path}')
    sys.path.insert(0, str(package_root))
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == str(package_root):
            sys.path.pop(0)
    return module.Model


def load_model(backbone: str, checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = load_backbone_class(backbone)()
    checkpoint = _torch_load(checkpoint_path, device)
    model.load_state_dict(_state_dict(checkpoint), strict=True)
    return model.to(device).eval()


def texture_to_rgb(texture: torch.Tensor) -> torch.Tensor:
    if texture.shape[1] != 1:
        raise ValueError(f'Expected one-channel texture tensor, got {texture.shape}')
    return texture.repeat(1, 3, 1, 1)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    validate_runtime_config(config, args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    passive_context = int(config.get('passive_context', 4))
    active_stride = int(config.get('active_stride', 10))
    sample_passive_context = resolve_sample_passive_context(config, passive_context)
    pseudo_flow_root = Path(config['pseudo_flow_dir']) if config.get('pseudo_flow_dir') else None
    dataset = TextureDataset(
        args.data_root,
        args.data_root / f'{args.split}.txt',
        passive_context=passive_context,
        crop_size=None,
        random_crop=False,
        pseudo_flow_root=pseudo_flow_root,
        active_stride=active_stride,
        sample_passive_context=sample_passive_context,
    )
    batch_size = args.batch_size or int(config.get('eval_batch_size', config['batch_size']))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs(config, device))
    model = load_model(args.backbone, args.pretrained, device)

    frame_rows: list[dict[str, Any]] = []
    scene_rows: dict[str, list[dict[str, float]]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            output = model(
                texture_to_rgb(batch['texture0']),
                texture_to_rgb(batch['texture1']),
                batch['time'].view(batch['time'].shape[0], 1, 1, 1),
                eval=True,
            )
            prediction = output['imgt_pred'].mean(dim=1, keepdim=True).clamp(0.0, 1.0)
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
                row = {
                    'scene': scene,
                    'left_id': left_id,
                    'target_id': target_id,
                    'right_id': right_id,
                    **metrics,
                    'pred_path': _relative(pred_path, args.output_dir),
                    'err_path': _relative(err_path, args.output_dir),
                }
                frame_rows.append(row)
                scene_rows[scene].append(metrics)

    if not frame_rows:
        raise ValueError(f'No samples were evaluated for split {args.split!r}')

    overall = _mean_rows([{key: float(row[key]) for key in MAIN_METRIC_KEYS} for row in frame_rows])
    scene_summary = [{'scene': scene, 'frames': len(rows), **_mean_rows(rows)} for scene, rows in sorted(scene_rows.items())]
    write_csv(args.output_dir / 'frame.csv', frame_rows, ['scene', 'left_id', 'target_id', 'right_id', *MAIN_METRIC_KEYS, 'pred_path', 'err_path'])
    write_csv(args.output_dir / 'scene.csv', scene_summary, ['scene', 'frames', *MAIN_METRIC_KEYS])

    metrics = {
        'backbone': args.backbone,
        'setting': 'vanilla',
        'split': args.split,
        'frames': len(frame_rows),
        'overall': overall,
        'by_scene': scene_summary,
    }
    manifest = {
        **metrics,
        'data_root': str(args.data_root.resolve()),
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
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metrics['overall'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
