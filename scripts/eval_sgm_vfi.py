"""Evaluate official SGM-VFI checkpoints on the T2exture ROI split."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import resolve_sample_passive_context, validate_runtime_config
from data import TextureDataset
from eval import _batch_ints, _batch_strings, _mean_rows, _move_batch, _relative, _sample_name, _save_png
from metrics import MAIN_METRIC_KEYS, compute_main_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pretrained-root', type=Path, default=Path('pretrained/sgm-vfi'))
    parser.add_argument('--exp-name', default='ours-1-2-points')
    parser.add_argument('--num-key-points', type=float, default=0.5)
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='test')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('train.yaml'))
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--down-scale', type=float, default=1.0)
    return parser.parse_args()


def texture_to_rgb(texture: torch.Tensor) -> torch.Tensor:
    if texture.shape[1] != 1:
        raise ValueError(f'Expected one-channel texture tensor, got {texture.shape}')
    return texture.repeat(1, 3, 1, 1)


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_model(args: argparse.Namespace):
    if args.device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('SGM-VFI runner expects CUDA because the official Trainer_x4k moves the model to cuda.')

    package_root = ROOT / 'third_party' / 'SGM-VFI'
    pretrained_root = args.pretrained_root.resolve()
    os.environ['SGM_VFI_PRETRAINED_ROOT'] = str(pretrained_root)

    # SGM imports top-level modules named "config" and "model", which collide
    # with this project's packages. Isolate those imports and restore after load.
    isolated_names = [
        name
        for name in sys.modules
        if name == 'config'
        or name == 'model'
        or name.startswith('model.')
        or name == 'Trainer_x4k'
        or name.startswith('Trainer_x4k.')
    ]
    previous_modules = {name: sys.modules[name] for name in isolated_names}
    for name in isolated_names:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(package_root))
    try:
        sgm_config = importlib.import_module('config')
        sgm_config.MODEL_CONFIG['LOGNAME'] = 'ours_small'
        sgm_config.MODEL_CONFIG['MODEL_ARCH'] = sgm_config.init_model_config(
            F=16,
            depth=[2, 2, 2, 4],
            num_key_points=args.num_key_points,
        )
        trainer_module = importlib.import_module('Trainer_x4k')
        model = trainer_module.Model(-1)
        model.load_model(name=args.exp_name)
        model.eval()
        model.device()
    finally:
        if sys.path and sys.path[0] == str(package_root):
            sys.path.pop(0)
        loaded_names = [
            name
            for name in sys.modules
            if name == 'config'
            or name == 'model'
            or name.startswith('model.')
            or name == 'Trainer_x4k'
            or name.startswith('Trainer_x4k.')
        ]
        for name in loaded_names:
            sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
    return model


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
    batch_size = args.batch_size or int(config.get('eval_batch_size', 1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **loader_kwargs(config, device))
    model = load_model(args)

    frame_rows: list[dict[str, Any]] = []
    scene_rows: dict[str, list[dict[str, float]]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            prediction = model.hr_inference(
                texture_to_rgb(batch['texture0']),
                texture_to_rgb(batch['texture1']),
                TTA=False,
                down_scale=args.down_scale,
                timestep=batch['time'].view(batch['time'].shape[0], 1, 1, 1),
            ).mean(dim=1, keepdim=True).clamp(0.0, 1.0)
            target = batch['target'].clamp(0.0, 1.0)
            metric_tensors = compute_main_metrics(prediction, target)

            scenes = _batch_strings(batch['scene'])
            left_ids = _batch_ints(batch['left_id'])
            target_ids = _batch_ints(batch['target_id'])
            right_ids = _batch_ints(batch['right_id'])
            for index in range(int(prediction.shape[0])):
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
        'method': 'SGM-VFI',
        'split': args.split,
        'frames': len(frame_rows),
        'overall': overall,
        'by_scene': scene_summary,
    }
    manifest = {
        **metrics,
        'data_root': str(args.data_root.resolve()),
        'pretrained_root': str(args.pretrained_root.resolve()),
        'exp_name': args.exp_name,
        'num_key_points': args.num_key_points,
        'down_scale': args.down_scale,
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
