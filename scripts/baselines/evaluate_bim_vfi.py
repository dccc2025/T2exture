"""Evaluate official BiM-VFI checkpoints on the T2exture dataset split."""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import resolve_sample_passive_context, validate_runtime_config
from data import TextureDataset
from eval import _batch_ints, _batch_strings, _mean_rows, _move_batch, _relative, _sample_name, _save_png, _state_dict, _torch_load
from metrics import MAIN_METRIC_KEYS, compute_main_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('pretrained/bim-vfi/bim_vfi.pth'))
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='test')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('train.yaml'))
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pyr-level', type=int, default=3)
    parser.add_argument('--feat-channels', type=int, default=32)
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


def load_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    package_root = ROOT / 'third_party' / 'BiM-VFI'
    modules_root = package_root / 'modules'
    components_root = modules_root / 'components'
    bim_root = components_root / 'bim_vfi'
    isolated_names = [
        name
        for name in sys.modules
        if name == 'modules'
        or name.startswith('modules.')
        or name == 'utils'
        or name.startswith('utils.')
    ]
    previous_modules = {name: sys.modules[name] for name in isolated_names}
    for name in isolated_names:
        sys.modules.pop(name, None)

    sys.path.insert(0, str(package_root))
    try:
        modules_pkg = types.ModuleType('modules')
        modules_pkg.__path__ = [str(modules_root)]
        components_pkg = types.ModuleType('modules.components')
        components_pkg.__path__ = [str(components_root)]
        bim_pkg = types.ModuleType('modules.components.bim_vfi')
        bim_pkg.__path__ = [str(bim_root)]
        sys.modules['modules'] = modules_pkg
        sys.modules['modules.components'] = components_pkg
        sys.modules['modules.components.bim_vfi'] = bim_pkg

        registry_spec = importlib.util.spec_from_file_location(
            'modules.components.components',
            components_root / 'components.py',
        )
        if registry_spec is None or registry_spec.loader is None:
            raise RuntimeError('Cannot load BiM-VFI component registry')
        registry_module = importlib.util.module_from_spec(registry_spec)
        sys.modules['modules.components.components'] = registry_module
        registry_spec.loader.exec_module(registry_module)
        components_pkg.register = registry_module.register
        components_pkg.make_components = registry_module.make_components

        model_spec = importlib.util.spec_from_file_location(
            'modules.components.bim_vfi.bim_vfi',
            bim_root / 'bim_vfi.py',
        )
        if model_spec is None or model_spec.loader is None:
            raise RuntimeError('Cannot load BiM-VFI network component')
        module = importlib.util.module_from_spec(model_spec)
        sys.modules['modules.components.bim_vfi.bim_vfi'] = module
        model_spec.loader.exec_module(module)
        model = module.BiMVFI(pyr_level=args.pyr_level, feat_channels=args.feat_channels)
        checkpoint = _torch_load(args.checkpoint, device)
        result = model.load_state_dict(_state_dict(checkpoint), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f'BiM-VFI checkpoint mismatch: missing={len(result.missing_keys)}, '
                f'unexpected={len(result.unexpected_keys)}'
            )
        model = model.to(device).eval()
    finally:
        if sys.path and sys.path[0] == str(package_root):
            sys.path.pop(0)
        loaded_names = [
            name
            for name in sys.modules
            if name == 'modules'
            or name.startswith('modules.')
            or name == 'utils'
            or name.startswith('utils.')
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
    model = load_model(args, device)

    frame_rows: list[dict[str, Any]] = []
    scene_rows: dict[str, list[dict[str, float]]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            output = model(
                img0=texture_to_rgb(batch['texture0']),
                img1=texture_to_rgb(batch['texture1']),
                time_step=batch['time'].view(batch['time'].shape[0], 1, 1, 1),
                pyr_level=args.pyr_level,
                run_with_gt=False,
            )
            prediction = output['imgt_pred'].mean(dim=1, keepdim=True).clamp(0.0, 1.0)
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
        'method': 'BiM-VFI',
        'split': args.split,
        'frames': len(frame_rows),
        'overall': overall,
        'by_scene': scene_summary,
    }
    manifest = {
        **metrics,
        'data_root': str(args.data_root.resolve()),
        'checkpoint': str(args.checkpoint.resolve()),
        'pyr_level': args.pyr_level,
        'feat_channels': args.feat_channels,
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
