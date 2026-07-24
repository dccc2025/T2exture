"""Preflight checks before launching formal T2exture experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import resolve_sample_passive_context, validate_runtime_config
from data import TextureDataset, read_split


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Expected a YAML mapping in {path}')
    return data


def fail(errors: list[str], message: str) -> None:
    errors.append(message)
    print(f'FAIL {message}')


def ok(message: str) -> None:
    print(f'OK {message}')


def check_required_paths(workspace: Path, runbook: dict[str, Any], errors: list[str]) -> None:
    preflight = runbook.get('preflight', {})
    required_paths = preflight.get('required_paths', [])
    missing_expected = preflight.get('currently_missing_expected', [])
    if missing_expected:
        fail(errors, f'runbook still lists expected missing paths: {missing_expected}')
    else:
        ok('runbook currently_missing_expected is empty')

    missing = 0
    for item in required_paths:
        path = workspace / str(item)
        if not path.exists():
            missing += 1
            fail(errors, f'missing required path: {item}')
    if missing == 0:
        ok(f'checked {len(required_paths)} required paths')


def check_default_protocol(config: dict[str, Any], runbook: dict[str, Any], errors: list[str]) -> None:
    protocol = runbook.get('default_protocol', {})
    mismatches = 0
    scalar_keys = [
        'active_stride',
        'passive_context',
        'sample_passive_context',
        'pseudo_flow_dir',
        'crop_size',
        'adapter_iterations',
        'finetune_iterations',
        'batch_size',
        'valid_batch_size',
        'eval_batch_size',
        'num_workers',
        'pin_memory',
        'persistent_workers',
        'prefetch_factor',
        'valid_interval',
    ]
    for key in scalar_keys:
        if key in protocol and config.get(key) != protocol.get(key):
            mismatches += 1
            fail(errors, f'train.yaml {key}={config.get(key)!r} differs from runbook {protocol.get(key)!r}')

    for key, value in protocol.get('loss', {}).items():
        config_loss = (config.get('loss') or {}).get(key)
        if config_loss != value:
            mismatches += 1
            fail(errors, f'train.yaml loss.{key}={config_loss!r} differs from runbook {value!r}')
    if mismatches == 0:
        ok('train.yaml matches runbook default_protocol')


def check_roi_manifest(workspace: Path, runbook: dict[str, Any], errors: list[str]) -> None:
    data_root = workspace / str(runbook.get('formal_data_root', 'dataset_roi'))
    preflight = runbook.get('preflight', {})
    manifest_path = data_root / 'roi_manifest.json'
    if not manifest_path.is_file():
        fail(errors, f'missing ROI manifest: {manifest_path}')
        return

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    scenes = manifest.get('scenes', {})
    expected_scene_count = preflight.get('expected_roi_scene_count')
    if expected_scene_count is not None and len(scenes) != int(expected_scene_count):
        fail(errors, f'ROI manifest has {len(scenes)} scenes, expected {expected_scene_count}')
    else:
        ok(f'ROI manifest scene count: {len(scenes)}')

    expected_shape = preflight.get('expected_roi_shape_hw')
    if expected_shape is not None and manifest.get('output_shape_hw') != list(expected_shape):
        fail(errors, f'ROI manifest shape is {manifest.get("output_shape_hw")}, expected {expected_shape}')
    else:
        ok(f'ROI manifest shape: {manifest.get("output_shape_hw")}')

    total_flow_files = sum(int(scene.get('flow_files', 0)) for scene in scenes.values())
    expected_flow_files = preflight.get('expected_s10_flow_files')
    if expected_flow_files is not None and total_flow_files != int(expected_flow_files):
        fail(errors, f'ROI manifest lists {total_flow_files} s10 flow files, expected {expected_flow_files}')
    else:
        ok(f'ROI manifest s10 flow files: {total_flow_files}')

    if scenes:
        max_bbox_h = max(int(scene['bbox_shape_hw'][0]) for scene in scenes.values())
        max_bbox_w = max(int(scene['bbox_shape_hw'][1]) for scene in scenes.values())
        expected_max_bbox = preflight.get('expected_max_texture_bbox_hw')
        if expected_max_bbox is not None and [max_bbox_h, max_bbox_w] != list(expected_max_bbox):
            fail(errors, f'ROI max bbox is {[max_bbox_h, max_bbox_w]}, expected {expected_max_bbox}')
        else:
            ok(f'ROI max texture bbox HxW: {max_bbox_h}x{max_bbox_w}')

    preview_count = len(list((data_root / 'roi_preview').glob('*.png')))
    expected_preview_count = preflight.get('expected_roi_preview_count')
    if expected_preview_count is not None and preview_count != int(expected_preview_count):
        fail(errors, f'ROI preview count is {preview_count}, expected {expected_preview_count}')
    else:
        ok(f'ROI preview count: {preview_count}')


def check_dataset(workspace: Path, runbook: dict[str, Any], config: dict[str, Any], errors: list[str]) -> None:
    data_root = workspace / str(runbook.get('formal_data_root', 'dataset_roi'))
    validate_runtime_config(config, data_root)
    ok(f'formal data root accepted by runtime guard: {data_root.name}')

    preflight = runbook.get('preflight', {})
    expected_split_counts = preflight.get('expected_split_counts', {})
    expected_sample_counts = preflight.get('expected_sample_counts', {})
    expected_shape = tuple(preflight.get('expected_roi_shape_hw', [640, 960]))
    expected_frames = int(preflight.get('expected_frames_per_scene', 180))

    passive_context = int(config.get('passive_context', 4))
    sample_passive_context = resolve_sample_passive_context(config, passive_context)
    active_stride = int(config.get('active_stride', 10))
    pseudo_flow_root = Path(config['pseudo_flow_dir']) if config.get('pseudo_flow_dir') else None

    for split in ('train', 'valid', 'test'):
        split_file = data_root / f'{split}.txt'
        scenes = read_split(split_file)
        expected_scene_count = expected_split_counts.get(split)
        if expected_scene_count is not None and len(scenes) != int(expected_scene_count):
            fail(errors, f'{split}.txt has {len(scenes)} scenes, expected {expected_scene_count}')
        else:
            ok(f'{split}.txt scene count: {len(scenes)}')

        for scene in scenes:
            for modality in ('texture', 'passive'):
                frame_dir = data_root / 'sim' / scene / modality
                files = sorted(frame_dir.glob('*.npy'))
                if len(files) != expected_frames:
                    fail(errors, f'{scene}/{modality} has {len(files)} frames, expected {expected_frames}')
                    continue
                shape = np.load(files[0]).shape
                if shape != expected_shape:
                    fail(errors, f'{scene}/{modality}/001.npy shape is {shape}, expected {expected_shape}')

        dataset = TextureDataset(
            data_root,
            split_file,
            passive_context=passive_context,
            pseudo_flow_root=pseudo_flow_root,
            active_stride=active_stride,
            sample_passive_context=sample_passive_context,
        )
        expected_samples = expected_sample_counts.get(split)
        if expected_samples is not None and len(dataset) != int(expected_samples):
            fail(errors, f'{split} dataset has {len(dataset)} samples, expected {expected_samples}')
        else:
            ok(f'{split} dataset samples: {len(dataset)}')

        sample = dataset[0]
        expected_passive_shape = (passive_context, *expected_shape)
        checks = {
            'texture0': tuple(sample['texture0'].shape) == (1, *expected_shape),
            'texture1': tuple(sample['texture1'].shape) == (1, *expected_shape),
            'target': tuple(sample['target'].shape) == (1, *expected_shape),
            'passive_context': tuple(sample['passive_context'].shape) == expected_passive_shape,
            'flow': tuple(sample['flow'].shape) == (4, *expected_shape),
        }
        for name, passed in checks.items():
            if not passed:
                fail(errors, f'{split} sample {name} shape check failed')
        if all(checks.values()):
            ok(f'{split} sample tensors match ROI shape')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    parser.add_argument('--runbook', type=Path, default=Path('docs/formal_experiment_runbook.yaml'))
    parser.add_argument('--config', type=Path, default=Path('train.yaml'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    runbook_path = args.runbook if args.runbook.is_absolute() else workspace / args.runbook
    config_path = args.config if args.config.is_absolute() else workspace / args.config
    errors: list[str] = []

    runbook = load_yaml(runbook_path)
    config = load_yaml(config_path)
    check_required_paths(workspace, runbook, errors)
    check_default_protocol(config, runbook, errors)
    check_roi_manifest(workspace, runbook, errors)
    check_dataset(workspace, runbook, config, errors)

    if errors:
        print(f'Preflight failed with {len(errors)} issue(s).')
        raise SystemExit(1)
    print('Preflight passed.')


if __name__ == '__main__':
    main()
