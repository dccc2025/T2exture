"""Check that the prepared dataset matches the default T2exture setup."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import resolve_sample_passive_context, validate_runtime_config
from data import TextureDataset, read_split


EXPECTED_FRAME_SHAPE = (640, 960)
EXPECTED_FRAMES_PER_SCENE = 180
EXPECTED_SCENES = 32
EXPECTED_S10_FLOW_FILES = 4892
EXPECTED_SPLIT_SCENES = {'train': 20, 'valid': 4, 'test': 8}
EXPECTED_SPLIT_SAMPLES = {'train': 3040, 'valid': 608, 'test': 1216}
EXPECTED_CONFIG = {
    'active_stride': 10,
    'passive_context': 5,
    'sample_passive_context': None,
    'source_off_dir': None,
    'pseudo_flow_dir': 'flow/s10',
    'adapter_iterations': 10_000,
    'finetune_iterations': 5_000,
    'crop_size': 384,
    'batch_size': 4,
    'valid_batch_size': 1,
    'eval_batch_size': 1,
}
EXPECTED_SOURCE_OFF_BACKBONES = ('amt-s', 'amt-l', 'amt-g')
EXPECTED_SOURCE_OFF_FRAMES_PER_SCENE = 18


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Expected a YAML mapping in {path}')
    return data


def ok(message: str) -> None:
    print(f'OK {message}')


def fail(errors: list[str], message: str) -> None:
    errors.append(message)
    print(f'FAIL {message}')


def same(errors: list[str], label: str, got: Any, expected: Any) -> None:
    if got == expected:
        ok(f'{label}: {got}')
    else:
        fail(errors, f'{label}: got {got!r}, expected {expected!r}')


def check_required_paths(data_root: Path, errors: list[str]) -> None:
    required = [
        data_root / 'train.txt',
        data_root / 'valid.txt',
        data_root / 'test.txt',
        data_root / 'dataset_manifest.json',
        data_root / 'sim',
        data_root / 'flow' / 's10',
        data_root / 'real',
    ]
    for path in required:
        if path.exists():
            ok(f'found {path.relative_to(data_root)}')
        else:
            fail(errors, f'missing {path}')


def check_config(data_root: Path, config: dict[str, Any], errors: list[str]) -> None:
    try:
        validate_runtime_config(config, data_root)
        ok('train.yaml passes runtime checks')
    except Exception as exc:
        fail(errors, f'train.yaml runtime check failed: {exc}')

    for key, expected in EXPECTED_CONFIG.items():
        same(errors, f'train.yaml {key}', config.get(key), expected)


def check_manifest(data_root: Path, errors: list[str]) -> None:
    path = data_root / 'dataset_manifest.json'
    if not path.is_file():
        return

    manifest_text = path.read_text(encoding='utf-8-sig')
    manifest = json.loads(manifest_text)
    private_markers = ('\\', 'Users' + '/', 'Users' + '\\', 'App' + 'Data')
    if any(marker in manifest_text for marker in private_markers) or re.search(r'(?<![A-Za-z])[A-Za-z]:[\\/]', manifest_text):
        fail(errors, 'dataset manifest contains local path markers')
    else:
        ok('dataset manifest paths are portable')
    scenes = manifest.get('scenes', {})
    same(errors, 'scene count', len(scenes), EXPECTED_SCENES)
    same(errors, 'frame shape', tuple(manifest.get('output_shape_hw', ())), EXPECTED_FRAME_SHAPE)

    flow_files = sum(int(scene.get('flow_files', 0)) for scene in scenes.values())
    same(errors, 'flow/s10 files in manifest', flow_files, EXPECTED_S10_FLOW_FILES)

    source_off = manifest.get('source_off', {})
    if source_off:
        for backbone in EXPECTED_SOURCE_OFF_BACKBONES:
            entry = source_off.get(backbone, {})
            same(errors, f'{backbone} source_off scene count in manifest', int(entry.get('scene_count', 0)), EXPECTED_SCENES)
            same(errors, f'{backbone} source_off frame count in manifest', int(entry.get('frame_count', 0)), EXPECTED_SCENES * EXPECTED_SOURCE_OFF_FRAMES_PER_SCENE)


def check_source_off(data_root: Path, errors: list[str]) -> None:
    source_off_root = data_root / 'source_off'
    if not source_off_root.is_dir():
        fail(errors, f'missing {source_off_root}')
        return
    for backbone in EXPECTED_SOURCE_OFF_BACKBONES:
        root = source_off_root / backbone
        if not root.is_dir():
            fail(errors, f'missing {root}')
            continue
        manifest = root / 'manifest.json'
        if manifest.is_file():
            ok(f'found source_off/{backbone}/manifest.json')
        else:
            fail(errors, f'missing {manifest}')
        scene_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        same(errors, f'{backbone} source_off scene count', len(scene_dirs), EXPECTED_SCENES)
        total = 0
        for scene_dir in scene_dirs:
            files = sorted(scene_dir.glob('*.npy'))
            total += len(files)
            same(errors, f'{backbone}/{scene_dir.name} source_off frame count', len(files), EXPECTED_SOURCE_OFF_FRAMES_PER_SCENE)
            if files:
                shape = tuple(np.load(files[0]).shape)
                same(errors, f'{backbone}/{scene_dir.name} source_off frame shape', shape, EXPECTED_FRAME_SHAPE)
        same(errors, f'{backbone} source_off total frames', total, EXPECTED_SCENES * EXPECTED_SOURCE_OFF_FRAMES_PER_SCENE)


def check_scene_frames(data_root: Path, scenes: list[str], errors: list[str], require_full_count: bool) -> None:
    for scene in scenes:
        for folder in ('texture', 'passive'):
            frame_dir = data_root / 'sim' / scene / folder
            files = sorted(frame_dir.glob('*.npy'))
            if require_full_count:
                same(errors, f'{scene}/{folder} frame count', len(files), EXPECTED_FRAMES_PER_SCENE)
            elif files:
                ok(f'{scene}/{folder} frame count: {len(files)}')
            else:
                fail(errors, f'{scene}/{folder} has no frames')
            if files:
                shape = tuple(np.load(files[0]).shape)
                same(errors, f'{scene}/{folder} frame shape', shape, EXPECTED_FRAME_SHAPE)


def check_split(data_root: Path, config: dict[str, Any], split: str, errors: list[str], profile: str) -> None:
    split_file = data_root / f'{split}.txt'
    if not split_file.is_file():
        return

    scenes = read_split(split_file)
    if profile == 'full':
        same(errors, f'{split} scene count', len(scenes), EXPECTED_SPLIT_SCENES[split])
    elif scenes:
        ok(f'{split} scene count: {len(scenes)}')
    else:
        fail(errors, f'{split} split has no scenes')
    check_scene_frames(data_root, scenes, errors, require_full_count=profile == 'full')

    passive_context = int(config.get('passive_context', 5))
    sample_context = resolve_sample_passive_context(config, passive_context)
    pseudo_flow_root = Path(config['pseudo_flow_dir']) if config.get('pseudo_flow_dir') else None

    try:
        dataset = TextureDataset(
            data_root,
            split_file,
            passive_context=passive_context,
            pseudo_flow_root=pseudo_flow_root,
            active_stride=int(config.get('active_stride', 10)),
            sample_passive_context=sample_context,
        )
        if profile == 'full':
            same(errors, f'{split} sample count', len(dataset), EXPECTED_SPLIT_SAMPLES[split])
        elif len(dataset) > 0:
            ok(f'{split} sample count: {len(dataset)}')
        else:
            fail(errors, f'{split} has no samples')
        sample = dataset[0]
    except Exception as exc:
        fail(errors, f'could not read {split} samples: {exc}')
        return

    expected_shapes = {
        'texture0': (1, *EXPECTED_FRAME_SHAPE),
        'texture1': (1, *EXPECTED_FRAME_SHAPE),
        'target': (1, *EXPECTED_FRAME_SHAPE),
        'passive_context': (passive_context, *EXPECTED_FRAME_SHAPE),
        'flow': (4, *EXPECTED_FRAME_SHAPE),
    }
    for key, expected in expected_shapes.items():
        same(errors, f'{split} sample {key} shape', tuple(sample[key].shape), expected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('datasets'))
    parser.add_argument('--config', type=Path, default=Path('train.yaml'))
    parser.add_argument('--profile', choices=['full', 'subset'], default='full')
    parser.add_argument('--require-source-off', action='store_true', help='Also require the formal Stage 1 source-off caches for S/L/G.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    config = read_yaml(args.config.resolve())
    errors: list[str] = []

    check_required_paths(data_root, errors)
    check_config(data_root, config, errors)
    check_manifest(data_root, errors)
    if args.require_source_off:
        check_source_off(data_root, errors)
    for split in ('train', 'valid', 'test'):
        check_split(data_root, config, split, errors, args.profile)

    if errors:
        print(f'Preflight failed with {len(errors)} issue(s).')
        raise SystemExit(1)
    print('Preflight passed.')


if __name__ == '__main__':
    main()
