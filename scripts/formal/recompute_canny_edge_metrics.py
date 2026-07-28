"""Recompute synthetic Edge-FI@2px from saved predictions using Canny edges."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import _normalise
from metrics.image_metrics import MAIN_METRIC_KEYS, canny_edge_map, edge_f1_from_maps


EDGE_KEY = 'Edge-FI@2px'
CANNY_SETTINGS = {
    'low_threshold': 100,
    'high_threshold': 200,
    'aperture_size': 3,
    'l2_gradient': True,
    'tolerance_px': 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('datasets'))
    parser.add_argument('--outputs-root', type=Path, default=Path('outputs/final'))
    parser.add_argument('--eval-dir', type=Path, action='append', default=None, help='Specific eval directory to update. May be repeated.')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def discover_eval_dirs(outputs_root: Path) -> list[Path]:
    return sorted(path.parent for path in outputs_root.glob('table*/**/test/frame.csv'))


def load_prediction_edges(eval_dir: Path, row: dict[str, str]) -> np.ndarray:
    path = eval_dir / row['pred_path']
    if not path.is_file():
        raise FileNotFoundError(f'Missing saved prediction: {path}')
    image = np.asarray(Image.open(path).convert('L'), dtype=np.uint8)
    return canny_edge_map(image, CANNY_SETTINGS['low_threshold'], CANNY_SETTINGS['high_threshold'], CANNY_SETTINGS['aperture_size'], CANNY_SETTINGS['l2_gradient'])


def load_target_edges(data_root: Path, row: dict[str, str], target_cache: dict[tuple[str, int], np.ndarray]) -> np.ndarray:
    scene = row['scene']
    target_id = int(row['target_id'])
    cache_key = (scene, target_id)
    if cache_key in target_cache:
        return target_cache[cache_key]
    path = data_root / 'sim' / scene / 'texture' / f'{target_id:03d}.npy'
    if not path.is_file():
        raise FileNotFoundError(f'Missing target frame: {path}')
    image = np.round(_normalise(np.load(path).astype(np.float32)) * 255.0).astype(np.uint8)
    target_cache[cache_key] = canny_edge_map(
        image,
        CANNY_SETTINGS['low_threshold'],
        CANNY_SETTINGS['high_threshold'],
        CANNY_SETTINGS['aperture_size'],
        CANNY_SETTINGS['l2_gradient'],
    )
    return target_cache[cache_key]


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(np.mean(values))


def aggregate(rows: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    overall = {key: mean_metric(rows, key) for key in MAIN_METRIC_KEYS}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row['scene'])].append(row)
    scene_rows = [
        {'scene': scene, 'frames': len(scene_group), **{key: mean_metric(scene_group, key) for key in MAIN_METRIC_KEYS}}
        for scene, scene_group in sorted(grouped.items())
    ]
    return overall, scene_rows


def update_metrics_json(path: Path, frame_count: int, overall: dict[str, float], scene_rows: list[dict[str, Any]], dry_run: bool) -> None:
    metrics = read_json(path)
    metrics['frames'] = frame_count
    metrics['overall'] = overall
    metrics['by_scene'] = scene_rows
    metrics.setdefault('metric_settings', {})[EDGE_KEY] = {
        'edge_detector': 'Canny',
        **CANNY_SETTINGS,
        'input_unit': '8-bit grayscale converted from normalized [0,1] images',
    }
    if not dry_run:
        write_json(path, metrics)


def update_manifest_json(path: Path, frame_count: int, overall: dict[str, float], scene_rows: list[dict[str, Any]], dry_run: bool) -> None:
    if not path.is_file():
        return
    manifest = read_json(path)
    manifest['frames'] = frame_count
    manifest['overall'] = overall
    manifest['by_scene'] = scene_rows
    manifest.setdefault('metric_settings', {})[EDGE_KEY] = {
        'edge_detector': 'Canny',
        **CANNY_SETTINGS,
        'input_unit': '8-bit grayscale converted from normalized [0,1] images',
    }
    if not dry_run:
        write_json(path, manifest)


def update_eval_dir(eval_dir: Path, data_root: Path, target_cache: dict[tuple[str, int], np.ndarray], dry_run: bool) -> dict[str, Any]:
    frame_csv = eval_dir / 'frame.csv'
    scene_csv = eval_dir / 'scene.csv'
    metrics_json = eval_dir / 'metrics.json'
    if not metrics_json.is_file():
        raise FileNotFoundError(f'Missing metrics.json next to frame.csv: {metrics_json}')

    fieldnames, rows = read_csv(frame_csv)
    if EDGE_KEY not in fieldnames:
        raise ValueError(f'{frame_csv} does not contain {EDGE_KEY}')
    if 'pred_path' not in fieldnames:
        raise ValueError(f'{frame_csv} does not contain pred_path')

    old_overall = read_json(metrics_json).get('overall', {}).get(EDGE_KEY)
    for row in rows:
        score = edge_f1_from_maps(
            load_prediction_edges(eval_dir, row),
            load_target_edges(data_root, row, target_cache),
            CANNY_SETTINGS['tolerance_px'],
        )
        row[EDGE_KEY] = str(float(score))

    overall, scene_rows = aggregate(rows)
    if not dry_run:
        write_csv(frame_csv, fieldnames, rows)
        write_csv(scene_csv, ['scene', 'frames', *MAIN_METRIC_KEYS], scene_rows)
    update_metrics_json(metrics_json, len(rows), overall, scene_rows, dry_run)
    update_manifest_json(eval_dir / 'manifest.json', len(rows), overall, scene_rows, dry_run)
    return {
        'eval_dir': str(eval_dir),
        'frames': len(rows),
        'old_edge': old_overall,
        'new_edge': overall[EDGE_KEY],
    }


def main() -> None:
    args = parse_args()
    eval_dirs = args.eval_dir or discover_eval_dirs(args.outputs_root)
    target_cache: dict[tuple[str, int], np.ndarray] = {}
    summaries = [update_eval_dir(eval_dir, args.data_root, target_cache, args.dry_run) for eval_dir in eval_dirs]
    print(json.dumps({'updated': len(summaries), 'dry_run': args.dry_run, 'summaries': summaries}, indent=2), flush=True)


if __name__ == '__main__':
    main()
