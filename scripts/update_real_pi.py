"""Backfill NIQE, NRQM, and PI for Table 6 real-benchmark predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics import REAL_METRIC_KEYS, mean_real_metrics

PI_COMPONENT_KEYS = ('NIQE', 'NRQM')
PI_FIELD_KEYS = ('En', 'AG', 'SF', 'SD', 'SCD', 'NIQE', 'NRQM', 'PI')
DEFAULT_METHODS = ('ifrnet', 'sgm-vfi', 'bim-vfi', 'gimm-vfi-f', 'amt-l-vanilla', 'ours-l')


def parse_args() -> argparse.Namespace:
    """Read Table 6 output location and PI evaluation controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('outputs/final/table06_real_benchmark'))
    parser.add_argument('--methods', nargs='*', default=list(DEFAULT_METHODS))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--force', action='store_true', help='Recompute existing NIQE/NRQM/PI values.')
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read one CSV file into dictionaries."""
    if not path.is_file():
        raise FileNotFoundError(f'Missing CSV: {path}')
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows with stable field order while preserving extra keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = sorted({key for row in rows for key in row} - set(fieldnames))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extra])
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object or return an empty mapping."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON artifact."""
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def load_images(paths: list[Path], device: torch.device) -> torch.Tensor:
    """Load grayscale PNGs as RGB tensors in [0, 1]."""
    tensors = []
    for path in paths:
        image = Image.open(path).convert('RGB')
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(tensors).to(device)


def mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    """Average one numeric row key, preserving None if all rows are missing."""
    values = [float(row[key]) for row in rows if row.get(key) not in (None, '')]
    return float(np.mean(values)) if values else None


def sequence_summary(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Summarize one real sequence, including PI component means."""
    summary = mean_real_metrics([{key: row.get(key) for key in REAL_METRIC_KEYS} for row in rows])
    for key in PI_COMPONENT_KEYS:
        summary[key] = mean_numeric(rows, key)
    return summary


def compute_pi_rows(
    rows: list[dict[str, Any]],
    sequence_dir: Path,
    pi_metric,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> list[dict[str, Any]]:
    """Fill NIQE, NRQM, and PI in one sequence frame table."""
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    pending = [row for row in rows if force or row.get('PI') in (None, '') or row.get('NIQE') in (None, '') or row.get('NRQM') in (None, '')]
    for start in range(0, len(pending), batch_size):
        batch_rows = pending[start:start + batch_size]
        paths = [sequence_dir / str(row['pred_path']) for row in batch_rows]
        with torch.no_grad():
            images = load_images(paths, device)
            niqe_values = pi_metric.net.niqe(images).detach().float().cpu().flatten().tolist()
            nrqm_values = pi_metric.net.nrqm(images).detach().float().cpu().flatten().tolist()
        for row, niqe_value, nrqm_value in zip(batch_rows, niqe_values, nrqm_values):
            pi_value = (float(niqe_value) + (10.0 - float(nrqm_value))) / 2.0
            row['NIQE'] = float(niqe_value)
            row['NRQM'] = float(nrqm_value)
            row['PI'] = pi_value
    return rows


def update_sequence(
    sequence_dir: Path,
    pi_metric,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    """Update one sequence's frame.csv, metrics.json, and manifest.json."""
    frame_csv = sequence_dir / 'frame.csv'
    rows = compute_pi_rows(read_csv(frame_csv), sequence_dir, pi_metric, device, batch_size, force)
    summary = sequence_summary(rows)
    write_csv(
        frame_csv,
        rows,
        ['sequence', 'left_id', 'target_id', 'right_id', 'time', 'passive_context_ids', *PI_FIELD_KEYS, 'pred_path'],
    )

    metrics = read_json(sequence_dir / 'metrics.json')
    metrics.update({'sequence': sequence_dir.name, 'frames': len(rows), 'overall': summary})
    write_json(sequence_dir / 'metrics.json', metrics)

    manifest = read_json(sequence_dir / 'manifest.json')
    manifest.update(
        {
            'pi_formula': 'PI = (NIQE + (10 - NRQM)) / 2',
            'pi_source': 'pyiqa PI metric on saved grayscale predictions replicated to RGB; NIQE and NRQM are taken from the same PI module settings',
            'pi_settings': {'crop_border': 4, 'color_space': 'ycbcr'},
            'pi_note': 'Lower PI is better.',
        }
    )
    write_json(sequence_dir / 'manifest.json', manifest)
    return rows, summary


def update_method(method_dir: Path, pi_metric, device: torch.device, batch_size: int, force: bool) -> None:
    """Update all sequence and method-level artifacts for one Table 6 method."""
    sequence_dirs = sorted(path for path in method_dir.iterdir() if path.is_dir() and (path / 'frame.csv').is_file())
    if not sequence_dirs:
        raise FileNotFoundError(f'No sequence frame.csv files found under {method_dir}')

    method_frame_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    for index, sequence_dir in enumerate(sequence_dirs, start=1):
        rows, summary = update_sequence(sequence_dir, pi_metric, device, batch_size, force)
        method_frame_rows.extend(rows)
        sequence_rows.append({'sequence': sequence_dir.name, 'frames': len(rows), **summary})
        print(
            json.dumps(
                {
                    'method': method_dir.name,
                    'sequence': sequence_dir.name,
                    'progress': f'{index}/{len(sequence_dirs)}',
                    'overall': summary,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    overall = sequence_summary(sequence_rows)
    write_csv(method_dir / 'frame.csv', method_frame_rows, ['sequence', 'left_id', 'target_id', 'right_id', 'time', 'passive_context_ids', *PI_FIELD_KEYS, 'pred_path'])
    write_csv(method_dir / 'sequence.csv', sequence_rows, ['sequence', 'frames', *PI_FIELD_KEYS])

    metrics = read_json(method_dir / 'metrics.json')
    metrics.update({'frames': len(method_frame_rows), 'sequences': len(sequence_rows), 'overall': overall, 'by_sequence': sequence_rows})
    write_json(method_dir / 'metrics.json', metrics)

    manifest = read_json(method_dir / 'manifest.json')
    manifest.update(
        {
            'pi_formula': 'PI = (NIQE + (10 - NRQM)) / 2',
            'pi_source': 'pyiqa PI metric on saved grayscale predictions replicated to RGB; NIQE and NRQM are taken from the same PI module settings',
            'pi_settings': {'crop_border': 4, 'color_space': 'ycbcr'},
            'pi_note': 'Lower PI is better.',
        }
    )
    write_json(method_dir / 'manifest.json', manifest)
    print(json.dumps({'method': method_dir.name, 'overall': overall}, ensure_ascii=False), flush=True)


def main() -> None:
    """Backfill PI artifacts in-place."""
    args = parse_args()
    try:
        import pyiqa
    except ImportError as exc:
        raise ImportError('PI backfill requires pyiqa. Install with: python -m pip install pyiqa==0.1.16 --no-deps') from exc

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    pi_metric = pyiqa.create_metric('pi', device=device)
    for method in args.methods:
        update_method(args.output_root / method, pi_metric, device, args.batch_size, args.force)


if __name__ == '__main__':
    main()
