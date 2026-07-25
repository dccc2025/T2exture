"""Collect formal experiment metrics into table-ready CSV and Markdown files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MAIN_METRIC_KEYS = ('PSNR', 'SSIM', 'Edge-FI@2px', 'IE', 'NIE')
REAL_METRIC_KEYS = ('En', 'AG', 'SF', 'SD', 'SCD', 'PI')
DEPLOYMENT_METRIC_KEYS = ('Latency', 'FLOPs', 'Params', 'Trainable Params')
STATIC_RUNTIME_VALUES = {
    ('table01_prior', 'ifrnet'): {'Params': '5.00M'},
    ('table01_prior', 'sgm-vfi'): {'Params': '20.80M'},
    ('table01_prior', 'bim-vfi'): {'Params': '6.88M'},
    ('table01_prior', 'gimm-vfi-f'): {'Params': '30.61M'},
    ('table01_prior', 'amt-l-vanilla'): {'Params': '12.94M'},
    ('table01_prior', 'ours-l'): {'Params': '13.14M'},
    ('table06_real_benchmark', 'ifrnet'): {'Params': '5.00M'},
    ('table06_real_benchmark', 'sgm-vfi'): {'Params': '20.80M'},
    ('table06_real_benchmark', 'bim-vfi'): {'Params': '6.88M'},
    ('table06_real_benchmark', 'gimm-vfi-f'): {'Params': '30.61M'},
    ('table06_real_benchmark', 'amt-l-vanilla'): {'Params': '12.94M'},
    ('table06_real_benchmark', 'ours-l'): {'Params': '13.14M'},
}
TABLE_ROWS = {
    'table01_prior': [
        ('ifrnet', 'IFRNet [CVPR 2022]'),
        ('sgm-vfi', 'SGM-VFI [CVPR 2024]'),
        ('bim-vfi', 'BiM-VFI [CVPR 2025]'),
        ('gimm-vfi-f', 'GIMM-VFI-F [NeurIPS 2024]'),
        ('amt-l-vanilla', 'AMT-L [CVPR 2023]'),
        ('ours-l', 'Ours-L'),
    ],
    'table02_amt_size': [
        ('amt-s-vanilla', 'AMT-S vanilla'),
        ('amt-s-t2texture', 'AMT-S T2exture'),
        ('amt-l-vanilla', 'AMT-L vanilla'),
        ('amt-l-t2texture', 'AMT-L T2exture'),
        ('amt-g-vanilla', 'AMT-G vanilla'),
        ('amt-g-t2texture', 'AMT-G T2exture'),
    ],
    'table03_passive_context': [
        ('context-per-side-01', '1 passive ref per side'),
        ('context-per-side-02', '2 passive refs per side'),
        ('context-per-side-03', '3 passive refs per side'),
        ('context-per-side-04', '4 passive refs per side'),
        ('context-per-side-05', '5 passive refs per side'),
        ('context-per-side-06', '6 passive refs per side'),
        ('context-per-side-07', '7 passive refs per side'),
        ('context-per-side-08', '8 passive refs per side'),
        ('context-per-side-09', '9 passive refs per side'),
        ('context-per-side-10', '10 passive refs per side'),
    ],
    'table04_active_sparsity': [
        ('active-stride-02', '1 passive frame'),
        ('active-stride-03', '2 passive frames'),
        ('active-stride-04', '3 passive frames'),
        ('active-stride-05', '4 passive frames'),
        ('active-stride-06', '5 passive frames'),
        ('active-stride-07', '6 passive frames'),
        ('active-stride-08', '7 passive frames'),
        ('active-stride-09', '8 passive frames'),
        ('active-stride-10', '9 passive frames'),
        ('active-stride-11', '10 passive frames'),
    ],
    'table05_deployment': [
        ('amt-s-vanilla', 'AMT-S vanilla'),
        ('amt-s-t2texture', 'AMT-S T2exture'),
        ('amt-l-vanilla', 'AMT-L vanilla'),
        ('amt-l-t2texture', 'AMT-L T2exture'),
        ('amt-g-vanilla', 'AMT-G vanilla'),
        ('amt-g-t2texture', 'AMT-G T2exture'),
    ],
    'table06_real_benchmark': [
        ('ifrnet', 'IFRNet [CVPR 2022]'),
        ('sgm-vfi', 'SGM-VFI [CVPR 2024]'),
        ('bim-vfi', 'BiM-VFI [CVPR 2025]'),
        ('gimm-vfi-f', 'GIMM-VFI-F [NeurIPS 2024]'),
        ('amt-l-vanilla', 'AMT-L [CVPR 2023]'),
        ('ours-l', 'Ours-L'),
    ],
}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def find_metrics(exp_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    candidates = [
        exp_dir / 'test' / 'metrics.json',
        exp_dir / 'metrics.json',
    ]
    for path in candidates:
        metrics = read_json(path)
        if metrics is not None:
            return metrics, path
    return None, None


def fallback_metrics_path(outputs_root: Path, table_id: str, exp_id: str) -> Path | None:
    """Return a metric source shared by another formal table when appropriate."""
    if table_id == 'table05_deployment':
        return outputs_root / 'table02_amt_size' / exp_id / 'test' / 'metrics.json'
    return None


def find_runtime(exp_dir: Path) -> tuple[dict[str, Any] | None, Path | None]:
    candidates = [
        exp_dir / 'runtime.json',
        exp_dir / 'test' / 'runtime.json',
    ]
    for path in candidates:
        runtime = read_json(path)
        if runtime is not None:
            return runtime, path
    return None, None


def format_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


def collect_rows(outputs_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for table_id, entries in TABLE_ROWS.items():
        for exp_id, label in entries:
            exp_dir = outputs_root / table_id / exp_id
            metrics, metrics_path = find_metrics(exp_dir)
            if metrics is None:
                fallback_path = fallback_metrics_path(outputs_root, table_id, exp_id)
                if fallback_path is not None:
                    metrics = read_json(fallback_path)
                    metrics_path = fallback_path if metrics is not None else None
            runtime, runtime_path = find_runtime(exp_dir)
            overall = metrics.get('overall', {}) if metrics else {}
            runtime_values = runtime.get('overall', runtime) if runtime else {}
            runtime_values = {**STATIC_RUNTIME_VALUES.get((table_id, exp_id), {}), **runtime_values}
            if table_id == 'table05_deployment':
                done = metrics is not None and runtime is not None
            else:
                done = metrics is not None or runtime is not None
            row = {
                'table': table_id,
                'experiment': exp_id,
                'label': label,
                'status': 'done' if done else 'missing',
                'metrics_path': str(metrics_path.relative_to(outputs_root)).replace('\\', '/') if metrics_path else '',
                'runtime_path': str(runtime_path.relative_to(outputs_root)).replace('\\', '/') if runtime_path else '',
            }
            for key in MAIN_METRIC_KEYS:
                row[key] = format_value(overall.get(key))
            for key in REAL_METRIC_KEYS:
                row[key] = format_value(overall.get(key))
            for key in DEPLOYMENT_METRIC_KEYS:
                row[key] = format_value(runtime_values.get(key))
            rows.append(row)
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['table', 'experiment', 'label', 'status', *MAIN_METRIC_KEYS, *REAL_METRIC_KEYS, *DEPLOYMENT_METRIC_KEYS, 'metrics_path', 'runtime_path']
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_by_table(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row['table'], []).append(row)
    return grouped


def markdown_table(rows: list[dict[str, str]]) -> str:
    lines = [
        '| Table | Experiment | Status | PSNR | SSIM | Edge-FI@2px | IE | NIE | En | AG | SF | SD | SCD | PI | Latency | FLOPs | Params | Trainable Params | Metrics | Runtime |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|',
    ]
    for row in rows:
        lines.append(
            f"| {row['table']} | {row['label']} | {row['status']} | {row['PSNR']} | {row['SSIM']} | {row['Edge-FI@2px']} | {row['IE']} | {row['NIE']} | {row['En']} | {row['AG']} | {row['SF']} | {row['SD']} | {row['SCD']} | {row['PI']} | {row['Latency']} | {row['FLOPs']} | {row['Params']} | {row['Trainable Params']} | {row['metrics_path']} | {row['runtime_path']} |"
        )
    return '\n'.join(lines) + '\n'


def write_per_table_outputs(rows: list[dict[str, str]], output_dir: Path) -> dict[str, dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, int]] = {}
    for table_id, table_rows in rows_by_table(rows).items():
        write_csv(table_rows, output_dir / f'{table_id}.csv')
        (output_dir / f'{table_id}.md').write_text(markdown_table(table_rows), encoding='utf-8')
        done = sum(1 for row in table_rows if row['status'] == 'done')
        summary[table_id] = {'rows': len(table_rows), 'done': done, 'missing': len(table_rows) - done}
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs-root', type=Path, default=Path('outputs/final'))
    parser.add_argument('--csv', type=Path, default=Path('outputs/final/_summary/metrics_summary.csv'))
    parser.add_argument('--markdown', type=Path, default=Path('outputs/final/_summary/metrics_summary.md'))
    parser.add_argument('--per-table-dir', type=Path, default=Path('outputs/final/_summary/by_table'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.outputs_root)
    write_csv(rows, args.csv)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown_table(rows), encoding='utf-8')
    by_table = write_per_table_outputs(rows, args.per_table_dir)
    done = sum(1 for row in rows if row['status'] == 'done')
    print(json.dumps({'rows': len(rows), 'done': done, 'missing': len(rows) - done, 'by_table': by_table}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
