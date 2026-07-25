"""Run remaining formal experiments serially.

This is a thin, restartable runner around the commands documented in
``docs/formal_experiment_runbook.yaml``. It does not change experiment
protocols; it only skips completed rows and writes command logs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DATA_ROOT = Path('dataset_roi')
PRETRAINED_AMT_L = Path('pretrained/amt-l.pth')
OUTPUTS_ROOT = Path('outputs/final')
LOG_ROOT = OUTPUTS_ROOT / '_logs'
TABLE06_CONFIG = Path('configs/formal/table06/real-benchmark.yaml')
TABLE06_METHODS = ('ifrnet', 'sgm-vfi', 'bim-vfi', 'gimm-vfi-f', 'amt-l-vanilla', 'ours-l')
TABLE05_ROWS = (
    ('amt-s', 'vanilla', 'amt-s-vanilla'),
    ('amt-s', 't2texture', 'amt-s-t2texture'),
    ('amt-l', 'vanilla', 'amt-l-vanilla'),
    ('amt-l', 't2texture', 'amt-l-t2texture'),
    ('amt-g', 'vanilla', 'amt-g-vanilla'),
    ('amt-g', 't2texture', 'amt-g-t2texture'),
)


def now_stamp() -> str:
    return dt.datetime.now().strftime('%Y%m%d_%H%M%S')


def run_command(command: list[str], log_path: Path) -> None:
    """Run one command, stream output to the console, and persist a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'RUN {" ".join(command)}', flush=True)
    print(f'LOG {log_path}', flush=True)
    with log_path.open('w', encoding='utf-8') as handle:
        handle.write(f'$ {" ".join(command)}\n')
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end='', flush=True)
            handle.write(line)
            handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def metrics_path(exp_dir: Path) -> Path:
    return exp_dir / 'test' / 'metrics.json'


def best_path(exp_dir: Path) -> Path:
    return exp_dir / 'best.pt'


def sync_metrics() -> None:
    run_command(
        [
            PYTHON,
            '-B',
            'scripts/sync_formal_metrics.py',
            '--outputs-root',
            str(OUTPUTS_ROOT),
            '--csv',
            str(OUTPUTS_ROOT / '_summary/metrics_summary.csv'),
            '--markdown',
            str(OUTPUTS_ROOT / '_summary/metrics_summary.md'),
            '--per-table-dir',
            str(OUTPUTS_ROOT / '_summary/by_table'),
        ],
        LOG_ROOT / f'sync_{now_stamp()}.log',
    )


def train_eval_vis(exp_dir: Path, config: Path, method_label: str, log_prefix: str) -> None:
    """Run train if needed, then eval/vis/sync for one T2exture experiment row."""
    if metrics_path(exp_dir).is_file():
        print(f'SKIP {exp_dir}: metrics already exist', flush=True)
        return

    if not best_path(exp_dir).is_file():
        train_cmd = [
            PYTHON,
            '-B',
            'train.py',
            '--data-root',
            str(DATA_ROOT),
            '--pretrained',
            str(PRETRAINED_AMT_L),
            '--backbone',
            'amt-l',
            '--output-dir',
            str(exp_dir),
            '--config',
            str(config),
        ]
        last = exp_dir / 'last.pt'
        if last.is_file():
            train_cmd.extend(['--resume', str(last)])
        run_command(train_cmd, LOG_ROOT / f'{log_prefix}_train_{now_stamp()}.log')

    if not best_path(exp_dir).is_file():
        raise FileNotFoundError(f'Missing best checkpoint after training: {best_path(exp_dir)}')

    run_command(
        [
            PYTHON,
            '-B',
            'eval.py',
            '--data-root',
            str(DATA_ROOT),
            '--pretrained',
            str(PRETRAINED_AMT_L),
            '--backbone',
            'amt-l',
            '--checkpoint',
            str(best_path(exp_dir)),
            '--split',
            'test',
            '--output-dir',
            str(exp_dir / 'test'),
            '--config',
            str(config),
        ],
        LOG_ROOT / f'{log_prefix}_eval_{now_stamp()}.log',
    )
    run_command(
        [
            PYTHON,
            '-B',
            'vis.py',
            '--eval-dir',
            str(exp_dir / 'test'),
            '--output-dir',
            str(exp_dir / 'test/vis'),
            '--method-label',
            method_label,
            '--max-frames-per-scene',
            '5',
        ],
        LOG_ROOT / f'{log_prefix}_vis_{now_stamp()}.log',
    )
    sync_metrics()


def flow_manifest(flow_set: str) -> Path:
    return DATA_ROOT / 'flow' / flow_set / 'flow_manifest.json'


def flow_dir(flow_set: str) -> Path:
    return DATA_ROOT / 'flow' / flow_set


def ensure_flow(active_stride: int) -> None:
    """Generate one Table 4 pseudo-flow set if it is not already present."""
    flow_set = f's{active_stride:02d}'
    if flow_set == 's10' and flow_dir(flow_set).is_dir():
        print('SKIP flow/s10: existing ROI pseudo-flow cache', flush=True)
        return
    if flow_manifest(flow_set).is_file():
        print(f'SKIP flow/{flow_set}: manifest already exists', flush=True)
        return
    run_command(
        [
            PYTHON,
            '-B',
            '-m',
            'flow_generation.generate_liteflownet_flow',
            '--data-root',
            str(DATA_ROOT),
            '--active-stride',
            str(active_stride),
            '--splits',
            'train',
            'valid',
            'test',
            '--device',
            'cuda',
        ],
        LOG_ROOT / f'table04_flow_{flow_set}_{now_stamp()}.log',
    )


def run_table03(start: int, stop: int) -> None:
    for index in range(start, stop + 1):
        config = Path(f'configs/formal/table03/context-per-side-{index:02d}.yaml')
        exp_dir = OUTPUTS_ROOT / 'table03_passive_context' / f'context-per-side-{index:02d}'
        train_eval_vis(exp_dir, config, f'Context-Per-Side-{index:02d}', f'table03_context_{index:02d}')


def run_table04(start_stride: int, stop_stride: int) -> None:
    for stride in range(start_stride, stop_stride + 1):
        ensure_flow(stride)
        config = Path(f'configs/formal/table04/active-stride-{stride:02d}.yaml')
        exp_dir = OUTPUTS_ROOT / 'table04_active_sparsity' / f'active-stride-{stride:02d}'
        label = f'Active-Stride-{stride:02d}'
        train_eval_vis(exp_dir, config, label, f'table04_stride_{stride:02d}')


def run_table05() -> None:
    """Profile deployment metrics for AMT-S/L/G vanilla and T2exture rows."""
    for backbone, setting, row_id in TABLE05_ROWS:
        exp_dir = OUTPUTS_ROOT / 'table05_deployment' / row_id
        if (exp_dir / 'runtime.json').is_file():
            print(f'SKIP {exp_dir}: runtime already exists', flush=True)
            continue
        command = [
            PYTHON,
            '-B',
            'scripts/profile_deployment.py',
            '--backbone',
            backbone,
            '--setting',
            setting,
            '--output-dir',
            str(exp_dir),
            '--config',
            'train.yaml',
            '--data-root',
            str(DATA_ROOT),
        ]
        checkpoint = OUTPUTS_ROOT / 'table02_amt_size' / row_id / 'best.pt'
        if setting == 't2texture' and checkpoint.is_file():
            command.extend(['--checkpoint', str(checkpoint)])
        run_command(command, LOG_ROOT / f'table05_{row_id}_{now_stamp()}.log')
    sync_metrics()


def run_table06() -> None:
    """Evaluate all Table 6 real-benchmark rows."""
    for method in TABLE06_METHODS:
        exp_dir = OUTPUTS_ROOT / 'table06_real_benchmark' / method
        if (exp_dir / 'metrics.json').is_file():
            print(f'SKIP {exp_dir}: metrics already exist', flush=True)
            continue
        run_command(
            [
                PYTHON,
                '-B',
                'scripts/eval_real.py',
                '--method',
                method,
                '--config',
                str(TABLE06_CONFIG),
                '--output-dir',
                str(exp_dir),
                '--device',
                'cuda',
            ],
            LOG_ROOT / f'table06_{method}_{now_stamp()}.log',
        )
        sync_metrics()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table03', action='store_true', help='Run missing Table 3 passive-context rows.')
    parser.add_argument('--table04', action='store_true', help='Run missing Table 4 active-sparsity rows.')
    parser.add_argument('--table05', action='store_true', help='Run missing Table 5 deployment profiling rows.')
    parser.add_argument('--table06', action='store_true', help='Run missing Table 6 real-benchmark rows.')
    parser.add_argument('--table03-start', type=int, default=1)
    parser.add_argument('--table03-stop', type=int, default=10)
    parser.add_argument('--table04-start-stride', type=int, default=2)
    parser.add_argument('--table04-stop-stride', type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_any = args.table03 or args.table04 or args.table05 or args.table06
    if not run_any:
        args.table03 = True
        args.table04 = True
        args.table05 = True
        args.table06 = True
    if args.table03:
        run_table03(args.table03_start, args.table03_stop)
    if args.table04:
        run_table04(args.table04_start_stride, args.table04_stop_stride)
    if args.table05:
        run_table05()
    if args.table06:
        run_table06()
    sync_metrics()


if __name__ == '__main__':
    main()
