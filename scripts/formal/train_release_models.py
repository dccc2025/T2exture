"""Train, evaluate, and visualize the three release T2exture checkpoints."""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.checkpoint import torch_load_portable

PYTHON = sys.executable
DATA_ROOT = ROOT / 'datasets'
CONFIG = ROOT / 'train.yaml'
OUTPUT_ROOT = ROOT / 'outputs' / 'final' / 'release'
LOG_ROOT = ROOT / 'outputs' / 'final' / '_logs'

RELEASE_MODELS = {
    's': {'backbone': 'amt-s', 'pretrained': Path('pretrained/amt-s.pth'), 'label': 'T2exture-S'},
    'l': {'backbone': 'amt-l', 'pretrained': Path('pretrained/amt-l.pth'), 'label': 'T2exture-L'},
    'g': {'backbone': 'amt-g', 'pretrained': Path('pretrained/amt-g.pth'), 'label': 'T2exture-G'},
}
DEFAULT_VARIANTS = ['s', 'l', 'g']


def now_stamp() -> str:
    return dt.datetime.now().strftime('%Y%m%d_%H%M%S')


def run(command: list[str], log_path: Path) -> None:
    """Run one command while streaming and saving logs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'RUN {" ".join(command)}', flush=True)
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


def training_complete(best: Path) -> bool:
    """Return true only when the checkpoint reached the full release schedule."""
    if not best.is_file():
        return False
    config = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    expected_step = int(config['adapter_iterations']) + int(config['finetune_iterations'])
    try:
        checkpoint = torch_load_portable(best, 'cpu')
    except Exception as exc:
        print(f'RETRAIN {best}: cannot inspect checkpoint ({exc})', flush=True)
        return False
    if not isinstance(checkpoint, dict):
        return False
    return checkpoint.get('phase') == 'finetune' and int(checkpoint.get('global_step', 0)) >= expected_step


def run_one(variant: str) -> None:
    spec = RELEASE_MODELS[variant]
    exp_dir = OUTPUT_ROOT / f't2exture-{variant}'
    best = exp_dir / 'best.pt'
    metrics = exp_dir / 'test' / 'metrics.json'
    log_prefix = f'release_t2exture_{variant}_{now_stamp()}'

    if not training_complete(best):
        command = [
            PYTHON,
            '-B',
            'train.py',
            '--data-root',
            str(DATA_ROOT),
            '--pretrained',
            str(spec['pretrained']),
            '--backbone',
            str(spec['backbone']),
            '--output-dir',
            str(exp_dir),
            '--config',
            str(CONFIG),
        ]
        last = exp_dir / 'last.pt'
        if last.is_file():
            command.extend(['--resume', str(last)])
        run(command, LOG_ROOT / f'{log_prefix}_train.log')
    else:
        print(f'SKIP train {exp_dir}: release checkpoint is complete', flush=True)

    if not metrics.is_file():
        run(
            [
                PYTHON,
                '-B',
                'eval.py',
                '--data-root',
                str(DATA_ROOT),
                '--checkpoint',
                str(best),
                '--backbone',
                str(spec['backbone']),
                '--split',
                'test',
                '--output-dir',
                str(exp_dir / 'test'),
                '--config',
                str(CONFIG),
            ],
            LOG_ROOT / f'{log_prefix}_eval.log',
        )
    else:
        print(f'SKIP eval {exp_dir}: metrics.json exists', flush=True)

    if not (exp_dir / 'test' / 'vis' / 'manifest.json').is_file():
        run(
            [
                PYTHON,
                '-B',
                'vis.py',
                '--eval-dir',
                str(exp_dir / 'test'),
                '--output-dir',
                str(exp_dir / 'test/vis'),
                '--method-label',
                str(spec['label']),
                '--max-frames-per-scene',
                '5',
            ],
            LOG_ROOT / f'{log_prefix}_vis.log',
        )
    else:
        print(f'SKIP vis {exp_dir}: vis manifest exists', flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', choices=DEFAULT_VARIANTS, default=DEFAULT_VARIANTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for variant in args.variants:
        run_one(variant)


if __name__ == '__main__':
    main()
