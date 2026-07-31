"""Train and evaluate the T2exture-S/L/G variants."""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.checkpoint import torch_load_portable


PYTHON = sys.executable
MODELS = {
    's': {'backbone': 'amt-s', 'pretrained': 'amt-s.pth'},
    'l': {'backbone': 'amt-l', 'pretrained': 'amt-l.pth'},
    'g': {'backbone': 'amt-g', 'pretrained': 'amt-g.pth'},
}
DEFAULT_VARIANTS = ['s', 'l', 'g']
EXPECTED_ARCHITECTURE = 'main_figure_centered_context_v3'


def now_stamp() -> str:
    return dt.datetime.now().strftime('%Y%m%d_%H%M%S')


def run(command: list[str], log_path: Path) -> None:
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


def checkpoint_matches(checkpoint_path: Path, config: dict, expected_backbone: str) -> dict | None:
    if not checkpoint_path.is_file():
        return None
    try:
        checkpoint = torch_load_portable(checkpoint_path, 'cpu')
    except Exception as exc:
        print(f'RETRAIN {checkpoint_path}: cannot inspect checkpoint ({exc})', flush=True)
        return None
    if not isinstance(checkpoint, dict):
        return None
    checkpoint_config = checkpoint.get('config', {})
    if checkpoint.get('architecture') != EXPECTED_ARCHITECTURE:
        return None
    checkpoint_backbone = checkpoint.get('backbone') or checkpoint_config.get('backbone')
    if checkpoint_backbone != expected_backbone:
        return None
    if int(checkpoint_config.get('passive_context', 0)) != int(config.get('passive_context', 0)):
        return None
    return checkpoint


def training_complete(best: Path, last: Path, config_path: Path, expected_backbone: str) -> bool:
    if not best.is_file() or not last.is_file():
        return False
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    expected_step = int(config['adapter_iterations']) + int(config['finetune_iterations'])
    best_checkpoint = checkpoint_matches(best, config, expected_backbone)
    last_checkpoint = checkpoint_matches(last, config, expected_backbone)
    if best_checkpoint is None or last_checkpoint is None:
        return False
    return (
        last_checkpoint.get('phase') == 'finetune'
        and int(last_checkpoint.get('global_step', 0)) >= expected_step
    )


def can_resume(checkpoint_path: Path, config_path: Path, expected_backbone: str) -> bool:
    if not checkpoint_path.is_file():
        return False
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    return checkpoint_matches(checkpoint_path, config, expected_backbone) is not None


def variant_source_off_root(source_off_root: Path | None, backbone: str) -> Path | None:
    if source_off_root is None:
        return None
    known_backbones = {str(spec['backbone']) for spec in MODELS.values()}
    if source_off_root.name in known_backbones and source_off_root.name != backbone:
        raise ValueError(f'--source-off-root points to {source_off_root.name}, but variant expects {backbone}')
    candidate = source_off_root / backbone
    if not candidate.is_dir() and any((source_off_root / item).is_dir() for item in known_backbones):
        raise FileNotFoundError(f'Missing source-off cache for {backbone}: {candidate}')
    return candidate if candidate.is_dir() else source_off_root


def resolve_amt_checkpoint(filename: str) -> Path:
    """Resolve an AMT initializer from the documented or legacy local layout."""
    candidates = [
        ROOT / 'pretrained' / 't2exture_model' / filename,
        ROOT / 'pretrained' / filename,
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        checked = ', '.join(str(candidate.relative_to(ROOT)) for candidate in candidates)
        raise FileNotFoundError(f'Missing AMT initializer {filename}. Checked: {checked}')
    return path


def evaluation_complete(metrics: Path, checkpoint: Path) -> bool:
    if not metrics.is_file() or not checkpoint.is_file():
        return False
    return metrics.stat().st_mtime >= checkpoint.stat().st_mtime


def run_one(variant: str, data_root: Path, config_path: Path, output_root: Path, log_root: Path, source_off_root: Path | None) -> None:
    spec = MODELS[variant]
    exp_dir = output_root / f't2exture-{variant}'
    best = exp_dir / 'best.pt'
    last = exp_dir / 'last.pt'
    metrics = exp_dir / 'test' / 'metrics.json'
    log_prefix = f't2exture_{variant}_{now_stamp()}'
    stage1_root = variant_source_off_root(source_off_root, str(spec['backbone']))

    backbone = str(spec['backbone'])
    if not training_complete(best, last, config_path, backbone):
        pretrained = resolve_amt_checkpoint(str(spec['pretrained']))
        command = [
            PYTHON,
            '-B',
            'stage2.py',
            '--data-root',
            str(data_root),
            '--pretrained',
            str(pretrained),
            '--backbone',
            backbone,
            '--output-dir',
            str(exp_dir),
            '--config',
            str(config_path),
        ]
        if stage1_root is not None:
            command.extend(['--source-off-root', str(stage1_root)])
        if can_resume(last, config_path, backbone):
            command.extend(['--resume', str(last)])
        run(command, log_root / f'{log_prefix}_train.log')
    else:
        print(f'SKIP train {exp_dir}: checkpoint is complete', flush=True)

    if not evaluation_complete(metrics, best):
        command = [
            PYTHON,
            '-B',
            'eval.py',
            '--data-root',
            str(data_root),
            '--checkpoint',
            str(best),
            '--backbone',
            backbone,
            '--split',
            'test',
            '--output-dir',
            str(exp_dir / 'test'),
            '--config',
            str(config_path),
        ]
        if stage1_root is not None:
            command.extend(['--source-off-root', str(stage1_root)])
        run(command, log_root / f'{log_prefix}_eval.log')
    else:
        print(f'SKIP eval {exp_dir}: metrics.json exists', flush=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', choices=DEFAULT_VARIANTS, default=DEFAULT_VARIANTS)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'datasets')
    parser.add_argument('--config', type=Path, default=ROOT / 'train.yaml')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs' / 'final' / 'ours')
    parser.add_argument('--log-root', type=Path, default=ROOT / 'outputs' / 'final' / '_logs')
    parser.add_argument('--source-off-root', type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for variant in args.variants:
        run_one(
            variant,
            args.data_root.resolve(),
            args.config.resolve(),
            args.output_root.resolve(),
            args.log_root.resolve(),
            args.source_off_root.resolve() if args.source_off_root is not None else None,
        )


if __name__ == '__main__':
    main()
