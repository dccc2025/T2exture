"""Train one Stage 2 T2exture backbone with the fixed three-term objective."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import resolve_sample_passive_context, validate_runtime_config
from data import TextureDataset
from losses.loss import CompositeLoss
from model import build_t2texture_model
from utils.checkpoint import torch_load_portable


ARCHITECTURE_VERSION = 'main_figure_centered_context_v3'


def parse_args() -> argparse.Namespace:
    """Read only user-facing paths and an optional training configuration path."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--backbone', choices=['amt-s', 'amt-l', 'amt-g'], default='amt-l')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('train.yaml'))
    parser.add_argument('--source-off-root', type=Path, default=None)
    parser.add_argument('--resume', type=Path, default=None)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Make dataset sampling and model initialization repeatable when possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(model: torch.nn.Module, base_lr: float, decay: float) -> torch.optim.Optimizer:
    """Create parameter groups with lower rates for earlier AMT backbone layers."""
    groups = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        depth = 0 if not name.startswith('backbone.') else name.count('.')
        groups.append({'params': [parameter], 'lr': base_lr * (decay ** depth)})
    return torch.optim.AdamW(groups, weight_decay=1e-5)


def infinite_loader(loader: DataLoader):
    """Yield fresh batches forever without caching previous epochs."""
    while True:
        for batch in loader:
            yield batch


def loader_kwargs(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Return DataLoader performance options that are safe on Windows."""
    workers = int(config.get('num_workers', 0))
    kwargs: dict[str, Any] = {
        'num_workers': workers,
        'pin_memory': bool(config.get('pin_memory', False)) and device.type == 'cuda',
    }
    if workers > 0:
        kwargs['persistent_workers'] = bool(config.get('persistent_workers', False))
        kwargs['prefetch_factor'] = int(config.get('prefetch_factor', 2))
    return kwargs


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values to the training device without touching metadata fields."""
    non_blocking = device.type == 'cuda'
    return {key: value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value for key, value in batch.items()}


def batch_psnr(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Return one PSNR value per sample for tensors in [0, 1]."""
    mse = (prediction - target).pow(2).flatten(1).mean(dim=1).clamp_min(eps)
    return -10.0 * torch.log10(mse)


def resolve_data_path(root: Path, value: str | Path) -> Path:
    """Resolve a config path relative to the dataset root unless it is absolute."""
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == root.name:
        return root.parent.joinpath(*path.parts)
    return root / path


def portable_data_path(root: Path, path: Path | None) -> str | None:
    """Return a dataset-relative path for public run configs when possible."""
    if path is None:
        return None
    if not path.is_absolute():
        parts = path.parts
        if parts and parts[0] == root.name:
            return str(Path(*parts[1:])).replace('\\', '/')
        return str(path).replace('\\', '/')
    try:
        return str(path.resolve().relative_to(root.resolve())).replace('\\', '/')
    except ValueError:
        return path.name


def public_run_config(config: dict[str, Any], data_root: Path, pseudo_flow_root: Path | None, source_off_root: Path | None) -> dict[str, Any]:
    """Copy a run config while keeping dataset-dependent paths portable."""
    result = dict(config)
    result['pseudo_flow_dir'] = portable_data_path(data_root, pseudo_flow_root)
    result['source_off_dir'] = portable_data_path(data_root, source_off_root)
    return result


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint and require the training checkpoint dictionary format."""
    checkpoint = torch_load_portable(path, device)
    if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
        raise ValueError(f'Expected a training checkpoint with a model state dict: {path}')
    return checkpoint


def checkpoint_payload(
    model: torch.nn.Module,
    config: dict[str, Any],
    backbone: str,
    global_step: int,
    phase: str,
    best_psnr: float,
    step: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Build the checkpoint dictionary used by best.pt and last.pt."""
    payload: dict[str, Any] = {
        'model': model.state_dict(),
        'config': config,
        'backbone': backbone,
        'architecture': ARCHITECTURE_VERSION,
        'global_step': global_step,
        'phase': phase,
        'best_psnr': best_psnr,
    }
    if step is not None:
        payload['step'] = step
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    return payload


@torch.no_grad()
def validate(model: torch.nn.Module, criterion: CompositeLoss, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Evaluate validation loss and PSNR on the full validation split."""
    model.eval()
    totals = {'total': 0.0, 'charbonnier': 0.0, 'css': 0.0, 'flow': 0.0, 'psnr': 0.0}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        result = model(batch['texture0'], batch['texture1'], batch['time'], batch['passive_context'], return_flow=True)
        losses = criterion(result['prediction'], batch['target'], result['flow0_pred'], result['flow1_pred'], batch['flow'])
        batch_size = int(batch['target'].shape[0])
        for key in ('total', 'charbonnier', 'css', 'flow'):
            totals[key] += float(losses[key].detach().cpu()) * batch_size
        totals['psnr'] += float(batch_psnr(result['prediction'], batch['target']).mean().detach().cpu()) * batch_size
        count += batch_size
    if count == 0:
        raise ValueError('Validation loader is empty')
    return {f'valid_{key}': value / count for key, value in totals.items()}


def main() -> None:
    """Run adapter-only training followed by whole-model layerwise fine-tuning."""
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.pretrained = args.pretrained.resolve()
    args.output_dir = args.output_dir.resolve()
    args.config = args.config.resolve()
    if args.source_off_root is not None:
        args.source_off_root = resolve_data_path(args.data_root, args.source_off_root)
    if args.resume is not None:
        args.resume = args.resume.resolve()
    config = yaml.safe_load(args.config.read_text())
    validate_runtime_config(config, args.data_root)
    pseudo_flow_root = resolve_data_path(args.data_root, config['pseudo_flow_dir']) if config.get('pseudo_flow_dir') else None
    source_off_root = args.source_off_root or (resolve_data_path(args.data_root, config['source_off_dir']) if config.get('source_off_dir') else None)
    run_config = {
        **public_run_config(config, args.data_root, pseudo_flow_root, source_off_root),
        'backbone': args.backbone,
        'architecture': ARCHITECTURE_VERSION,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'config.json').write_text(json.dumps(run_config, indent=2) + '\n')
    seed_everything(int(config['seed']))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    passive_context = int(config.get('passive_context', 5))
    active_stride = int(config.get('active_stride', 10))
    sample_passive_context = resolve_sample_passive_context(config, passive_context)
    train_data = TextureDataset(
        args.data_root,
        args.data_root / 'train.txt',
        passive_context,
        config['crop_size'],
        True,
        pseudo_flow_root,
        active_stride,
        sample_passive_context,
        source_off_root,
        require_source_off=True,
    )
    valid_data = TextureDataset(
        args.data_root,
        args.data_root / 'valid.txt',
        passive_context,
        pseudo_flow_root=pseudo_flow_root,
        active_stride=active_stride,
        sample_passive_context=sample_passive_context,
        source_off_root=source_off_root,
        require_source_off=True,
    )
    train_loader = infinite_loader(DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, **loader_kwargs(config, device)))
    valid_loader = DataLoader(valid_data, batch_size=int(config.get('valid_batch_size', config['batch_size'])), shuffle=False, **loader_kwargs(config, device))
    model = build_t2texture_model(args.backbone, args.pretrained, passive_context).to(device)
    criterion = CompositeLoss(config.get('loss')).to(device)
    resume_state = load_checkpoint(args.resume, device) if args.resume is not None else None
    if resume_state is not None:
        model.load_state_dict(resume_state['model'], strict=True)
    best_psnr = float('-inf')
    if resume_state is not None and 'best_psnr' in resume_state:
        best_psnr = float(resume_state['best_psnr'])
    global_step = int(resume_state.get('global_step', 0)) if resume_state is not None else 0
    resume_phase = resume_state.get('phase') if resume_state is not None else None
    resume_step = int(resume_state.get('step', 0)) if resume_state is not None else 0
    valid_interval = int(config.get('valid_interval', 500))
    phases = [('adapter', config['adapter_iterations'], config['adapter_lr']), ('finetune', config['finetune_iterations'], config['finetune_lr'])]
    phase_names = [item[0] for item in phases]
    if resume_phase is not None and resume_phase not in phase_names:
        raise ValueError(f'Unknown resume phase {resume_phase!r}; expected one of {phase_names}')
    for phase, iterations, lr in phases:
        if resume_phase is not None and phase_names.index(phase) < phase_names.index(resume_phase):
            continue
        start_step = resume_step if phase == resume_phase else 0
        model.set_phase(phase)
        optimizer = make_optimizer(model, lr, config['layerwise_lr_decay'])
        if phase == resume_phase and resume_state is not None and 'optimizer' in resume_state:
            optimizer.load_state_dict(resume_state['optimizer'])
        if start_step >= iterations:
            continue
        if start_step:
            print(json.dumps({'resume': True, 'phase': phase, 'start_step': start_step, 'global_step': global_step}), flush=True)
        for step in range(start_step, iterations):
            batch = next(train_loader)
            batch = move_batch(batch, device)
            model.train(); optimizer.zero_grad(set_to_none=True)
            result = model(batch['texture0'], batch['texture1'], batch['time'], batch['passive_context'], return_flow=True)
            losses = criterion(result['prediction'], batch['target'], result['flow0_pred'], result['flow1_pred'], batch['flow'])
            losses['total'].backward(); optimizer.step()
            global_step += 1
            if (step + 1) % 100 == 0:
                record = {
                    'phase': phase,
                    'step': step + 1,
                    'global_step': global_step,
                    **{key: float(value.detach().cpu()) for key, value in losses.items()},
                }
                print(json.dumps(record), flush=True)
                torch.save(
                    checkpoint_payload(model, run_config, args.backbone, global_step, phase, best_psnr, step=step + 1, optimizer=optimizer),
                    args.output_dir / 'last.pt',
                )
            if (step + 1) % valid_interval == 0 or step + 1 == iterations:
                valid_record = {'phase': phase, 'step': step + 1, 'global_step': global_step, **validate(model, criterion, valid_loader, device)}
                print(json.dumps(valid_record), flush=True)
                if valid_record['valid_psnr'] > best_psnr:
                    best_psnr = valid_record['valid_psnr']
                    torch.save(
                        checkpoint_payload(model, run_config, args.backbone, global_step, phase, best_psnr),
                        args.output_dir / 'best.pt',
                    )


if __name__ == '__main__':
    main()
