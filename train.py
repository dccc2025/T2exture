"""Fine-tune T2exture AMT-L with the fixed three-term objective."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import TextureDataset
from losses.loss import CompositeLoss
from metrics import batch_psnr
from model import build_t2texture_model


def parse_args() -> argparse.Namespace:
    """Read only user-facing paths and an optional training configuration path."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--backbone', choices=['amt-s', 'amt-l', 'amt-g'], default='amt-l')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('train.yaml'))
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
    """Create parameter groups with lower rates for earlier AMT-L backbone layers."""
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


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values to the training device without touching metadata fields."""
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def resolve_data_path(root: Path, value: str | Path) -> Path:
    """Resolve a config path relative to the dataset root unless it is absolute."""
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint and require the training checkpoint dictionary format."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
        raise ValueError(f'Expected a training checkpoint with a model state dict: {path}')
    return checkpoint


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
    config = yaml.safe_load(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    seed_everything(int(config['seed']))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    pseudo_flow_root = Path(config['pseudo_flow_dir']) if config.get('pseudo_flow_dir') else None
    train_data = TextureDataset(args.data_root, args.data_root / 'train.txt', config['passive_context'], config['crop_size'], True, pseudo_flow_root)
    valid_data = TextureDataset(args.data_root, args.data_root / 'valid.txt', config['passive_context'], pseudo_flow_root=pseudo_flow_root)
    train_loader = infinite_loader(DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']))
    valid_loader = DataLoader(valid_data, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    model = build_t2texture_model(args.backbone, args.pretrained, config['passive_context']).to(device)
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
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config, 'step': step + 1, 'global_step': global_step, 'phase': phase, 'best_psnr': best_psnr}, args.output_dir / 'last.pt')
            if (step + 1) % valid_interval == 0 or step + 1 == iterations:
                valid_record = {'phase': phase, 'step': step + 1, 'global_step': global_step, **validate(model, criterion, valid_loader, device)}
                print(json.dumps(valid_record), flush=True)
                if valid_record['valid_psnr'] > best_psnr:
                    best_psnr = valid_record['valid_psnr']
                    torch.save({'model': model.state_dict(), 'config': config, 'best_psnr': best_psnr, 'global_step': global_step, 'phase': phase}, args.output_dir / 'best.pt')


if __name__ == '__main__':
    main()
