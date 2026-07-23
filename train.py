"""Fine-tune T2exture AMT-L with the fixed three-term objective."""

from __future__ import annotations

import argparse
import json
import random
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import TextureDataset
from losses.loss import CompositeLoss
from model import T2textureAMTL


def parse_args() -> argparse.Namespace:
    """Read only user-facing paths and an optional training configuration path."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('train.yaml'))
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Make dataset sampling and model initialization repeatable when possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(model: T2textureAMTL, base_lr: float, decay: float) -> torch.optim.Optimizer:
    """Create parameter groups with lower rates for earlier AMT-L backbone layers."""
    groups = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        depth = 0 if not name.startswith('backbone.') else name.count('.')
        groups.append({'params': [parameter], 'lr': base_lr * (decay ** depth)})
    return torch.optim.AdamW(groups, weight_decay=1e-5)


def main() -> None:
    """Run adapter-only training followed by whole-model layerwise fine-tuning."""
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(config['seed']))
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train_data = TextureDataset(args.data_root, args.data_root / 'train.txt', config['passive_context'], config['crop_size'], True)
    valid_data = TextureDataset(args.data_root, args.data_root / 'valid.txt', config['passive_context'])
    train_loader = cycle(DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers']))
    valid_loader = DataLoader(valid_data, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    model = T2textureAMTL(args.pretrained, config['passive_context']).to(device)
    criterion = CompositeLoss().to(device)
    best = float('inf')
    for phase, iterations, lr in [('adapter', config['adapter_iterations'], config['adapter_lr']), ('finetune', config['finetune_iterations'], config['finetune_lr'])]:
        model.set_phase(phase)
        optimizer = make_optimizer(model, lr, config['layerwise_lr_decay'])
        for step in range(iterations):
            batch = next(train_loader)
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            model.train(); optimizer.zero_grad(set_to_none=True)
            result = model(batch['texture0'], batch['texture1'], batch['time'], batch['passive_context'], return_flow=True)
            losses = criterion(result['prediction'], batch['target'], result['flow0_pred'], result['flow1_pred'], batch['flow'])
            losses['total'].backward(); optimizer.step()
            if (step + 1) % 100 == 0:
                record = {'phase': phase, 'step': step + 1, **{key: float(value.detach().cpu()) for key, value in losses.items()}}
                print(json.dumps(record), flush=True)
                torch.save({'model': model.state_dict(), 'config': config, 'step': step + 1, 'phase': phase}, args.output_dir / 'last.pt')
                if record['total'] < best:
                    best = record['total']; torch.save({'model': model.state_dict(), 'config': config}, args.output_dir / 'best.pt')
    (args.output_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')


if __name__ == '__main__':
    main()
