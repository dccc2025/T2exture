from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from tt_data import TextureSequenceDataset, parse_left_ids
from tt_losses import charbonnier, contrast_structure, edge_loss, temporal_loss
from tt_model import GeometryGatedAMT


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=Path('dataset'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scenes', default=None,
                   help='Comma-separated scene names; bypasses train/valid split files.')
    p.add_argument('--train-left-ids', default=None)
    p.add_argument('--valid-left-ids', default=None)
    p.add_argument('--adapter-epochs', type=int, default=6)
    p.add_argument('--refine-epochs', type=int, default=4)
    p.add_argument('--crop', type=int, default=384)
    p.add_argument('--passive-context-radius', type=int, default=1,
                   help='Number of neighboring passive frames on each side: 0/1/3/5/10 for ablation.')
    p.add_argument('--no-passive', action='store_true',
                   help='Disable passive geometry branch for passive/no-passive ablation.')
    p.add_argument('--refine-modules', default='all',
                   help='AMT modules unfrozen during refine: none, decoder1, decoder2, update2, comb, decoder12, all, or comma list.')
    p.add_argument('--css-weight', type=float, default=0.1)
    p.add_argument('--edge-weight', type=float, default=0.1)
    p.add_argument('--temp-weight', type=float, default=0.0)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def flatten_batch(batch: dict, device: torch.device):
    b, n = batch['p'].shape[:2]
    x0 = batch['x0'].to(device).repeat_interleave(n, 0)
    x1 = batch['x1'].to(device).repeat_interleave(n, 0)
    p0 = batch['p0'].to(device).repeat_interleave(n, 0)
    p1 = batch['p1'].to(device).repeat_interleave(n, 0)
    passive = batch['p'].to(device).flatten(0, 1)
    passive_context = batch.get('pbar', batch['p']).to(device).flatten(0, 1)
    target = batch['target'].to(device).flatten(0, 1)
    times = batch['times'].to(device).reshape(b * n, 1, 1, 1)
    return x0, x1, p0, p1, passive, passive_context, target, times, b, n


def losses_for(model, batch, device, train: bool, args):
    x0, x1, p0, p1, passive, passive_context, target, times, b, n = flatten_batch(batch, device)
    with autocast(enabled=device.type == 'cuda'):
        prediction = model(x0, x1, times, p0, p1, passive_context).mean(1, keepdim=True)
        rec = charbonnier(prediction, target)
        css = contrast_structure(prediction, target)
        edge = edge_loss(prediction, target, passive)
        pred_seq, target_seq = prediction.view(b, n, *prediction.shape[1:]), target.view(b, n, *target.shape[1:])
        temp = temporal_loss(pred_seq, target_seq)
        total = rec + args.css_weight * css + args.edge_weight * edge + args.temp_weight * temp
    return total, {'total': total, 'charbonnier': rec, 'css': css, 'edge': edge, 'temp': temp}


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval(); sums = {key: 0. for key in ('total', 'charbonnier', 'css', 'edge', 'temp')}; count = 0
    for batch in loader:
        _, values = losses_for(model, batch, device, False, args)
        for key, value in values.items(): sums[key] += float(value.detach().cpu())
        count += 1
    return {key: value / max(count, 1) for key, value in sums.items()}


def plot_curve(rows, path: Path):
    plt.figure(figsize=(8, 5))
    for key in ('train_total', 'valid_total', 'train_charbonnier', 'valid_charbonnier'):
        plt.plot([int(row['epoch']) for row in rows], [float(row[key]) for row in rows], label=key)
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()


def main():
    args = arguments(); args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed); device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    scenes = [scene.strip() for scene in args.scenes.split(',') if scene.strip()] if args.scenes else None
    if scenes is None:
        split_dir = args.data_root / 'train_valid_test_splits'
        train_set = TextureSequenceDataset(
            args.data_root, split_dir / 'train.txt', crop_size=args.crop,
            context_radius=args.passive_context_radius,
            left_ids=parse_left_ids(args.train_left_ids))
        valid_set = TextureSequenceDataset(
            args.data_root, split_dir / 'valid.txt', crop_size=None,
            context_radius=args.passive_context_radius,
            left_ids=parse_left_ids(args.valid_left_ids), random_crop=False)
    else:
        train_set = TextureSequenceDataset(
            args.data_root, scenes=scenes, left_ids=parse_left_ids(args.train_left_ids),
            crop_size=args.crop, context_radius=args.passive_context_radius)
        valid_set = TextureSequenceDataset(
            args.data_root, scenes=scenes, left_ids=parse_left_ids(args.valid_left_ids),
            crop_size=None, context_radius=args.passive_context_radius, random_crop=False)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    model = GeometryGatedAMT(
        context_radius=args.passive_context_radius,
        use_passive=not args.no_passive,
        refine_modules=args.refine_modules).to(device)
    scaler = GradScaler(enabled=device.type == 'cuda')
    rows, best = [], float('inf')
    phases = [('adapter', args.adapter_epochs, 2e-4), ('refine', args.refine_epochs, 5e-5)]
    epoch = 0
    config = vars(args) | {'device': str(device), 'train_scenes': train_set.scenes, 'valid_scenes': valid_set.scenes}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2, default=str) + '\n')
    for phase, phase_epochs, learning_rate in phases:
        model.set_phase(phase)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=1e-5)
        for _ in range(phase_epochs):
            epoch += 1; model.train(); sums = {key: 0. for key in ('total', 'charbonnier', 'css', 'edge', 'temp')}; count = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                total, values = losses_for(model, batch, device, True, args)
                scaler.scale(total).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer); scaler.update()
                for key, value in values.items(): sums[key] += float(value.detach().cpu())
                count += 1
            train_values = {key: value / count for key, value in sums.items()}
            valid_values = validate(model, valid_loader, device, args)
            row = {'epoch': epoch, 'phase': phase} | {f'train_{k}': v for k, v in train_values.items()} | {f'valid_{k}': v for k, v in valid_values.items()}
            rows.append(row)
            with (args.output / 'history.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
            plot_curve(rows, args.output / 'loss_curve.png')
            torch.save({'epoch': epoch, 'phase': phase, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'valid': valid_values, 'config': config}, args.output / 'last.pt')
            if valid_values['total'] < best:
                best = valid_values['total']; torch.save({'epoch': epoch, 'phase': phase, 'model': model.state_dict(), 'valid': valid_values, 'config': config}, args.output / 'best.pt')
            print(json.dumps(row), flush=True)
    print(json.dumps({'best_valid_total': best, 'epochs': epoch, 'checkpoint': str(args.output / 'best.pt')}))


if __name__ == '__main__': main()
