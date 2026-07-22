from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from tt_data import TextureSequenceDataset, parse_left_ids
from tt_model import GeometryGatedAMT
from scripts.runtime_utils import (
    InferenceTimer,
    count_parameters_m,
    cuda_warmup,
    efficiency_from_runtime,
    make_table_row,
    profile_flops_t,
    write_runtime,
)


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, default=Path('dataset'))
    p.add_argument('--checkpoint', type=Path, default=None)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--method-label', default=None)
    p.add_argument('--scenes', default=None)
    p.add_argument('--left-ids', default=None)
    p.add_argument('--passive-context-radius', type=int, default=1)
    p.add_argument('--no-passive', action='store_true')
    p.add_argument('--refine-modules', default='all')
    p.add_argument('--lpips-path', type=Path, default=Path('/essfs10/daicheng/AMT/third_party'))
    p.add_argument('--device', default='cuda')
    p.add_argument('--warmup-iters', type=int, default=5)
    return p.parse_args()


def load_checkpoint(path: Path) -> dict:
    original_posix = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        return torch.load(path, map_location='cpu', weights_only=False)
    finally:
        pathlib.PosixPath = original_posix


def infer_context_radius(state: dict, fallback: int) -> int:
    model_state = state.get('model', {}) if isinstance(state, dict) else {}
    weight = model_state.get('geometry.encoder.0.weight')
    if weight is None:
        return fallback
    in_channels = int(weight.shape[1])
    inferred = (in_channels - 7) // 2
    if inferred < 0 or 2 * inferred + 7 != in_channels:
        return fallback
    return inferred


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    state = None
    context_radius = args.passive_context_radius
    use_passive = not args.no_passive
    if args.checkpoint is not None:
        state = load_checkpoint(args.checkpoint)
        config = state.get('config', {}) if isinstance(state, dict) else {}
        context_radius = int(config.get('passive_context_radius', infer_context_radius(state, context_radius)))
        use_passive = not bool(config.get('no_passive', not use_passive))
        args.refine_modules = config.get('refine_modules', args.refine_modules)
    model = GeometryGatedAMT(
        context_radius=context_radius,
        use_passive=use_passive,
        refine_modules=args.refine_modules).to(device)
    if state is not None:
        model.load_state_dict(state['model'], strict=True)
    model.eval()
    params_m = count_parameters_m(model)
    method_label = args.method_label or ('Ours' if args.checkpoint is not None else 'GeometryGatedAMT-zero')

    try:
        sys.path.insert(0, str(args.lpips_path))
        import lpips
    except ImportError as exc:
        raise RuntimeError(
            'LPIPS is required for evaluation. Use a complete environment such as '
            '`conda run -n gaussian_flow ...` or `conda run -n gflow ...`; '
            'do not run degraded metrics.'
        ) from exc
    perceptual = lpips.LPIPS(net='alex').to(device).eval()
    scenes = [scene.strip() for scene in args.scenes.split(',') if scene.strip()] if args.scenes else None
    if scenes is None:
        split = args.data_root / 'train_valid_test_splits' / 'test.txt'
        dataset = TextureSequenceDataset(args.data_root, split, crop_size=None,
                                         context_radius=context_radius,
                                         left_ids=parse_left_ids(args.left_ids), random_crop=False)
    else:
        split = None
        dataset = TextureSequenceDataset(
            args.data_root, scenes=scenes, left_ids=parse_left_ids(args.left_ids),
            crop_size=None, context_radius=context_radius, random_crop=False)

    rows = []
    timer = InferenceTimer(device)
    warmed_up = False
    flops_t: float | None = None
    flops_error: str | None = None
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
            x0, x1 = batch['x0'].to(device), batch['x1'].to(device)
            p0, p1 = batch['p0'].to(device), batch['p1'].to(device)
            if not warmed_up:
                warmup_time = batch['times'][:, 0].to(device).view(1, 1, 1, 1)
                warmup_passive_context = batch.get('pbar', batch['p'])[:, 0].to(device)
                flops_t, flops_error = profile_flops_t(
                    lambda: model(x0, x1, warmup_time, p0, p1, warmup_passive_context), device)
                cuda_warmup(device, args.warmup_iters,
                            lambda: model(x0, x1, warmup_time, p0, p1, warmup_passive_context))
                warmed_up = True
            for j in range(9):
                passive = batch['p'][:, j].to(device)
                target = batch['target'][:, j].to(device)
                time = batch['times'][:, j].to(device).view(1, 1, 1, 1)
                passive_context = batch.get('pbar', batch['p'])[:, j].to(device)
                pred = timer.measure(lambda: model(x0, x1, time, p0, p1, passive_context))
                pred = pred.mean(1, keepdim=True).clamp(0, 1)
                gt, pr = target[0, 0].cpu().numpy(), pred[0, 0].cpu().numpy()
                mse = float(np.mean((gt-pr)**2))
                psnr = float('inf') if mse == 0 else float(-10*np.log10(mse))
                ssim = float(structural_similarity(gt, pr, data_range=1.0))
                mae = float(np.mean(np.abs(gt-pr)))
                lp = float(perceptual(target.repeat(1,3,1,1)*2-1, pred.repeat(1,3,1,1)*2-1).item())
                rows.append({
                    'scene': batch['scene'][0],
                    'left_id': int(batch['left_id'][0]),
                    'target_id': int(batch['left_id'][0])+j+1,
                    'psnr': psnr,
                    'ssim': ssim,
                    'mae': mae,
                    'lpips': lp,
                })

    if not rows:
        raise RuntimeError('No evaluation rows were produced.')

    with (args.output/'test_per_frame.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    metric_names = ('psnr', 'ssim', 'mae', 'lpips')
    mean_metrics = {k: float(np.mean([r[k] for r in rows])) for k in metric_names}
    runtime = timer.summary(
        method_label,
        len(rows),
        args.warmup_iters,
        params_m=params_m,
        flops_t=flops_t,
        flops_error=flops_error,
        extra={
            'checkpoint': str(args.checkpoint) if args.checkpoint else 'AMT-L zero-adapter baseline',
            'passive_context_radius': context_radius,
            'use_passive': use_passive,
            'refine_modules': args.refine_modules,
        },
    )
    write_runtime(args.output/'runtime.json', runtime)
    summary = {
        'model': method_label,
        'checkpoint': str(args.checkpoint) if args.checkpoint else 'AMT-L zero-adapter baseline',
        'frames': len(rows),
        'split': str(split) if split is not None else None,
        'scenes': dataset.scenes,
        'mean': mean_metrics,
        'std': {k: float(np.std([r[k] for r in rows])) for k in metric_names},
        'efficiency': efficiency_from_runtime(runtime),
        'table_row': make_table_row(method_label, len(rows), mean_metrics, runtime),
        'runtime': runtime,
    }
    (args.output/'test_metrics.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
