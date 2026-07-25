"""Evaluate trained and official VFI models on real T2exture sequences."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import _relative, _sample_name, _save_png, _state_dict, _torch_load
from metrics import REAL_METRIC_KEYS, compute_real_metrics, mean_real_metrics
from model import build_t2texture_model
from scripts.eval_amt_vanilla import load_model as load_amt_model
from scripts.eval_bim_vfi import load_model as load_bim_model
from scripts.eval_gimm_vfi_f import load_model as load_gimm_model
from scripts.eval_ifrnet import load_model as load_ifrnet_model
from scripts.eval_sgm_vfi import load_model as load_sgm_model


METHODS = ('ifrnet', 'sgm-vfi', 'bim-vfi', 'gimm-vfi-f', 'amt-l-vanilla', 'ours-l')


def parse_args() -> argparse.Namespace:
    """Read real-benchmark model, data, and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/formal/table06/real-benchmark.yaml'))
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-frames-per-video', type=int, default=0)
    parser.add_argument('--fps', type=int, default=12)

    parser.add_argument('--ifrnet-checkpoint', type=Path, default=Path('pretrained/IFRNet/IFRNet_Vimeo90K.pth'))
    parser.add_argument('--sgm-pretrained-root', type=Path, default=Path('pretrained/sgm-vfi'))
    parser.add_argument('--sgm-exp-name', default='ours-1-2-points')
    parser.add_argument('--bim-checkpoint', type=Path, default=Path('pretrained/bim-vfi/bim_vfi.pth'))
    parser.add_argument('--gimm-checkpoint', type=Path, default=Path('pretrained/GIMM-VFI-F/gimmvfi_f_arb.pt'))
    parser.add_argument('--gimm-flowformer', type=Path, default=Path('pretrained/GIMM-VFI-F/flowformer_sintel.pth'))
    parser.add_argument('--gimm-config', type=Path, default=Path('third_party/GIMM-VFI/configs/gimmvfi/gimmvfi_f_arb.yaml'))
    parser.add_argument('--amt-l-pretrained', type=Path, default=Path('pretrained/amt-l.pth'))
    parser.add_argument('--ours-checkpoint', type=Path, default=Path('outputs/final/table01_prior/ours-l/best.pt'))
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    """Load one YAML mapping."""
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Expected a YAML mapping in {path}')
    return data


def frame_root(sequence_root: Path) -> Path:
    """Return the directory containing ordered real frames."""
    nested = sequence_root / 'frames'
    return nested if nested.is_dir() else sequence_root


def frame_path(data_root: Path, sequence: str, frame_id: int) -> Path:
    """Return one real frame path, supporting both flat and frames/ layouts."""
    path = frame_root(data_root / sequence) / f'{frame_id:03d}.png'
    if not path.is_file():
        raise FileNotFoundError(f'Missing real frame: {path}')
    return path


def load_frame(data_root: Path, sequence: str, frame_id: int, device: torch.device) -> torch.Tensor:
    """Load one RGB PNG as a normalized grayscale tensor [1, 1, H, W]."""
    image = Image.open(frame_path(data_root, sequence, frame_id)).convert('L')
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)


def tensor_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """Convert a grayscale batch tensor to three channels for RGB-pretrained VFI models."""
    if image.shape[1] != 1:
        raise ValueError(f'Expected one-channel tensor, got {image.shape}')
    return image.repeat(1, 3, 1, 1)


def real_samples(active_ids: list[int]) -> list[dict[str, int | float]]:
    """Build all target-frame samples between successive active anchors."""
    samples: list[dict[str, int | float]] = []
    for left_id, right_id in zip(active_ids[:-1], active_ids[1:]):
        span = right_id - left_id
        if span <= 1:
            continue
        for target_id in range(left_id + 1, right_id):
            samples.append(
                {
                    'left_id': left_id,
                    'target_id': target_id,
                    'right_id': right_id,
                    'time': (target_id - left_id) / float(span),
                }
            )
    if not samples:
        raise ValueError('Real benchmark protocol produced no interpolation samples')
    return samples


def passive_context_ids(target_id: int, active_ids: set[int], context_size: int, min_id: int = 1, max_id: int = 180) -> list[int]:
    """Select nearest non-anchor passive context frames around one target."""
    if context_size < 0:
        raise ValueError('passive_context must be non-negative')
    if context_size == 0:
        return []
    candidates = [
        frame_id
        for frame_id in range(min_id, max_id + 1)
        if frame_id != target_id and frame_id not in active_ids
    ]
    candidates.sort(key=lambda frame_id: (abs(frame_id - target_id), frame_id))
    selected = sorted(candidates[:context_size])
    if len(selected) != context_size:
        raise ValueError(f'Only found {len(selected)} passive context frames for target {target_id}')
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV rows with a stable header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_video(path: Path, frame_paths: list[Path], fps: int) -> None:
    """Write a list of PNG frames to one MP4 video."""
    if not frame_paths:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
        for frame in frame_paths:
            image = Image.open(frame).convert('RGB')
            width, height = image.size
            if width % 2 or height % 2:
                canvas = Image.new('RGB', (width + width % 2, height + height % 2), (0, 0, 0))
                canvas.paste(image, (0, 0))
                image = canvas
            writer.append_data(np.asarray(image))


def font(size: int) -> ImageFont.ImageFont:
    """Load a readable font with fallback."""
    for name in ('arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def labeled_panel(image: Image.Image, label: str, width: int = 320) -> Image.Image:
    """Resize and label one visualization panel."""
    scale_height = round(image.height * width / image.width)
    resized = image.resize((width, scale_height), Image.BICUBIC).convert('RGB')
    label_height = 36
    panel = Image.new('RGB', (width, scale_height + label_height), (15, 15, 15))
    panel.paste(resized, (0, label_height))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 9), label, fill=(245, 245, 245), font=font(18))
    return panel


def make_vis(data_root: Path, sequence: str, row: dict[str, Any], pred_path: Path, label: str) -> Image.Image:
    """Create one real-comparison sheet without GT or error maps."""
    panels = [
        labeled_panel(Image.open(frame_path(data_root, sequence, int(row['left_id']))).convert('L'), 'Left Active'),
        labeled_panel(Image.open(pred_path).convert('L'), label),
        labeled_panel(Image.open(frame_path(data_root, sequence, int(row['right_id']))).convert('L'), 'Right Active'),
    ]
    canvas = Image.new('RGB', (sum(panel.width for panel in panels), max(panel.height for panel in panels)), (0, 0, 0))
    left = 0
    for panel in panels:
        canvas.paste(panel, (left, 0))
        left += panel.width
    return canvas


def load_predictor(args: argparse.Namespace, device: torch.device, passive_context: int) -> tuple[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]:
    """Load one method and return a normalized grayscale prediction callback."""
    if args.method == 'ifrnet':
        model = load_ifrnet_model(args.ifrnet_checkpoint, device)
        return 'IFRNet [CVPR 2022]', lambda x0, x1, t, _ctx: model.inference(tensor_to_rgb(x0), tensor_to_rgb(x1), t.view(1, 1, 1, 1)).mean(dim=1, keepdim=True).clamp(0.0, 1.0)

    if args.method == 'sgm-vfi':
        namespace = SimpleNamespace(
            device=args.device,
            pretrained_root=args.sgm_pretrained_root,
            exp_name=args.sgm_exp_name,
            num_key_points=0.5,
        )
        model = load_sgm_model(namespace)
        return 'SGM-VFI [CVPR 2024]', lambda x0, x1, t, _ctx: model.hr_inference(tensor_to_rgb(x0), tensor_to_rgb(x1), TTA=False, down_scale=1.0, timestep=t.view(1, 1, 1, 1)).mean(dim=1, keepdim=True).clamp(0.0, 1.0)

    if args.method == 'bim-vfi':
        namespace = SimpleNamespace(checkpoint=args.bim_checkpoint, pyr_level=3, feat_channels=32)
        model = load_bim_model(namespace, device)
        return 'BiM-VFI [CVPR 2025]', lambda x0, x1, t, _ctx: model(img0=tensor_to_rgb(x0), img1=tensor_to_rgb(x1), time_step=t.view(1, 1, 1, 1), pyr_level=3, run_with_gt=False)['imgt_pred'].mean(dim=1, keepdim=True).clamp(0.0, 1.0)

    if args.method == 'gimm-vfi-f':
        namespace = SimpleNamespace(
            checkpoint=args.gimm_checkpoint,
            flowformer=args.gimm_flowformer,
            model_config=args.gimm_config,
            ds_factor=1.0,
        )
        model, _gimm_config, input_padder = load_gimm_model(namespace, device)

        def predict_gimm(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, _ctx: torch.Tensor) -> torch.Tensor:
            image0 = tensor_to_rgb(x0)
            image1 = tensor_to_rgb(x1)
            padder = input_padder(image0.shape, 32)
            image0, image1 = padder.pad(image0, image1)
            xs = torch.cat((image0.unsqueeze(2), image1.unsqueeze(2)), dim=2)
            timestep = float(t[0].detach().cpu())
            coord_inputs = [(model.sample_coord_input(xs.shape[0], xs.shape[-2:], [timestep], device=xs.device, upsample_ratio=1.0), None)]
            timesteps = [torch.tensor([timestep], device=xs.device, dtype=torch.float32)]
            output = model(xs, coord_inputs, t=timesteps, ds_factor=1.0)
            return padder.unpad(output['imgt_pred'][0]).mean(dim=1, keepdim=True).clamp(0.0, 1.0)

        return 'GIMM-VFI-F [NeurIPS 2024]', predict_gimm

    if args.method == 'amt-l-vanilla':
        model = load_amt_model('amt-l', args.amt_l_pretrained, device)
        return 'AMT-L [CVPR 2023]', lambda x0, x1, t, _ctx: model(tensor_to_rgb(x0), tensor_to_rgb(x1), t.view(1, 1, 1, 1), eval=True)['imgt_pred'].mean(dim=1, keepdim=True).clamp(0.0, 1.0)

    if args.method == 'ours-l':
        model = build_t2texture_model('amt-l', args.amt_l_pretrained, passive_context).to(device)
        checkpoint = _torch_load(args.ours_checkpoint, device)
        model.load_state_dict(_state_dict(checkpoint), strict=True)
        model.eval()
        return 'Ours-L', lambda x0, x1, t, ctx: model(x0, x1, t, ctx, return_flow=False)['prediction'].clamp(0.0, 1.0)

    raise ValueError(f'Unsupported method: {args.method}')


def sequence_mean(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Average real metrics for one sequence."""
    return mean_real_metrics([{key: row.get(key) for key in REAL_METRIC_KEYS} for row in rows])


def main() -> None:
    """Run real-sequence inference and write per-frame, per-sequence, and method artifacts."""
    args = parse_args()
    config = read_config(args.config)
    protocol = config.get('protocol', {})
    data_root = Path(config.get('real_data_root', 'dataset/real'))
    output_dir = args.output_dir or Path(config.get('output_root', 'outputs/final/table06_real_benchmark')) / args.method
    output_dir.mkdir(parents=True, exist_ok=True)

    active_ids = [int(item) for item in protocol.get('active_anchor_ids', [])]
    if not active_ids:
        raise ValueError('Table 6 config must provide protocol.active_anchor_ids')
    active_set = set(active_ids)
    sequences = [str(item) for item in config.get('sequences', [])]
    if not sequences:
        raise ValueError('Table 6 config must provide sequences')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    passive_context = int(protocol.get('passive_context', 4))
    label, predict = load_predictor(args, device, passive_context)
    samples = real_samples(active_ids)

    sequence_rows: list[dict[str, Any]] = []
    method_frame_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for sequence in sequences:
            sequence_dir = output_dir / sequence
            pred_dir = sequence_dir / 'pred'
            vis_png_dir = sequence_dir / 'vis' / 'png'
            frame_rows: list[dict[str, Any]] = []
            vis_frames: list[Path] = []
            cache: dict[int, torch.Tensor] = {}

            def get_frame(frame_id: int) -> torch.Tensor:
                if frame_id not in cache:
                    cache[frame_id] = load_frame(data_root, sequence, frame_id, device)
                return cache[frame_id]

            for sample in samples:
                left_id = int(sample['left_id'])
                target_id = int(sample['target_id'])
                right_id = int(sample['right_id'])
                time = torch.tensor([float(sample['time'])], device=device, dtype=torch.float32)
                context_ids = passive_context_ids(target_id, active_set, passive_context)
                context = torch.cat([get_frame(frame_id) for frame_id in context_ids], dim=1) if context_ids else torch.empty((1, 0, *get_frame(left_id).shape[-2:]), device=device)
                left = get_frame(left_id)
                right = get_frame(right_id)
                prediction = predict(left, right, time, context)

                name = _sample_name(left_id, target_id, right_id)
                pred_path = pred_dir / f'{name}.png'
                _save_png(prediction[0], pred_path)
                metrics = compute_real_metrics(prediction[0], left[0], right[0])
                row = {
                    'sequence': sequence,
                    'left_id': left_id,
                    'target_id': target_id,
                    'right_id': right_id,
                    'time': float(sample['time']),
                    'passive_context_ids': ' '.join(f'{frame_id:03d}' for frame_id in context_ids),
                    **metrics,
                    'pred_path': _relative(pred_path, sequence_dir),
                }
                frame_rows.append(row)
                method_frame_rows.append({'sequence': sequence, **row})

                if args.max_frames_per_video <= 0 or len(vis_frames) < args.max_frames_per_video:
                    vis_png_dir.mkdir(parents=True, exist_ok=True)
                    vis_path = vis_png_dir / f'{name}_real_cmp.png'
                    make_vis(data_root, sequence, row, pred_path, label).save(vis_path)
                    vis_frames.append(vis_path)

            if not frame_rows:
                raise ValueError(f'No real frames evaluated for sequence {sequence}')
            seq_overall = sequence_mean(frame_rows)
            sequence_rows.append({'sequence': sequence, 'frames': len(frame_rows), **seq_overall})
            write_csv(sequence_dir / 'frame.csv', frame_rows, ['sequence', 'left_id', 'target_id', 'right_id', 'time', 'passive_context_ids', *REAL_METRIC_KEYS, 'pred_path'])
            (sequence_dir / 'metrics.json').write_text(json.dumps({'sequence': sequence, 'frames': len(frame_rows), 'overall': seq_overall}, indent=2) + '\n', encoding='utf-8')
            write_video(sequence_dir / 'vis' / 'video' / f'{sequence}_real_cmp.mp4', vis_frames, args.fps)
            (sequence_dir / 'manifest.json').write_text(
                json.dumps(
                    {
                        'method': args.method,
                        'label': label,
                        'sequence': sequence,
                        'data_root': str(data_root.resolve()),
                        'active_anchor_ids': active_ids,
                        'passive_context': passive_context,
                        'passive_context_rule': 'nearest non-anchor passive frames excluding the target frame',
                        'time_normalization': '(target_id - left_id) / (right_id - left_id)',
                        'metrics': list(REAL_METRIC_KEYS),
                        'pi_note': 'PI is left null unless an external perceptual-IQA implementation is supplied.',
                    },
                    indent=2,
                )
                + '\n',
                encoding='utf-8',
            )

    overall = mean_real_metrics([{key: row.get(key) for key in REAL_METRIC_KEYS} for row in sequence_rows])
    write_csv(output_dir / 'sequence.csv', sequence_rows, ['sequence', 'frames', *REAL_METRIC_KEYS])
    write_csv(output_dir / 'frame.csv', method_frame_rows, ['sequence', 'left_id', 'target_id', 'right_id', 'time', 'passive_context_ids', *REAL_METRIC_KEYS, 'pred_path'])
    metrics = {'method': args.method, 'label': label, 'sequences': len(sequence_rows), 'overall': overall, 'by_sequence': sequence_rows}
    manifest = {
        **metrics,
        'config': str(args.config.resolve()),
        'data_root': str(data_root.resolve()),
        'active_anchor_ids': active_ids,
        'real_finetune': False,
        'aggregation': 'frame metrics averaged per sequence, then unweighted mean over sequences',
        'pi_note': 'PI is not computed by this script without a confirmed external perceptual-IQA implementation.',
    }
    (output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(overall, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
