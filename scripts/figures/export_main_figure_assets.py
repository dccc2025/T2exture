"""Export individual T2exture assets for paper main-figure composition."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model import build_t2texture_model


def parse_args() -> argparse.Namespace:
    """Parse command-line options for one main-figure asset export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('datasets'))
    parser.add_argument('--scene', default='spot')
    parser.add_argument('--left-id', type=int, default=10)
    parser.add_argument('--target-id', type=int, default=21)
    parser.add_argument('--right-id', type=int, default=30)
    parser.add_argument('--passive-ids', type=int, nargs='+', default=[15, 17, 19, 23, 25, 27])
    parser.add_argument('--backbone', default='amt-l', choices=['amt-s', 'amt-l', 'amt-g'])
    parser.add_argument('--pretrained', type=Path, default=Path('pretrained/amt-l.pth'))
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('outputs/final/table03_passive_context/context-per-side-03/best.pt'),
    )
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/main_figure_assets/spot_t010_t030_t021'))
    parser.add_argument('--prediction-png', type=Path, default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--flow-stages', type=int, default=3)
    parser.add_argument('--flow-percentile', type=float, default=95.0)
    return parser.parse_args()


def load_frame(data_root: Path, scene: str, modality: str, frame_id: int) -> np.ndarray:
    """Load and normalise one two-dimensional frame to [0, 1]."""
    path = data_root / 'sim' / scene / modality / f'{frame_id:03d}.npy'
    if not path.is_file():
        raise FileNotFoundError(f'Missing {modality} frame: {path}')
    frame = np.load(path).astype(np.float32)
    if frame.ndim != 2:
        raise ValueError(f'Expected a 2-D frame at {path}, got {frame.shape}')
    minimum = float(frame.min())
    maximum = float(frame.max())
    if maximum <= minimum:
        return np.zeros_like(frame, dtype=np.float32)
    return (frame - minimum) / (maximum - minimum)


def save_gray(path: Path, frame: np.ndarray) -> None:
    """Save a normalised grayscale frame as an 8-bit PNG."""
    image = np.clip(frame, 0.0, 1.0)
    Image.fromarray((image * 255.0 + 0.5).astype(np.uint8), mode='L').save(path)


def tensor_from_frame(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a normalised frame to a batched model tensor."""
    return torch.from_numpy(frame.copy()).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    """Load a training checkpoint saved by train.py."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {checkpoint}')
    state: Any = torch.load(checkpoint, map_location=device, weights_only=False)
    model_state = state.get('model', state.get('state_dict', state)) if isinstance(state, dict) else state
    model.load_state_dict(model_state, strict=True)


def representative_flow(flow: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """Convert an AMT flow tensor to one H x W x 2 representative flow map."""
    flow_cpu = flow.detach().float().cpu()
    if flow_cpu.ndim == 5:
        flow_cpu = flow_cpu[0].mean(dim=0)
    elif flow_cpu.ndim == 4:
        flow_cpu = flow_cpu[0]
    else:
        raise ValueError(f'Unsupported flow tensor shape: {tuple(flow_cpu.shape)}')
    if flow_cpu.shape[0] != 2:
        raise ValueError(f'Expected two flow channels, got {tuple(flow_cpu.shape)}')
    height, width = size
    if tuple(flow_cpu.shape[-2:]) != (height, width):
        flow_cpu = F.interpolate(flow_cpu.unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False)[0]
    return flow_cpu.permute(1, 2, 0).numpy()


def flow_to_pastel(flow: np.ndarray, magnitude_max: float) -> np.ndarray:
    """Render a flow field with light blue, yellow, and green tones."""
    dx = flow[..., 0]
    dy = flow[..., 1]
    magnitude = np.sqrt(dx * dx + dy * dy)
    normalised = np.clip(magnitude / max(float(magnitude_max), 1e-6), 0.0, 1.0)
    hue = np.mod((np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi), 1.0)
    palette = np.array(
        [
            [0.62, 0.84, 1.00],
            [0.98, 0.92, 0.52],
            [0.66, 0.90, 0.70],
            [0.62, 0.84, 1.00],
        ],
        dtype=np.float32,
    )
    scaled = hue * 3.0
    index = np.floor(scaled).astype(np.int32)
    fraction = (scaled - index)[..., None]
    colour = palette[index] * (1.0 - fraction) + palette[index + 1] * fraction
    strength = (0.22 + 0.62 * normalised)[..., None]
    rgb = np.ones_like(colour) * (1.0 - strength) + colour * strength
    return np.clip(rgb, 0.0, 1.0)


def save_rgb(path: Path, image: np.ndarray) -> None:
    """Save a normalised RGB image as an 8-bit PNG."""
    Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode='RGB').save(path)


def main() -> None:
    """Run one asset export."""
    args = parse_args()
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    texture0 = load_frame(args.data_root, args.scene, 'texture', args.left_id)
    texture1 = load_frame(args.data_root, args.scene, 'texture', args.right_id)
    target = load_frame(args.data_root, args.scene, 'texture', args.target_id)
    passive_frames = [load_frame(args.data_root, args.scene, 'passive', frame_id) for frame_id in args.passive_ids]
    height, width = target.shape
    if any(frame.shape != (height, width) for frame in [texture0, texture1, *passive_frames]):
        raise ValueError('All exported frames must share the same spatial shape')

    save_gray(output_dir / f'texture_{args.left_id:03d}.png', texture0)
    save_gray(output_dir / f'texture_{args.right_id:03d}.png', texture1)
    save_gray(output_dir / f'target_texture_{args.target_id:03d}.png', target)
    for frame_id, frame in zip(args.passive_ids, passive_frames, strict=True):
        save_gray(output_dir / f'passive_{frame_id:03d}.png', frame)

    passive_context = torch.stack([torch.from_numpy(frame.copy()) for frame in passive_frames], dim=0)
    passive_context = passive_context.to(device=device, dtype=torch.float32).unsqueeze(0)
    time_value = (args.target_id - args.left_id) / float(args.right_id - args.left_id)
    time = torch.tensor([[time_value]], device=device, dtype=torch.float32)

    model = build_t2texture_model(args.backbone, args.pretrained, passive_context=len(args.passive_ids)).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    with torch.no_grad():
        result = model(
            tensor_from_frame(texture0, device),
            tensor_from_frame(texture1, device),
            time,
            passive_context,
            return_flow=True,
        )

    pred_output = output_dir / f'pred_texture_{args.target_id:03d}.png'
    if args.prediction_png is None:
        prediction = result['prediction'].detach().float().cpu()[0, 0].numpy()
        save_gray(pred_output, prediction)
        prediction_source = 'model_forward'
    else:
        if not args.prediction_png.is_file():
            raise FileNotFoundError(f'Missing prediction PNG: {args.prediction_png}')
        shutil.copy2(args.prediction_png, pred_output)
        prediction_source = str(args.prediction_png)

    flow0_pred = result.get('flow0_pred')
    flow1_pred = result.get('flow1_pred')
    if not isinstance(flow0_pred, list) or not isinstance(flow1_pred, list):
        raise ValueError('The selected model output does not include multi-stage AMT flows')
    stage_count = min(args.flow_stages, len(flow0_pred), len(flow1_pred))
    flows: list[tuple[str, np.ndarray]] = []
    for index in range(stage_count):
        flows.append((f'stage{index + 1}_flow_t_to_{args.left_id:03d}.png', representative_flow(flow0_pred[index], (height, width))))
        flows.append((f'stage{index + 1}_flow_t_to_{args.right_id:03d}.png', representative_flow(flow1_pred[index], (height, width))))

    magnitudes = np.concatenate([np.sqrt((flow[..., 0] ** 2) + (flow[..., 1] ** 2)).reshape(-1) for _, flow in flows])
    magnitude_max = float(np.percentile(magnitudes, args.flow_percentile))
    for filename, flow in flows:
        save_rgb(output_dir / filename, flow_to_pastel(flow, magnitude_max))

    manifest = {
        'scene': args.scene,
        'data_root': str(args.data_root),
        'left_id': args.left_id,
        'target_id': args.target_id,
        'right_id': args.right_id,
        'passive_ids': args.passive_ids,
        'time': time_value,
        'backbone': args.backbone,
        'pretrained': str(args.pretrained),
        'checkpoint': str(args.checkpoint),
        'prediction_source': prediction_source,
        'output_dir': str(output_dir),
        'shape': {'height': height, 'width': width},
        'flow_export': {
            'stages': stage_count,
            'directions': ['t_to_left', 't_to_right'],
            'multi_field_reduction': 'mean over AMT multi-field dimension when present',
            'style': 'pastel blue-yellow-green flow palette',
            'magnitude_percentile': args.flow_percentile,
            'magnitude_max': magnitude_max,
        },
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'output_dir': str(output_dir), 'png_files': len(list(output_dir.glob('*.png')))}, indent=2))


if __name__ == '__main__':
    main()
