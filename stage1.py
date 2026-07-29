"""Generate Stage 1 source-off passive-state caches with pretrained AMT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import read_split
from model.t2texture_base import AMT_G_SPEC, AMT_L_SPEC, AMT_S_SPEC, load_amt_class


SPECS = {
    'amt-s': AMT_S_SPEC,
    'amt-l': AMT_L_SPEC,
    'amt-g': AMT_G_SPEC,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['synthetic', 'real'], default='synthetic')
    parser.add_argument('--data-root', type=Path, default=Path('datasets'))
    parser.add_argument('--real-config', type=Path, default=Path('configs/real.yaml'))
    parser.add_argument('--splits', nargs='+', choices=['train', 'valid', 'test'], default=['train', 'valid', 'test'])
    parser.add_argument('--sequences', nargs='*', default=None, help='Real sequences to process. Defaults to all real-config sequences.')
    parser.add_argument('--backbone', choices=sorted(SPECS), required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--active-stride', type=int, default=10)
    parser.add_argument('--frame-count', type=int, default=180)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def passive_path(root: Path, scene: str, frame_id: int) -> Path:
    return root / 'sim' / scene / 'passive' / f'{frame_id:03d}.npy'


def load_passive(root: Path, scene: str, frame_id: int) -> np.ndarray:
    path = passive_path(root, scene, frame_id)
    if not path.is_file():
        raise FileNotFoundError(f'Missing passive frame: {path}')
    image = np.load(path).astype(np.float32)
    if image.ndim != 2:
        raise ValueError(f'Expected a 2-D passive frame at {path}, got {image.shape}')
    return image


def real_frame_path(root: Path, sequence: str, frame_id: int, modality: str | None = None) -> Path:
    sequence_root = root / sequence
    frame_root = sequence_root / modality if modality is not None and (sequence_root / modality).is_dir() else sequence_root
    candidates = [frame_root / f'{frame_id:03d}{suffix}' for suffix in ('.npy', '.png', '.jpg', '.jpeg')]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        checked = ', '.join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f'Missing real frame {sequence}/{frame_id:03d}. Checked: {checked}')
    return path


def load_frame(path: Path) -> np.ndarray:
    if path.suffix.lower() == '.npy':
        image = np.load(path).astype(np.float32)
        if image.ndim != 2:
            raise ValueError(f'Expected a 2-D frame at {path}, got {image.shape}')
        return image
    return np.asarray(Image.open(path).convert('L'), dtype=np.float32) / 255.0


def load_real_passive(root: Path, sequence: str, frame_id: int) -> np.ndarray:
    if (root / sequence / 'passive').is_dir():
        return load_frame(real_frame_path(root, sequence, frame_id, 'passive'))
    if (root / sequence / 'source_off').is_dir():
        return load_frame(real_frame_path(root, sequence, frame_id, 'source_off'))
    return load_frame(real_frame_path(root, sequence, frame_id))


def real_has_explicit_source_off(root: Path, sequence: str) -> bool:
    return (root / sequence / 'passive').is_dir() or (root / sequence / 'source_off').is_dir()


def to_model_input(left: np.ndarray, right: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    minimum = float(min(left.min(), right.min()))
    maximum = float(max(left.max(), right.max()))
    scale = maximum - minimum
    if scale <= 0.0:
        left_norm = np.zeros_like(left, dtype=np.float32)
        right_norm = np.zeros_like(right, dtype=np.float32)
    else:
        left_norm = (left - minimum) / scale
        right_norm = (right - minimum) / scale
    left_tensor = torch.from_numpy(np.repeat(left_norm[None, None], 3, axis=1)).to(device)
    right_tensor = torch.from_numpy(np.repeat(right_norm[None, None], 3, axis=1)).to(device)
    return left_tensor, right_tensor, minimum, scale


def active_ids(active_stride: int, frame_count: int) -> list[int]:
    if active_stride <= 1:
        raise ValueError('active_stride must be larger than 1')
    return list(range(1, frame_count - active_stride, active_stride)) + [frame_count - active_stride + 1]


def load_real_protocol(path: Path) -> tuple[list[str], list[int]]:
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError(f'Expected a YAML mapping in {path}')
    protocol = config.get('protocol', {})
    sequences = [str(item) for item in config.get('sequences', [])]
    active = [int(item) for item in protocol.get('active_anchor_ids', [])]
    if not sequences or not active:
        raise ValueError(f'{path} must define sequences and protocol.active_anchor_ids')
    return sequences, active


def public_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace('\\', '/')
    except ValueError:
        return path.name


def build_model(backbone: str, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    spec = SPECS[backbone]
    model_class = load_amt_class(spec.filename, f'stage1_{backbone.replace("-", "_")}')
    model = model_class().to(device)
    state: dict[str, Any] = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state.get('state_dict', state), strict=True)
    model.eval()
    return model


@torch.no_grad()
def estimate_source_off(model: torch.nn.Module, left: np.ndarray, right: np.ndarray, device: torch.device) -> np.ndarray:
    left_tensor, right_tensor, minimum, scale = to_model_input(left, right, device)
    time = torch.full((1, 1, 1, 1), 0.5, dtype=torch.float32, device=device)
    output = model(left_tensor, right_tensor, time, eval=True)['imgt_pred']
    image = output.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return image * scale + minimum


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = build_model(args.backbone, args.pretrained, device)

    if args.mode == 'real':
        scenes, ids = load_real_protocol(args.real_config)
        if args.sequences:
            requested = set(args.sequences)
            scenes = [scene for scene in scenes if scene in requested]
            missing = sorted(requested.difference(scenes))
            if missing:
                raise ValueError(f'Real sequences not found in {args.real_config}: {missing}')
        loader = load_real_passive
    else:
        scenes = []
        for split in args.splits:
            scenes.extend(read_split(args.data_root / f'{split}.txt'))
        scenes = sorted(set(scenes))
        ids = active_ids(args.active_stride, args.frame_count)
        loader = load_passive

    written = 0
    copied = 0
    skipped_boundary = 0
    for scene in scenes:
        for frame_id in ids:
            output_path = args.output_dir / scene / f'{frame_id:03d}.npy'
            if output_path.is_file():
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if 1 < frame_id < args.frame_count:
                prediction = estimate_source_off(
                    model,
                    loader(args.data_root, scene, frame_id - 1),
                    loader(args.data_root, scene, frame_id + 1),
                    device,
                )
                np.save(output_path, prediction.astype(np.float32))
                written += 1
            else:
                if args.mode == 'real' and not real_has_explicit_source_off(args.data_root, scene):
                    skipped_boundary += 1
                    continue
                np.save(output_path, loader(args.data_root, scene, frame_id).astype(np.float32))
                copied += 1

    manifest = {
        'stage': 'source_off_passive_state_estimation',
        'mode': args.mode,
        'backbone': args.backbone,
        'pretrained': public_path(args.pretrained),
        'data_root': public_path(args.data_root),
        'output_dir': public_path(args.output_dir),
        'splits': args.splits,
        'scene_count': len(scenes),
        'active_stride': args.active_stride,
        'frame_count': args.frame_count,
        'active_ids': ids,
        'estimated_frames': written,
        'copied_boundary_frames': copied,
        'skipped_boundary_frames': skipped_boundary,
        'format': '<output_dir>/<scene>/<frame_id>.npy stores S_hat^off in the passive-frame scale',
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output_dir': public_path(args.output_dir), 'estimated': written, 'copied_boundary': copied, 'skipped_boundary': skipped_boundary}), flush=True)


if __name__ == '__main__':
    main()
