"""Run T2exture inference on synthetic test splits or real sequences."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from config import passive_context_ids, resolve_sample_passive_context
from data import TextureDataset
from model import build_t2texture_model
from utils.checkpoint import extract_model_state, torch_load_portable


VARIANTS = {
    's': {'backbone': 'amt-s', 'filename': 't2exture-s.pt'},
    'l': {'backbone': 'amt-l', 'filename': 't2exture-l.pt'},
    'g': {'backbone': 'amt-g', 'filename': 't2exture-g.pt'},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['synthetic', 'real'], required=True)
    parser.add_argument('--variant', choices=sorted(VARIANTS), default='l')
    parser.add_argument('--checkpoint', type=Path, default=None, help='Local T2exture checkpoint.')
    parser.add_argument('--pretrained', type=Path, default=None, help='Optional AMT init checkpoint for legacy partial checkpoints.')
    parser.add_argument('--data-root', type=Path, default=Path('datasets'))
    parser.add_argument('--config', type=Path, default=Path('train.yaml'))
    parser.add_argument('--real-config', type=Path, default=Path('configs/real.yaml'))
    parser.add_argument('--source-off-root', type=Path, default=None, help='Optional completed S_off cache used for passive context at active keyframes.')
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='test')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-samples', type=int, default=None, help='Optional cap for quick smoke inference.')
    parser.add_argument('--scenes', nargs='*', default=None, help='Synthetic scenes to run. Defaults to all scenes in --split.')
    parser.add_argument('--sequences', nargs='*', default=None, help='Real sequences to run. Defaults to all sequences in --real-config.')
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Expected a YAML mapping in {path}')
    return data


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    local_candidates = [
        Path('pretrained') / VARIANTS[args.variant]['filename'],
        Path('pretrained') / 't2exture_model' / VARIANTS[args.variant]['filename'],
    ]
    for local in local_candidates:
        if local.is_file():
            return local
    raise FileNotFoundError(
        f'Pass --checkpoint or place {VARIANTS[args.variant]["filename"]} under '
        'pretrained/ or pretrained/t2exture_model/.'
    )


def to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().float().clamp(0.0, 1.0).cpu().numpy()
    return np.round(np.squeeze(array) * 255.0).astype(np.uint8)


def save_png(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(image), mode='L').save(path)


def sample_name(left_id: int, target_id: int, right_id: int) -> str:
    return f'{left_id:03d}_{target_id:03d}_{right_id:03d}'


def resolve_optional_root(base: Path, value: str | Path | None) -> Path | None:
    """Resolve an optional path relative to a dataset root."""
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == base.name:
        return path
    return base / path


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=device.type == 'cuda') if torch.is_tensor(value) else value for key, value in batch.items()}


def batch_values(value: Any) -> list[Any]:
    if torch.is_tensor(value):
        return value.detach().cpu().flatten().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    checkpoint_path = resolve_checkpoint(args)
    checkpoint = torch_load_portable(checkpoint_path, device)
    config = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
    passive_context = int(config.get('passive_context', read_yaml(args.config).get('passive_context', 5)))
    backbone = VARIANTS[args.variant]['backbone']
    model = build_t2texture_model(backbone, args.pretrained, passive_context).to(device)
    model.load_state_dict(extract_model_state(checkpoint), strict=True)
    model.eval()
    return model


@torch.no_grad()
def infer_synthetic(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> None:
    config = read_yaml(args.config)
    passive_context = int(config.get('passive_context', 5))
    sample_context = resolve_sample_passive_context(config, passive_context)
    source_off_root = resolve_optional_root(args.data_root, args.source_off_root or config.get('source_off_dir'))
    dataset = TextureDataset(
        args.data_root,
        args.data_root / f'{args.split}.txt',
        passive_context=passive_context,
        pseudo_flow_root=Path(config['pseudo_flow_dir']) if config.get('pseudo_flow_dir') else None,
        active_stride=int(config.get('active_stride', 10)),
        sample_passive_context=sample_context,
        source_off_root=source_off_root,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    scene_filter = set(args.scenes) if args.scenes else None
    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch['texture0'], batch['texture1'], batch['time'], batch['passive_context'], return_flow=False)['prediction']
        scenes = [str(item) for item in batch_values(batch['scene'])]
        left_ids = [int(item) for item in batch_values(batch['left_id'])]
        target_ids = [int(item) for item in batch_values(batch['target_id'])]
        right_ids = [int(item) for item in batch_values(batch['right_id'])]
        for index, scene in enumerate(scenes):
            if scene_filter is not None and scene not in scene_filter:
                continue
            if args.max_samples is not None and len(rows) >= args.max_samples:
                break
            name = sample_name(left_ids[index], target_ids[index], right_ids[index])
            pred_path = args.output_dir / 'pred' / scene / f'{name}.png'
            save_png(pred[index], pred_path)
            rows.append(
                {
                    'scene': scene,
                    'left_id': left_ids[index],
                    'target_id': target_ids[index],
                    'right_id': right_ids[index],
                    'pred_path': str(pred_path.relative_to(args.output_dir)).replace('\\', '/'),
                }
            )
        if args.max_samples is not None and len(rows) >= args.max_samples:
            break
    write_outputs(args.output_dir, rows, {'mode': 'synthetic', 'split': args.split})


def sequence_has_modality(data_root: Path, sequence: str, modality: str) -> bool:
    """Return whether a real sequence uses an explicit modality folder."""
    return (data_root / sequence / modality).is_dir()


def frame_root(sequence_root: Path, modality: str | None = None, allow_legacy: bool = True) -> Path:
    if modality is not None:
        modality_root = sequence_root / modality
        if modality_root.is_dir():
            return modality_root
        if not allow_legacy:
            raise FileNotFoundError(f'Missing modality folder: {modality_root}')
    nested = sequence_root / 'frames'
    return nested if nested.is_dir() else sequence_root


def frame_path(data_root: Path, sequence: str, frame_id: int, modality: str | None = None, allow_legacy: bool = True) -> Path:
    root = frame_root(data_root / sequence, modality, allow_legacy)
    candidates = [root / f'{frame_id:03d}{suffix}' for suffix in ('.png', '.jpg', '.jpeg', '.npy')]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        checked = ', '.join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f'Missing real frame {frame_id:03d}. Checked: {checked}')
    return path


def source_off_path(source_off_root: Path | None, sequence: str, frame_id: int) -> Path | None:
    if source_off_root is None:
        return None
    root = source_off_root / sequence if (source_off_root / sequence).is_dir() else source_off_root
    candidates = [root / f'{frame_id:03d}{suffix}' for suffix in ('.png', '.jpg', '.jpeg', '.npy')]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def load_real_frame(path: Path, device: torch.device) -> torch.Tensor:
    if path.suffix.lower() == '.npy':
        array = np.load(path).astype(np.float32)
        if array.ndim != 2:
            raise ValueError(f'Expected a 2-D real frame at {path}, got {array.shape}')
        maximum = float(array.max())
        minimum = float(array.min())
        array = (array - minimum) / (maximum - minimum) if maximum > minimum else np.zeros_like(array, dtype=np.float32)
    else:
        image = Image.open(path).convert('L')
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)


def real_texture_anchor(
    data_root: Path,
    source_off_root: Path | None,
    sequence: str,
    frame_id: int,
    device: torch.device,
) -> torch.Tensor:
    """Load a real texture anchor ``X_k`` or construct it as ``[S_k^on - S_k^off]_+``."""
    if sequence_has_modality(data_root, sequence, 'texture'):
        return load_real_frame(frame_path(data_root, sequence, frame_id, 'texture', allow_legacy=False), device)
    if sequence_has_modality(data_root, sequence, 'source_on'):
        source_on = load_real_frame(frame_path(data_root, sequence, frame_id, 'source_on', allow_legacy=False), device)
        source_off = source_off_path(source_off_root, sequence, frame_id)
        if source_off is None and sequence_has_modality(data_root, sequence, 'source_off'):
            source_off = frame_path(data_root, sequence, frame_id, 'source_off', allow_legacy=False)
        if source_off is None and sequence_has_modality(data_root, sequence, 'passive'):
            source_off = frame_path(data_root, sequence, frame_id, 'passive', allow_legacy=False)
        if source_off is None:
            raise FileNotFoundError(f'Missing source-off frame for real texture anchor {sequence}/{frame_id:03d}')
        return (source_on - load_real_frame(source_off, device)).clamp_min(0.0)
    source_on = load_real_frame(frame_path(data_root, sequence, frame_id), device)
    source_off = source_off_path(source_off_root, sequence, frame_id)
    if source_off is not None:
        return (source_on - load_real_frame(source_off, device)).clamp_min(0.0)
    return source_on


def real_samples(active_ids: list[int]) -> list[dict[str, int | float]]:
    samples: list[dict[str, int | float]] = []
    for left_id, right_id in zip(active_ids[:-1], active_ids[1:]):
        for target_id in range(left_id + 1, right_id):
            samples.append({'left_id': left_id, 'target_id': target_id, 'right_id': right_id, 'time': (target_id - left_id) / float(right_id - left_id)})
    return samples


def real_context_ids(target_id: int, context_size: int, min_id: int = 1, max_id: int = 180) -> list[int]:
    ids = list(passive_context_ids(target_id, context_size))
    if ids and (min(ids) < min_id or max(ids) > max_id):
        return []
    return ids


@torch.no_grad()
def infer_real(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> None:
    config = read_yaml(args.real_config)
    protocol = config.get('protocol', {})
    active_ids = [int(item) for item in protocol['active_anchor_ids']]
    active_set = set(active_ids)
    passive_context = int(protocol.get('passive_context', 5))
    source_off_root = resolve_optional_root(args.data_root, args.source_off_root or protocol.get('source_off_dir') or config.get('source_off_dir'))
    sequences = args.sequences or [str(item) for item in config['sequences']]
    rows: list[dict[str, Any]] = []
    skipped = 0
    for sequence in sequences:
        texture_cache: dict[int, torch.Tensor] = {}
        passive_cache: dict[int, torch.Tensor] = {}

        def get_texture(frame_id: int) -> torch.Tensor:
            if frame_id not in texture_cache:
                texture_cache[frame_id] = real_texture_anchor(args.data_root, source_off_root, sequence, frame_id, device)
            return texture_cache[frame_id]

        def can_load_passive(frame_id: int) -> bool:
            if source_off_path(source_off_root, sequence, frame_id) is not None:
                return True
            if sequence_has_modality(args.data_root, sequence, 'passive') or sequence_has_modality(args.data_root, sequence, 'source_off'):
                return True
            return frame_id not in active_set

        def get_passive(frame_id: int) -> torch.Tensor:
            if frame_id not in passive_cache:
                source_off = source_off_path(source_off_root, sequence, frame_id) if frame_id in active_set else None
                if source_off is not None:
                    path = source_off
                elif sequence_has_modality(args.data_root, sequence, 'source_off'):
                    path = frame_path(args.data_root, sequence, frame_id, 'source_off', allow_legacy=False)
                elif sequence_has_modality(args.data_root, sequence, 'passive'):
                    path = frame_path(args.data_root, sequence, frame_id, 'passive', allow_legacy=False)
                else:
                    path = frame_path(args.data_root, sequence, frame_id)
                passive_cache[frame_id] = load_real_frame(path, device)
            return passive_cache[frame_id]

        for sample in real_samples(active_ids):
            if args.max_samples is not None and len(rows) >= args.max_samples:
                break
            left_id = int(sample['left_id'])
            target_id = int(sample['target_id'])
            right_id = int(sample['right_id'])
            time = torch.tensor([float(sample['time'])], device=device)
            context_ids = real_context_ids(target_id, passive_context)
            if len(context_ids) != passive_context:
                skipped += 1
                continue
            if not all(can_load_passive(frame_id) for frame_id in context_ids):
                skipped += 1
                continue
            context = torch.cat([get_passive(frame_id) for frame_id in context_ids], dim=1)
            pred = model(get_texture(left_id), get_texture(right_id), time, context, return_flow=False)['prediction']
            name = sample_name(left_id, target_id, right_id)
            pred_path = args.output_dir / sequence / 'pred' / f'{name}.png'
            save_png(pred[0], pred_path)
            rows.append(
                {
                    'sequence': sequence,
                    'left_id': left_id,
                    'target_id': target_id,
                    'right_id': right_id,
                    'time': float(sample['time']),
                    'passive_context_ids': ' '.join(f'{frame_id:03d}' for frame_id in context_ids),
                    'pred_path': str(pred_path.relative_to(args.output_dir)).replace('\\', '/'),
                }
            )
        if args.max_samples is not None and len(rows) >= args.max_samples:
            break
    write_outputs(args.output_dir, rows, {'mode': 'real', 'sequences': sequences, 'skipped_boundary_frames': skipped})


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output_dir / 'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = {**manifest, 'frames': len(rows), 'prediction_csv': 'predictions.csv'}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'frames': len(rows), 'output_dir': str(output_dir)}, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = build_model(args, device)
    if args.mode == 'synthetic':
        infer_synthetic(args, model, device)
    else:
        infer_real(args, model, device)


if __name__ == '__main__':
    main()
