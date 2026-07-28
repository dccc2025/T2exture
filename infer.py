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

from config import resolve_sample_passive_context
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
    parser.add_argument('--checkpoint', type=Path, default=None, help='Local T2exture checkpoint. If omitted, download from --hf-repo.')
    parser.add_argument('--hf-repo', default='chenjiashuo/T2exture_model')
    parser.add_argument('--pretrained', type=Path, default=None, help='Optional AMT init checkpoint for legacy partial checkpoints.')
    parser.add_argument('--data-root', type=Path, default=Path('datasets'))
    parser.add_argument('--config', type=Path, default=Path('train.yaml'))
    parser.add_argument('--real-config', type=Path, default=Path('configs/formal/table06/real-benchmark.yaml'))
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='test')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cuda')
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
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError('Install huggingface_hub or pass --checkpoint explicitly.') from exc
    downloaded = hf_hub_download(repo_id=args.hf_repo, filename=VARIANTS[args.variant]['filename'])
    return Path(downloaded)


def to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().float().clamp(0.0, 1.0).cpu().numpy()
    return np.round(np.squeeze(array) * 255.0).astype(np.uint8)


def save_png(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(image), mode='L').save(path)


def sample_name(left_id: int, target_id: int, right_id: int) -> str:
    return f'{left_id:03d}_{target_id:03d}_{right_id:03d}'


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
    passive_context = int(config.get('passive_context', read_yaml(args.config).get('passive_context', 4)))
    backbone = VARIANTS[args.variant]['backbone']
    model = build_t2texture_model(backbone, args.pretrained, passive_context).to(device)
    model.load_state_dict(extract_model_state(checkpoint), strict=True)
    model.eval()
    return model


@torch.no_grad()
def infer_synthetic(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> None:
    config = read_yaml(args.config)
    passive_context = int(config.get('passive_context', 4))
    sample_context = resolve_sample_passive_context(config, passive_context)
    dataset = TextureDataset(
        args.data_root,
        args.data_root / f'{args.split}.txt',
        passive_context=passive_context,
        pseudo_flow_root=Path(config['pseudo_flow_dir']) if config.get('pseudo_flow_dir') else None,
        active_stride=int(config.get('active_stride', 10)),
        sample_passive_context=sample_context,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = move_batch(batch, device)
        pred = model(batch['texture0'], batch['texture1'], batch['time'], batch['passive_context'], return_flow=False)['prediction']
        scenes = [str(item) for item in batch_values(batch['scene'])]
        left_ids = [int(item) for item in batch_values(batch['left_id'])]
        target_ids = [int(item) for item in batch_values(batch['target_id'])]
        right_ids = [int(item) for item in batch_values(batch['right_id'])]
        for index, scene in enumerate(scenes):
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
    write_outputs(args.output_dir, rows, {'mode': 'synthetic', 'split': args.split})


def frame_root(sequence_root: Path) -> Path:
    nested = sequence_root / 'frames'
    return nested if nested.is_dir() else sequence_root


def frame_path(data_root: Path, sequence: str, frame_id: int) -> Path:
    path = frame_root(data_root / sequence) / f'{frame_id:03d}.png'
    if not path.is_file():
        raise FileNotFoundError(f'Missing real frame: {path}')
    return path


def load_real_frame(data_root: Path, sequence: str, frame_id: int, device: torch.device) -> torch.Tensor:
    image = Image.open(frame_path(data_root, sequence, frame_id)).convert('L')
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)


def real_samples(active_ids: list[int]) -> list[dict[str, int | float]]:
    samples: list[dict[str, int | float]] = []
    for left_id, right_id in zip(active_ids[:-1], active_ids[1:]):
        for target_id in range(left_id + 1, right_id):
            samples.append({'left_id': left_id, 'target_id': target_id, 'right_id': right_id, 'time': (target_id - left_id) / float(right_id - left_id)})
    return samples


def real_context_ids(target_id: int, active_ids: set[int], context_size: int, min_id: int = 1, max_id: int = 180) -> list[int]:
    candidates = [frame_id for frame_id in range(min_id, max_id + 1) if frame_id != target_id and frame_id not in active_ids]
    candidates.sort(key=lambda frame_id: (abs(frame_id - target_id), frame_id))
    selected = sorted(candidates[:context_size])
    if len(selected) != context_size:
        raise ValueError(f'Only found {len(selected)} passive frames for target {target_id}')
    return selected


@torch.no_grad()
def infer_real(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> None:
    config = read_yaml(args.real_config)
    protocol = config.get('protocol', {})
    active_ids = [int(item) for item in protocol['active_anchor_ids']]
    active_set = set(active_ids)
    passive_context = int(protocol.get('passive_context', 4))
    sequences = args.sequences or [str(item) for item in config['sequences']]
    rows: list[dict[str, Any]] = []
    for sequence in sequences:
        cache: dict[int, torch.Tensor] = {}

        def get_frame(frame_id: int) -> torch.Tensor:
            if frame_id not in cache:
                cache[frame_id] = load_real_frame(args.data_root, sequence, frame_id, device)
            return cache[frame_id]

        for sample in real_samples(active_ids):
            left_id = int(sample['left_id'])
            target_id = int(sample['target_id'])
            right_id = int(sample['right_id'])
            time = torch.tensor([float(sample['time'])], device=device)
            context_ids = real_context_ids(target_id, active_set, passive_context)
            context = torch.cat([get_frame(frame_id) for frame_id in context_ids], dim=1)
            pred = model(get_frame(left_id), get_frame(right_id), time, context, return_flow=False)['prediction']
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
    write_outputs(args.output_dir, rows, {'mode': 'real', 'sequences': sequences})


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
