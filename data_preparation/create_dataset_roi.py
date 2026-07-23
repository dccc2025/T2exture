"""Create the cropped ROI dataset used by formal synthetic experiments."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ('train', 'valid', 'test')


def read_split(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not scenes:
        raise ValueError(f'Split file is empty: {path}')
    return scenes


def numbered_npy_files(path: Path) -> list[Path]:
    files = sorted(path.glob('*.npy'))
    if not files:
        raise FileNotFoundError(f'No .npy files found under {path}')
    return files


def load_frame(path: Path) -> np.ndarray:
    image = np.load(path)
    if image.ndim != 2:
        raise ValueError(f'Expected a 2-D frame at {path}, got {image.shape}')
    return image


def texture_bbox(texture_dir: Path, threshold_ratio: float) -> tuple[tuple[int, int, int, int], tuple[int, int], int]:
    files = numbered_npy_files(texture_dir)
    shape: tuple[int, int] | None = None
    scene_max = 0.0

    for file in files:
        image = load_frame(file)
        if shape is None:
            shape = image.shape
        elif image.shape != shape:
            raise ValueError(f'Inconsistent texture shape in {texture_dir}: {image.shape} vs {shape}')
        scene_max = max(scene_max, float(np.nanmax(image)))

    if shape is None:
        raise FileNotFoundError(f'No texture frames found under {texture_dir}')
    threshold = scene_max * threshold_ratio
    y_min, x_min = shape
    y_max, x_max = 0, 0
    detected = 0

    for file in files:
        image = load_frame(file)
        mask = image > threshold
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        y_min = min(y_min, int(ys.min()))
        y_max = max(y_max, int(ys.max()) + 1)
        x_min = min(x_min, int(xs.min()))
        x_max = max(x_max, int(xs.max()) + 1)
        detected += 1

    if detected == 0:
        return (0, 0, shape[0], shape[1]), shape, detected
    return (y_min, x_min, y_max, x_max), shape, detected


def fixed_crop_box(
    bbox: tuple[int, int, int, int],
    source_shape: tuple[int, int],
    output_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    y_min, x_min, y_max, x_max = bbox
    source_h, source_w = source_shape
    output_h, output_w = output_shape
    bbox_h = y_max - y_min
    bbox_w = x_max - x_min
    if bbox_h > output_h or bbox_w > output_w:
        raise ValueError(
            f'ROI {bbox_h}x{bbox_w} exceeds requested crop {output_h}x{output_w}; '
            'increase --height/--width or adjust threshold.'
        )

    center_y = (y_min + y_max) / 2.0
    center_x = (x_min + x_max) / 2.0
    top = int(round(center_y - output_h / 2.0))
    left = int(round(center_x - output_w / 2.0))
    top = max(0, min(top, max(0, source_h - output_h)))
    left = max(0, min(left, max(0, source_w - output_w)))
    return top, left, top + output_h, left + output_w


def crop_array(array: np.ndarray, box: tuple[int, int, int, int], output_shape: tuple[int, int]) -> np.ndarray:
    top, left, bottom, right = box
    cropped = array[..., top:min(bottom, array.shape[-2]), left:min(right, array.shape[-1])]
    pad_h = output_shape[0] - cropped.shape[-2]
    pad_w = output_shape[1] - cropped.shape[-1]
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f'Cropped array is larger than requested output shape: {cropped.shape}')
    if pad_h or pad_w:
        pad_width = [(0, 0)] * cropped.ndim
        pad_width[-2] = (0, pad_h)
        pad_width[-1] = (0, pad_w)
        cropped = np.pad(cropped, pad_width, mode='constant')
    return cropped


def save_cropped_frames(source_dir: Path, output_dir: Path, box: tuple[int, int, int, int], output_shape: tuple[int, int]) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for file in numbered_npy_files(source_dir):
        image = load_frame(file)
        np.save(output_dir / file.name, crop_array(image, box, output_shape))
        count += 1
    return count


def save_cropped_flow(source_dir: Path, output_dir: Path, box: tuple[int, int, int, int], output_shape: tuple[int, int]) -> int:
    if not source_dir.is_dir():
        raise FileNotFoundError(f'Missing source flow directory: {source_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for file in sorted(source_dir.glob('*.npz')):
        with np.load(file) as data:
            arrays = {key: data[key] for key in data.files}
        for key in ('flow0', 'flow1'):
            if key not in arrays:
                raise ValueError(f'Flow file {file} is missing key {key}')
            flow = arrays[key]
            if flow.ndim != 3 or flow.shape[0] != 2:
                raise ValueError(f'Expected {key} to have shape [2, H, W] at {file}, got {flow.shape}')
            arrays[key] = crop_array(flow, box, output_shape)
        np.savez_compressed(output_dir / file.name, **arrays)
        count += 1
    return count


def normalise_to_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)
    minimum = float(image[finite].min())
    maximum = float(image[finite].max())
    if maximum <= minimum:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - minimum) / (maximum - minimum) * 255.0, 0, 255).astype(np.uint8)


def write_preview(texture_dir: Path, preview_path: Path, box: tuple[int, int, int, int], output_shape: tuple[int, int]) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return

    files = numbered_npy_files(texture_dir)
    projection = None
    for file in files:
        image = load_frame(file).astype(np.float32)
        projection = image if projection is None else np.maximum(projection, image)
    if projection is None:
        return
    crop = crop_array(projection, box, output_shape)
    rgb = np.repeat(normalise_to_uint8(crop)[..., None], 3, axis=2)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, output_shape[1] - 1, output_shape[0] - 1), outline=(255, 0, 0), width=3)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(preview_path)


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if not output_root.exists():
        output_root.mkdir(parents=True)
        return
    if any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f'{output_root} already exists and is not empty; pass --overwrite to replace it')
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def build_dataset_roi(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_shape = (args.height, args.width)
    source_flow_root = source_root / 'flow' / args.flow_set
    output_flow_root = output_root / 'flow' / args.flow_set

    if not source_root.is_dir():
        raise FileNotFoundError(f'Missing source dataset root: {source_root}')
    if not source_flow_root.is_dir():
        raise FileNotFoundError(f'Missing source flow set: {source_flow_root}')
    prepare_output_root(output_root, args.overwrite)

    split_scenes: dict[str, list[str]] = {}
    scenes: list[str] = []
    for split in SPLITS:
        split_file = source_root / f'{split}.txt'
        split_scenes[split] = read_split(split_file)
        (output_root / split_file.name).write_text('\n'.join(split_scenes[split]) + '\n', encoding='utf-8')
        for scene in split_scenes[split]:
            if scene not in scenes:
                scenes.append(scene)

    manifest: dict[str, Any] = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_root': str(source_root),
        'output_root': str(output_root),
        'output_shape_hw': list(output_shape),
        'threshold_ratio': args.threshold_ratio,
        'flow_set': args.flow_set,
        'splits': {split: {'count': len(split_scenes[split]), 'scenes': split_scenes[split]} for split in SPLITS},
        'scenes': {},
    }

    for scene in scenes:
        texture_dir = source_root / 'sim' / scene / 'texture'
        passive_dir = source_root / 'sim' / scene / 'passive'
        bbox, source_shape, detected_frames = texture_bbox(texture_dir, args.threshold_ratio)
        box = fixed_crop_box(bbox, source_shape, output_shape)
        if not passive_dir.is_dir():
            raise FileNotFoundError(f'Missing passive directory: {passive_dir}')

        texture_count = save_cropped_frames(texture_dir, output_root / 'sim' / scene / 'texture', box, output_shape)
        passive_count = save_cropped_frames(passive_dir, output_root / 'sim' / scene / 'passive', box, output_shape)
        flow_count = save_cropped_flow(source_flow_root / scene, output_flow_root / scene, box, output_shape)
        write_preview(texture_dir, output_root / 'roi_preview' / f'{scene}.png', box, output_shape)

        y_min, x_min, y_max, x_max = bbox
        top, left, bottom, right = box
        manifest['scenes'][scene] = {
            'source_shape_hw': list(source_shape),
            'bbox_yxyx': [y_min, x_min, y_max, x_max],
            'bbox_shape_hw': [y_max - y_min, x_max - x_min],
            'crop_yxyx': [top, left, bottom, right],
            'detected_texture_frames': detected_frames,
            'texture_frames': texture_count,
            'passive_frames': passive_count,
            'flow_files': flow_count,
        }
        print(f'{scene}: crop={top},{left},{bottom},{right} bbox={y_max - y_min}x{x_max - x_min} flow={flow_count}')

    manifest_path = output_root / 'roi_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path('dataset'))
    parser.add_argument('--output-root', type=Path, default=Path('dataset_roi'))
    parser.add_argument('--height', type=int, default=640)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--threshold-ratio', type=float, default=0.01)
    parser.add_argument('--flow-set', default='s10')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_dataset_roi(args)
    total_flows = sum(scene['flow_files'] for scene in manifest['scenes'].values())
    print(f'Wrote {len(manifest["scenes"])} scenes to {args.output_root} with {total_flows} flow files.')


if __name__ == '__main__':
    main()
