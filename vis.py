"""Create visual PNG sheets and MP4 comparison videos from eval.py outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORMAPS = {
    'gray': (
        (0.00, (0, 0, 0)),
        (1.00, (255, 255, 255)),
    ),
    'texture': (
        (0.00, (0, 0, 0)),
        (0.18, (35, 0, 48)),
        (0.38, (126, 0, 35)),
        (0.62, (220, 63, 0)),
        (0.82, (255, 181, 40)),
        (1.00, (255, 255, 230)),
    ),
    'error': (
        (0.00, (15, 23, 42)),
        (0.20, (37, 99, 235)),
        (0.50, (168, 85, 247)),
        (0.75, (249, 115, 22)),
        (1.00, (255, 209, 102)),
    ),
}


def parse_args() -> argparse.Namespace:
    """Read an eval output directory and visualization controls."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--method-label', default=None)
    parser.add_argument('--scenes', nargs='*', default=None)
    parser.add_argument('--max-frames-per-scene', type=int, default=None)
    parser.add_argument('--panel-width', type=int, default=256)
    parser.add_argument('--image-cmap', choices=['gray', 'texture'], default='texture')
    parser.add_argument('--error-cmap', choices=['gray', 'error'], default='error')
    parser.add_argument('--error-percentile', type=float, default=99.0)
    parser.add_argument('--fps', type=int, default=12)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    """Load one JSON object if the file exists, otherwise return an empty mapping."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def _read_frame_rows(eval_dir: Path) -> list[dict[str, str]]:
    """Load frame-level records exported by eval.py."""
    frame_csv = eval_dir / 'frame.csv'
    if not frame_csv.is_file():
        raise FileNotFoundError(f'Missing eval frame table: {frame_csv}')
    with frame_csv.open(newline='') as handle:
        return list(csv.DictReader(handle))


def _normalise(image: np.ndarray) -> np.ndarray:
    """Scale one non-negative image to [0, 1] for visualization."""
    image = image.astype(np.float32)
    maximum = float(image.max())
    minimum = float(image.min())
    if maximum <= minimum:
        return np.zeros_like(image, dtype=np.float32)
    return (image - minimum) / (maximum - minimum)


def _apply_colormap(gray_image: Image.Image, cmap: str) -> Image.Image:
    """Map one grayscale image to an RGB display colormap."""
    if cmap == 'gray':
        return gray_image.convert('RGB')
    anchors = COLORMAPS[cmap]
    values = np.asarray(gray_image.convert('L'), dtype=np.float32) / 255.0
    positions = np.array([anchor[0] for anchor in anchors], dtype=np.float32)
    colors = np.array([anchor[1] for anchor in anchors], dtype=np.float32)
    rgb = np.stack([np.interp(values, positions, colors[:, channel]) for channel in range(3)], axis=-1)
    return Image.fromarray(np.round(rgb).astype(np.uint8), mode='RGB')


def _stretch_contrast(gray_image: Image.Image, percentile: float) -> Image.Image:
    """Scale a low-dynamic-range error image for readable visualization."""
    values = np.asarray(gray_image.convert('L'), dtype=np.float32) / 255.0
    high = float(np.percentile(values, percentile))
    if high <= 1e-6:
        stretched = np.zeros_like(values)
    else:
        stretched = np.clip(values / high, 0.0, 1.0)
    return Image.fromarray(np.round(stretched * 255.0).astype(np.uint8), mode='L')


def _load_npy_image(root: Path, scene: str, modality: str, frame_id: int, cmap: str) -> Image.Image:
    """Load one dataset frame as a grayscale PIL image."""
    path = root / 'sim' / scene / modality / f'{frame_id:03d}.npy'
    if not path.is_file():
        raise FileNotFoundError(f'Missing {modality} frame: {path}')
    image = np.round(_normalise(np.load(path)) * 255.0).astype(np.uint8)
    return _apply_colormap(Image.fromarray(image, mode='L'), cmap)


def _load_png(path: Path, cmap: str, percentile: float | None = None) -> Image.Image:
    """Load one exported PNG frame as RGB."""
    if not path.is_file():
        raise FileNotFoundError(f'Missing visualization source: {path}')
    image = Image.open(path).convert('L')
    if percentile is not None:
        image = _stretch_contrast(image, percentile)
    return _apply_colormap(image, cmap)


def _font(size: int) -> ImageFont.ImageFont:
    """Load a readable UI font with a safe fallback."""
    for name in ('arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _resize_panel(image: Image.Image, panel_width: int) -> Image.Image:
    """Resize one panel to a shared width while preserving aspect ratio."""
    width, height = image.size
    panel_height = max(1, round(height * panel_width / width))
    return image.resize((panel_width, panel_height), Image.BICUBIC)


def _label_panel(image: Image.Image, label: str, font: ImageFont.ImageFont) -> Image.Image:
    """Add a compact title bar above one image panel."""
    label_height = 34
    panel = Image.new('RGB', (image.width, image.height + label_height), (18, 18, 18))
    panel.paste(image, (0, label_height))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 8), label, fill=(245, 245, 245), font=font)
    return panel


def _compose_row(panels: list[tuple[str, Image.Image]], panel_width: int) -> Image.Image:
    """Compose labeled image panels into one horizontal comparison row."""
    font = _font(18)
    labeled = [_label_panel(_resize_panel(image, panel_width), label, font) for label, image in panels]
    height = max(image.height for image in labeled)
    width = sum(image.width for image in labeled)
    canvas = Image.new('RGB', (width, height), (0, 0, 0))
    left = 0
    for image in labeled:
        canvas.paste(image, (left, 0))
        left += image.width
    return canvas


def _sample_name(row: dict[str, str]) -> str:
    """Return the stable endpoint-target filename stem for one frame row."""
    return f"{int(row['left_id']):03d}_{int(row['target_id']):03d}_{int(row['right_id']):03d}"


def _resolve_data_root(args: argparse.Namespace, manifest: dict[str, Any]) -> Path:
    """Resolve the dataset root from CLI args or eval manifest metadata."""
    if args.data_root is not None:
        return args.data_root
    if 'data_root' in manifest:
        return Path(manifest['data_root'])
    raise ValueError('Pass --data-root because manifest.json does not record data_root')


def _select_rows(rows: list[dict[str, str]], scenes: list[str] | None, max_frames_per_scene: int | None) -> dict[str, list[dict[str, str]]]:
    """Group rows by scene and apply optional scene/frame filters."""
    selected: dict[str, list[dict[str, str]]] = defaultdict(list)
    scene_filter = set(scenes) if scenes else None
    for row in rows:
        scene = row['scene']
        if scene_filter is not None and scene not in scene_filter:
            continue
        if max_frames_per_scene is not None and len(selected[scene]) >= max_frames_per_scene:
            continue
        selected[scene].append(row)
    if not selected:
        raise ValueError('No frame rows matched the requested visualization filters')
    return dict(sorted(selected.items()))


def _comparison_image(
    eval_dir: Path,
    data_root: Path,
    row: dict[str, str],
    method_label: str,
    panel_width: int,
    image_cmap: str,
    error_cmap: str,
    error_percentile: float,
) -> Image.Image:
    """Build one visual comparison image for a frame-level eval record."""
    scene = row['scene']
    left_id = int(row['left_id'])
    target_id = int(row['target_id'])
    right_id = int(row['right_id'])
    panels = [
        ('Input 0', _load_npy_image(data_root, scene, 'texture', left_id, image_cmap)),
        ('Input 1', _load_npy_image(data_root, scene, 'texture', right_id, image_cmap)),
        (method_label, _load_png(eval_dir / row['pred_path'], image_cmap)),
        ('GT', _load_npy_image(data_root, scene, 'texture', target_id, image_cmap)),
        ('Abs Error', _load_png(eval_dir / row['err_path'], error_cmap, error_percentile)),
    ]
    return _compose_row(panels, panel_width)


def _write_video(path: Path, frames: list[Path], fps: int) -> None:
    """Write a list of PNG comparison frames to one MP4 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, macro_block_size=1) as writer:
        for frame_path in frames:
            frame = Image.open(frame_path).convert('RGB')
            width, height = frame.size
            if width % 2 or height % 2:
                padded = Image.new('RGB', (width + width % 2, height + height % 2), (0, 0, 0))
                padded.paste(frame, (0, 0))
                frame = padded
            writer.append_data(np.asarray(frame))


def main() -> None:
    """Create per-frame PNG sheets and per-scene plus aggregate MP4 videos."""
    args = parse_args()
    manifest = _read_json(args.eval_dir / 'manifest.json')
    output_dir = args.output_dir or args.eval_dir / 'vis'
    png_dir = output_dir / 'png'
    video_dir = output_dir / 'video'
    png_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    data_root = _resolve_data_root(args, manifest)
    method_label = args.method_label or str(manifest.get('backbone', 'T2exture'))
    rows_by_scene = _select_rows(_read_frame_rows(args.eval_dir), args.scenes, args.max_frames_per_scene)

    all_frames: list[Path] = []
    scene_videos: dict[str, str] = {}
    for scene, rows in rows_by_scene.items():
        scene_frames: list[Path] = []
        for row in rows:
            name = _sample_name(row)
            image = _comparison_image(
                args.eval_dir,
                data_root,
                row,
                method_label,
                args.panel_width,
                args.image_cmap,
                args.error_cmap,
                args.error_percentile,
            )
            out_path = png_dir / f'{scene}_{name}_cmp.png'
            image.save(out_path)
            scene_frames.append(out_path)
            all_frames.append(out_path)
        scene_video = video_dir / f'{scene}_cmp.mp4'
        _write_video(scene_video, scene_frames, args.fps)
        scene_videos[scene] = str(scene_video.relative_to(output_dir)).replace('\\', '/')

    all_video = video_dir / 'all_cmp.mp4'
    _write_video(all_video, all_frames, args.fps)
    visual_manifest = {
        'eval_dir': str(args.eval_dir.resolve()),
        'data_root': str(data_root.resolve()),
        'frames': len(all_frames),
        'scenes': {scene: len(rows) for scene, rows in rows_by_scene.items()},
        'style': {
            'image_cmap': args.image_cmap,
            'error_cmap': args.error_cmap,
            'error_percentile': args.error_percentile,
        },
        'artifacts': {
            'png': 'png/',
            'scene_videos': scene_videos,
            'all_video': str(all_video.relative_to(output_dir)).replace('\\', '/'),
        },
    }
    (output_dir / 'manifest.json').write_text(json.dumps(visual_manifest, indent=2) + '\n')
    print(json.dumps({'frames': len(all_frames), 'all_video': visual_manifest['artifacts']['all_video']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
