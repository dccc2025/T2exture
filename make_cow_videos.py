from __future__ import annotations

"""Render ground-truth and sparse-anchor reconstructions for TT-sim/cow."""

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from tt_data import _SceneStore
from tt_model import GeometryGatedAMT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path('/essfs10/daicheng/TT_sim'))
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('/essfs10/daicheng/TTexture/outputs/edge_adapter_v1/best.pt'))
    parser.add_argument('--output', type=Path,
                        default=Path('/essfs10/daicheng/TTexture/outputs/cow_6s_video'))
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def write_gray_png(image: np.ndarray, path: Path) -> None:
    encoded = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    if not cv2.imwrite(str(path), encoded):
        raise RuntimeError(f'Failed to write {path}')


def encode_mp4(frame_dir: Path, video_path: Path) -> None:
    command = [
        'ffmpeg', '-y', '-loglevel', 'error', '-framerate', '30',
        '-i', str(frame_dir / '%03d.png'), '-c:v', 'libx264', '-preset', 'medium',
        '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video_path),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if not shutil.which('ffmpeg'):
        raise RuntimeError('ffmpeg is required to encode the requested videos')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # These 18 observed texture anchors cover all 180 frames exactly:
    # 16 regular ten-frame intervals, followed by the final 19-frame interval.
    anchors = list(range(1, 162, 10)) + [180]
    assert len(anchors) == 18
    store = _SceneStore(args.data_root, ['cow'])
    output_gt, output_pred = args.output / 'ground_truth_frames', args.output / 'predicted_frames'
    for directory in (output_gt, output_pred):
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob('*.png'):
            stale.unlink()

    gt = {frame_id: store.load('cow', 'texture', frame_id) for frame_id in range(1, 181)}
    prediction: dict[int, np.ndarray] = {frame_id: gt[frame_id] for frame_id in anchors}

    model = GeometryGatedAMT().to(device)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    model.eval()

    with torch.inference_mode():
        for left, right in zip(anchors[:-1], anchors[1:]):
            x0 = torch.from_numpy(gt[left]).to(device).unsqueeze(0).unsqueeze(0)
            x1 = torch.from_numpy(gt[right]).to(device).unsqueeze(0).unsqueeze(0)
            p0 = torch.from_numpy(store.load('cow', 'passive', left)).to(device).unsqueeze(0).unsqueeze(0)
            p1 = torch.from_numpy(store.load('cow', 'passive', right)).to(device).unsqueeze(0).unsqueeze(0)
            span = right - left
            for frame_id in range(left + 1, right):
                passive = torch.from_numpy(store.load('cow', 'passive', frame_id)).to(device).unsqueeze(0).unsqueeze(0)
                time = torch.full((1, 1, 1, 1), (frame_id - left) / span, device=device)
                prediction[frame_id] = model(x0, x1, time, p0, p1, passive).mean(1).squeeze(0).clamp(0, 1).cpu().numpy()

    if sorted(prediction) != list(range(1, 181)):
        raise RuntimeError('The reconstruction must contain exactly frames 001--180')
    if sum(frame_id in anchors for frame_id in prediction) != 18:
        raise RuntimeError('Unexpected number of copied texture anchors')

    for frame_id in range(1, 181):
        write_gray_png(gt[frame_id], output_gt / f'{frame_id:03d}.png')
        write_gray_png(prediction[frame_id], output_pred / f'{frame_id:03d}.png')
    encode_mp4(output_gt, args.output / 'cow_texture_ground_truth_6s.mp4')
    encode_mp4(output_pred, args.output / 'cow_texture_predicted_6s.mp4')
    np.save(args.output / 'anchor_frame_ids.npy', np.asarray(anchors, dtype=np.int16))
    (args.output / 'README.txt').write_text(
        'Both videos are 180 frames at 30 fps (6 seconds).\\n'
        'Anchors: 001, 011, ..., 161, 180 (18 observed texture frames).\\n'
        'All remaining 162 frames were inferred using the matching passive frame.\\n'
    )
    print(f'Wrote videos to {args.output}')


if __name__ == '__main__':
    main()
