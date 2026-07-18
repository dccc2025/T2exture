from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


def read_split(path: Path) -> list[str]:
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes:
        raise ValueError(f"Empty split: {path}")
    return scenes


class _SceneStore:
    def __init__(self, root: Path, scenes: Iterable[str]):
        self.root = root / 'sim'
        self.scenes = list(scenes)
        self.stats: dict[tuple[str, str], tuple[float, float]] = {}
        for scene in self.scenes:
            for modality in ('texture', 'passive'):
                directory = self.root / scene / modality
                frames = sorted(directory.glob('*.npy'), key=lambda p: int(p.stem))
                if [int(p.stem) for p in frames] != list(range(1, 181)):
                    raise ValueError(f"{scene}/{modality} must contain 001.npy--180.npy")
                lo, hi = float('inf'), float('-inf')
                for frame in frames:
                    image = np.load(frame, mmap_mode='r')
                    lo, hi = min(lo, float(image.min())), max(hi, float(image.max()))
                if not np.isfinite(lo) or not np.isfinite(hi):
                    raise ValueError(f"Non-finite range in {scene}/{modality}: {lo}, {hi}")
                # A zero texture sequence is a valid background-only scene.
                # Preserve it as zeros rather than rejecting the whole split.
                if hi <= lo:
                    hi = lo + 1.0
                self.stats[(scene, modality)] = (lo, hi)

    def load(self, scene: str, modality: str, frame_id: int) -> np.ndarray:
        image = np.load(self.root / scene / modality / f'{frame_id:03d}.npy').astype(np.float32)
        lo, hi = self.stats[(scene, modality)]
        return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def _crop(images: list[np.ndarray], crop_size: int | None) -> list[np.ndarray]:
    if crop_size is None:
        return images
    h, w = images[0].shape
    if crop_size > min(h, w):
        raise ValueError(f"crop_size={crop_size} exceeds image shape {(h, w)}")
    top = np.random.randint(0, h - crop_size + 1)
    left = np.random.randint(0, w - crop_size + 1)
    return [image[top:top + crop_size, left:left + crop_size] for image in images]


class TextureIntervalDataset(Dataset):
    """One target texture frame for each 10-frame keyframe interval."""
    def __init__(self, root: Path, split_file: Path, crop_size: int | None = None):
        self.store = _SceneStore(root, read_split(split_file))
        self.scenes = self.store.scenes
        self.crop_size = crop_size
        self.items = [(scene, left, offset)
                      for scene in self.scenes
                      for left in range(1, 171, 10)
                      for offset in range(1, 10)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        scene, left, offset = self.items[index]
        right, target_id = left + 10, left + offset
        x0 = self.store.load(scene, 'texture', left)
        x1 = self.store.load(scene, 'texture', right)
        p0 = self.store.load(scene, 'passive', left)
        p1 = self.store.load(scene, 'passive', right)
        p = self.store.load(scene, 'passive', target_id)
        target = self.store.load(scene, 'texture', target_id)
        x0, x1, p0, p1, p, target = _crop([x0, x1, p0, p1, p, target], self.crop_size)
        to_tensor = lambda x: torch.from_numpy(x.copy()).unsqueeze(0)
        return {'scene': scene, 'frame_ids': (left, right, target_id),
                'x0': to_tensor(x0), 'x1': to_tensor(x1), 'p': to_tensor(p),
                'p0': to_tensor(p0), 'p1': to_tensor(p1),
                'target': to_tensor(target), 'time': torch.tensor(offset / 10.0)}


class TextureSequenceDataset(Dataset):
    """One endpoint pair with all nine intermediate passive/texture frames."""
    def __init__(self, root: Path, split_file: Path, crop_size: int | None = None):
        self.store = _SceneStore(root, read_split(split_file))
        self.scenes = self.store.scenes
        self.crop_size = crop_size
        self.items = [(scene, left) for scene in self.scenes for left in range(1, 171, 10)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        scene, left = self.items[index]
        right = left + 10
        images = [self.store.load(scene, 'texture', left), self.store.load(scene, 'texture', right),
                  self.store.load(scene, 'passive', left), self.store.load(scene, 'passive', right)]
        images += [self.store.load(scene, 'passive', left + offset) for offset in range(1, 10)]
        images += [self.store.load(scene, 'texture', left + offset) for offset in range(1, 10)]
        images = _crop(images, self.crop_size)
        to_tensor = lambda x: torch.from_numpy(x.copy()).unsqueeze(0)
        return {'scene': scene, 'left_id': left, 'x0': to_tensor(images[0]), 'x1': to_tensor(images[1]),
                'p0': to_tensor(images[2]), 'p1': to_tensor(images[3]),
                'p': torch.stack([to_tensor(x) for x in images[4:13]]),
                'target': torch.stack([to_tensor(x) for x in images[13:22]]),
                'times': torch.arange(1, 10, dtype=torch.float32) / 10.0}
