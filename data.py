"""Dataset loading for AMT-L fine-tuning with four real passive neighbours."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import passive_context_ids
from flow_generation import load_pseudo_flow, validate_pseudo_flow_coverage


def read_split(path: Path) -> list[str]:
    """Read non-empty scene identifiers from a train, validation, or test split file."""
    scenes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not scenes:
        raise ValueError(f'Split file is empty: {path}')
    return scenes


def _frame_path(root: Path, scene: str, modality: str, frame_id: int) -> Path:
    """Return the canonical on-disk path for one numbered modality frame."""
    return root / 'sim' / scene / modality / f'{frame_id:03d}.npy'


def _load_frame(root: Path, scene: str, modality: str, frame_id: int) -> np.ndarray:
    """Load one grayscale frame and reject missing or non-two-dimensional arrays."""
    path = _frame_path(root, scene, modality, frame_id)
    if not path.is_file():
        raise FileNotFoundError(f'Missing {modality} frame: {path}')
    image = np.load(path).astype(np.float32)
    if image.ndim != 2:
        raise ValueError(f'Expected a 2-D {modality} frame at {path}, got {image.shape}')
    return image


def _normalise(image: np.ndarray) -> np.ndarray:
    """Scale one non-negative image to [0, 1] without changing an all-zero image."""
    maximum = float(image.max())
    minimum = float(image.min())
    if maximum <= minimum:
        return np.zeros_like(image, dtype=np.float32)
    return (image - minimum) / (maximum - minimum)


def _crop(images: list[np.ndarray], crop_size: int | None, random_crop: bool) -> list[np.ndarray]:
    """Apply one consistent crop to all image-shaped arrays in a training sample."""
    if crop_size is None:
        return images
    height, width = images[0].shape[-2:]
    if crop_size > min(height, width):
        raise ValueError(f'crop_size={crop_size} exceeds image shape {(height, width)}')
    if random_crop:
        top = np.random.randint(0, height - crop_size + 1)
        left = np.random.randint(0, width - crop_size + 1)
    else:
        top = (height - crop_size) // 2
        left = (width - crop_size) // 2
    return [image[..., top:top + crop_size, left:left + crop_size] for image in images]


class TextureDataset(Dataset):
    """Return one interpolated texture target with four passive neighbours and flow labels."""

    def __init__(
        self,
        root: Path,
        split_file: Path,
        passive_context: int = 4,
        crop_size: int | None = None,
        random_crop: bool = False,
        pseudo_flow_root: Path | None = None,
    ) -> None:
        """Build samples from named scenes and fixed ten-frame texture intervals."""
        self.root = root
        self.scenes = read_split(split_file)
        self.passive_context = passive_context
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.pseudo_flow_root = pseudo_flow_root
        self.items = [
            (scene, left, offset)
            for scene in self.scenes
            for left in range(1, 171, 10)
            for offset in range(1, 10)
            if 1 <= left + offset - passive_context // 2
            and left + offset + passive_context // 2 <= 180
        ]
        validate_pseudo_flow_coverage(self.root, self.items, self.pseudo_flow_root)

    def __len__(self) -> int:
        """Return the number of valid interval-target samples."""
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        """Load texture endpoints, a target, four passive neighbours, and target flow labels."""
        scene, left, offset = self.items[index]
        target_id = left + offset
        right = left + 10
        context_ids = passive_context_ids(target_id, self.passive_context)
        texture0 = _normalise(_load_frame(self.root, scene, 'texture', left))
        texture1 = _normalise(_load_frame(self.root, scene, 'texture', right))
        target = _normalise(_load_frame(self.root, scene, 'texture', target_id))
        passive = [_normalise(_load_frame(self.root, scene, 'passive', frame_id)) for frame_id in context_ids]
        flow = load_pseudo_flow(self.root, scene, left, target_id, right, self.pseudo_flow_root)
        texture0, texture1, target, *rest = _crop([texture0, texture1, target, *passive, flow], self.crop_size, self.random_crop)
        passive, flow = rest[:-1], rest[-1]
        tensor = lambda image: torch.from_numpy(image.copy()).unsqueeze(0)
        return {
            'scene': scene,
            'left_id': left,
            'right_id': right,
            'target_id': target_id,
            'texture0': tensor(texture0),
            'texture1': tensor(texture1),
            'target': tensor(target),
            'passive_context': torch.stack([torch.from_numpy(image.copy()) for image in passive]),
            'flow': torch.from_numpy(flow.copy()),
            'time': torch.tensor([offset / 10.0], dtype=torch.float32),
        }
