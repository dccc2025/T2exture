"""Dataset loading for AMT fine-tuning with centered passive context."""

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


def _auxiliary_frame_path(root: Path, scene: str, frame_id: int) -> Path:
    """Return the cache path for one completed source-off passive frame."""
    return root / scene / f'{frame_id:03d}.npy'


def _load_frame(root: Path, scene: str, modality: str, frame_id: int) -> np.ndarray:
    """Load one grayscale frame and reject missing or non-two-dimensional arrays."""
    path = _frame_path(root, scene, modality, frame_id)
    if not path.is_file():
        raise FileNotFoundError(f'Missing {modality} frame: {path}')
    image = np.load(path).astype(np.float32)
    if image.ndim != 2:
        raise ValueError(f'Expected a 2-D {modality} frame at {path}, got {image.shape}')
    return image


def _load_source_on_frame(root: Path, scene: str, frame_id: int) -> np.ndarray:
    """Load raw ``S^on`` when available, with a legacy residual fallback."""
    source_on = _frame_path(root, scene, 'source_on', frame_id)
    source_on_dir = root / 'sim' / scene / 'source_on'
    if source_on_dir.is_dir() and not source_on.is_file():
        raise FileNotFoundError(f'Missing source-on frame: {source_on}')
    if source_on.is_file():
        image = np.load(source_on).astype(np.float32)
        if image.ndim != 2:
            raise ValueError(f'Expected a 2-D source-on frame at {source_on}, got {image.shape}')
        return image

    # Older prepared roots store X and S^off only. This is exact when X is a
    # nonnegative additive source residual, which is the synthetic protocol.
    return _load_frame(root, scene, 'texture', frame_id) + _load_frame(root, scene, 'passive', frame_id)


def _resolve_auxiliary_root(data_root: Path, value: Path | None) -> Path | None:
    """Resolve an optional dataset-relative auxiliary frame root."""
    if value is None:
        return None
    if value.is_absolute():
        return value
    if value.parts and value.parts[0] == data_root.name:
        return data_root.parent.joinpath(*value.parts)
    return data_root / value


def _load_source_off_cache(source_off_root: Path | None, scene: str, frame_id: int) -> np.ndarray | None:
    """Load one cached Stage 1 source-off estimate when available."""
    if source_off_root is not None:
        cached = _auxiliary_frame_path(source_off_root, scene, frame_id)
        if cached.is_file():
            image = np.load(cached).astype(np.float32)
            if image.ndim != 2:
                raise ValueError(f'Expected a 2-D source-off frame at {cached}, got {image.shape}')
            return image
    return None


def _load_passive_frame(root: Path, source_off_root: Path | None, scene: str, frame_id: int) -> np.ndarray:
    """Load ``S^off`` from a completed cache when present, otherwise from synthetic passive frames."""
    cached = _load_source_off_cache(source_off_root, scene, frame_id)
    if cached is not None:
        return cached
    return _load_frame(root, scene, 'passive', frame_id)


def _load_texture_anchor(
    root: Path,
    source_off_root: Path | None,
    scene: str,
    frame_id: int,
    require_source_off: bool = False,
) -> np.ndarray:
    """Load an active texture anchor, correcting it with a Stage 1 source-off estimate when present."""
    texture = _load_frame(root, scene, 'texture', frame_id)
    estimated_off = _load_source_off_cache(source_off_root, scene, frame_id)
    if estimated_off is None:
        if require_source_off:
            raise FileNotFoundError(
                f'Missing Stage 1 source-off cache for active frame {scene}/{frame_id:03d}; '
                'run stage1.py or pass the matching cache root.'
            )
        return texture
    source_on = _load_source_on_frame(root, scene, frame_id)
    return np.maximum(source_on - estimated_off, 0.0).astype(np.float32)


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
    """Return one interpolated texture target with centered passive context and flow labels."""

    def __init__(
        self,
        root: Path,
        split_file: Path,
        passive_context: int = 5,
        crop_size: int | None = None,
        random_crop: bool = False,
        pseudo_flow_root: Path | None = None,
        active_stride: int = 10,
        sample_passive_context: int | None = None,
        source_off_root: Path | None = None,
        require_source_off: bool = False,
    ) -> None:
        """Build samples from named scenes and active-frame intervals."""
        if active_stride <= 1:
            raise ValueError('active_stride must be larger than 1 so an interpolation target exists')
        sample_context = passive_context if sample_passive_context is None else sample_passive_context
        if sample_context < passive_context:
            raise ValueError('sample_passive_context must be greater than or equal to passive_context')
        self.root = root
        self.scenes = read_split(split_file)
        self.passive_context = passive_context
        self.sample_passive_context = sample_context
        self.active_stride = active_stride
        self.crop_size = crop_size
        self.random_crop = random_crop
        self.pseudo_flow_root = pseudo_flow_root
        self.source_off_root = _resolve_auxiliary_root(root, source_off_root)
        self.require_source_off = require_source_off
        if self.require_source_off and self.source_off_root is None:
            raise ValueError('Stage 1 source-off cache is required for the paper-aligned pipeline')
        self.items = [
            (scene, left, offset)
            for scene in self.scenes
            for left in range(1, 181 - active_stride, active_stride)
            for offset in range(1, active_stride)
            if _has_valid_passive_context(left + offset, self.sample_passive_context)
        ]
        validate_pseudo_flow_coverage(self.root, self.items, self.pseudo_flow_root, active_stride=self.active_stride)

    def __len__(self) -> int:
        """Return the number of valid interval-target samples."""
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        """Load texture endpoints, a target, centered ``C_t``, and target flow labels."""
        scene, left, offset = self.items[index]
        target_id = left + offset
        right = left + self.active_stride
        context_ids = passive_context_ids(target_id, self.passive_context)
        texture0 = _normalise(
            _load_texture_anchor(self.root, self.source_off_root, scene, left, self.require_source_off)
        )
        texture1 = _normalise(
            _load_texture_anchor(self.root, self.source_off_root, scene, right, self.require_source_off)
        )
        target = _normalise(_load_frame(self.root, scene, 'texture', target_id))
        passive = [_normalise(_load_passive_frame(self.root, self.source_off_root, scene, frame_id)) for frame_id in context_ids]
        flow = load_pseudo_flow(self.root, scene, left, target_id, right, self.pseudo_flow_root)
        texture0, texture1, target, *rest = _crop([texture0, texture1, target, *passive, flow], self.crop_size, self.random_crop)
        passive, flow = rest[:-1], rest[-1]
        tensor = lambda image: torch.from_numpy(image.copy()).unsqueeze(0)
        if passive:
            passive_tensor = torch.stack([torch.from_numpy(image.copy()) for image in passive])
        else:
            passive_tensor = torch.empty((0, target.shape[-2], target.shape[-1]), dtype=torch.float32)
        return {
            'scene': scene,
            'left_id': left,
            'right_id': right,
            'target_id': target_id,
            'texture0': tensor(texture0),
            'texture1': tensor(texture1),
            'target': tensor(target),
            'passive_context': passive_tensor,
            'flow': torch.from_numpy(flow.copy()),
            'time': torch.tensor([offset / float(self.active_stride)], dtype=torch.float32),
        }


def _has_valid_passive_context(target_id: int, passive_context: int) -> bool:
    """Return whether all requested passive context frames exist in the 1..180 sequence."""
    context_ids = passive_context_ids(target_id, passive_context)
    return not context_ids or (min(context_ids) >= 1 and max(context_ids) <= 180)
