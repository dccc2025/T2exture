"""Dataset loading for AMT fine-tuning with optional passive neighbours."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import passive_context_ids

DEFAULT_FLOW_DIR = Path('flow') / 's10'


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


def _flow_file_name(left_id: int, target_id: int, right_id: int) -> str:
    """Return the canonical target-to-endpoints flow file name."""
    return f'{left_id:03d}_{target_id:03d}_{right_id:03d}.npz'


def _resolve_pseudo_flow_root(root: Path, pseudo_flow_root: Path | None) -> Path:
    """Resolve a pseudo-flow root relative to the dataset root."""
    if pseudo_flow_root is None:
        return root / DEFAULT_FLOW_DIR
    return pseudo_flow_root if pseudo_flow_root.is_absolute() else root / pseudo_flow_root


def _pseudo_flow_candidates(root: Path, scene: str, left_id: int, target_id: int, right_id: int, pseudo_flow_root: Path | None) -> list[Path]:
    """Return canonical and legacy flow-label candidates for one target."""
    flow_root = _resolve_pseudo_flow_root(root, pseudo_flow_root)
    name = _flow_file_name(left_id, target_id, right_id)
    candidates = [flow_root / scene / name]
    if flow_root == root / DEFAULT_FLOW_DIR:
        legacy = root / 'sim' / scene / 'flow' / name
        if legacy != candidates[0]:
            candidates.append(legacy)
    return candidates


def _load_pseudo_flow(root: Path, scene: str, left_id: int, target_id: int, right_id: int, pseudo_flow_root: Path | None) -> np.ndarray:
    """Load target-to-endpoint pseudo flows and concatenate them as [4, H, W]."""
    candidates = _pseudo_flow_candidates(root, scene, left_id, target_id, right_id, pseudo_flow_root)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        checked = '\n'.join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f'Missing pseudo-flow label. Checked:\n{checked}')

    with np.load(path) as data:
        missing = {'flow0', 'flow1'} - set(data.files)
        if missing:
            raise ValueError(f'Flow file {path} is missing keys: {sorted(missing)}')
        flow0 = data['flow0'].astype(np.float32)
        flow1 = data['flow1'].astype(np.float32)
    if flow0.ndim != 3 or flow1.ndim != 3 or flow0.shape[0] != 2 or flow1.shape[0] != 2:
        raise ValueError(f'Expected flow0/flow1 shapes [2, H, W] at {path}, got {flow0.shape} and {flow1.shape}')
    if flow0.shape[1:] != flow1.shape[1:]:
        raise ValueError(f'Flow spatial shapes do not match at {path}: {flow0.shape} vs {flow1.shape}')
    return np.concatenate((flow0, flow1), axis=0)


def _validate_pseudo_flow_coverage(root: Path, samples: list[tuple[str, int, int]], pseudo_flow_root: Path | None, active_stride: int, preview: int = 8) -> None:
    """Fail before training when the split is not covered by pseudo-flow labels."""
    missing = []
    for scene, left_id, offset in samples:
        target_id = left_id + offset
        right_id = left_id + active_stride
        candidates = _pseudo_flow_candidates(root, scene, left_id, target_id, right_id, pseudo_flow_root)
        if not any(path.is_file() for path in candidates):
            missing.append(candidates[0])
    if not missing:
        return
    shown = '\n'.join(str(path) for path in missing[:preview])
    suffix = '' if len(missing) <= preview else f'\n... and {len(missing) - preview} more missing pseudo-flow files'
    raise FileNotFoundError(f'Missing {len(missing)} pseudo-flow labels:\n{shown}{suffix}')


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
    """Return one interpolated texture target with optional passive context and flow labels."""

    def __init__(
        self,
        root: Path,
        split_file: Path,
        passive_context: int = 4,
        crop_size: int | None = None,
        random_crop: bool = False,
        pseudo_flow_root: Path | None = None,
        active_stride: int = 10,
        sample_passive_context: int | None = None,
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
        self.items = [
            (scene, left, offset)
            for scene in self.scenes
            for left in range(1, 181 - active_stride, active_stride)
            for offset in range(1, active_stride)
            if _has_valid_passive_context(left + offset, self.sample_passive_context)
        ]
        _validate_pseudo_flow_coverage(self.root, self.items, self.pseudo_flow_root, active_stride=self.active_stride)

    def __len__(self) -> int:
        """Return the number of valid interval-target samples."""
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        """Load texture endpoints, a target, four passive neighbours, and target flow labels."""
        scene, left, offset = self.items[index]
        target_id = left + offset
        right = left + self.active_stride
        context_ids = passive_context_ids(target_id, self.passive_context)
        texture0 = _normalise(_load_frame(self.root, scene, 'texture', left))
        texture1 = _normalise(_load_frame(self.root, scene, 'texture', right))
        target = _normalise(_load_frame(self.root, scene, 'texture', target_id))
        passive = [_normalise(_load_frame(self.root, scene, 'passive', frame_id)) for frame_id in context_ids]
        flow = _load_pseudo_flow(self.root, scene, left, target_id, right, self.pseudo_flow_root)
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
