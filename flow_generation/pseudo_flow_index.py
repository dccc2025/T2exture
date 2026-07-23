"""Index and load pseudo-flow labels for AMT flow distillation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_FLOW_SET = 's10'
DEFAULT_PSEUDO_FLOW_DIR = Path('flow') / DEFAULT_FLOW_SET
LEGACY_PSEUDO_FLOW_SUBDIR = Path('pseudo_flow') / 'liteflownet_default_train_valid'


def resolve_pseudo_flow_root(root: Path, pseudo_flow_root: Path | None = None) -> Path:
    """Return the dataset-level pseudo-flow root."""
    if pseudo_flow_root is None:
        return root / DEFAULT_PSEUDO_FLOW_DIR
    return pseudo_flow_root if pseudo_flow_root.is_absolute() else root / pseudo_flow_root


def scene_flow_dir(root: Path, scene: str, pseudo_flow_root: Path | None = None) -> Path:
    """Return the AMT-style per-scene pseudo-flow cache directory."""
    return resolve_pseudo_flow_root(root, pseudo_flow_root) / scene


def legacy_scene_flow_dir(root: Path, scene: str) -> Path:
    """Return the old per-scene flow directory kept only for compatibility."""
    return root / 'sim' / scene / 'flow'


def flow_file_name(left_id: int, target_id: int, right_id: int) -> str:
    """Return the target-to-endpoints flow file name."""
    return f'{left_id:03d}_{target_id:03d}_{right_id:03d}.npz'


def pseudo_flow_path(root: Path, scene: str, left_id: int, target_id: int, right_id: int, pseudo_flow_root: Path | None = None) -> Path:
    """Return the flow-label path for one interpolation target.

    The canonical layout mirrors AMT: raw frames stay under ``sim/<scene>``,
    while derived pseudo-flow labels live under ``flow/<flow_set>/<scene>``.
    """
    return scene_flow_dir(root, scene, pseudo_flow_root) / flow_file_name(left_id, target_id, right_id)


def legacy_pseudo_flow_path(root: Path, scene: str, left_id: int, target_id: int, right_id: int) -> Path:
    """Return the old ``sim/<scene>/flow`` cache path."""
    return legacy_scene_flow_dir(root, scene) / flow_file_name(left_id, target_id, right_id)


def pseudo_flow_candidates(root: Path, scene: str, left_id: int, target_id: int, right_id: int, pseudo_flow_root: Path | None = None) -> list[Path]:
    """Return load candidates, with old scene-local flow only for the default set."""
    primary = pseudo_flow_path(root, scene, left_id, target_id, right_id, pseudo_flow_root)
    candidates = [primary]
    if resolve_pseudo_flow_root(root, pseudo_flow_root) == root / DEFAULT_PSEUDO_FLOW_DIR:
        legacy = legacy_pseudo_flow_path(root, scene, left_id, target_id, right_id)
        if legacy != primary:
            candidates.append(legacy)
    return candidates


def load_pseudo_flow(root: Path, scene: str, left_id: int, target_id: int, right_id: int, pseudo_flow_root: Path | None = None) -> np.ndarray:
    """Load target-to-endpoint pseudo flows and concatenate them as ``[4, H, W]``."""
    candidates = pseudo_flow_candidates(root, scene, left_id, target_id, right_id, pseudo_flow_root)
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


def missing_pseudo_flow_paths(
    root: Path,
    samples: list[tuple[str, int, int]],
    pseudo_flow_root: Path | None = None,
    active_stride: int = 10,
) -> list[Path]:
    """Return missing flow-label paths for ``(scene, left_id, target_offset)`` samples."""
    missing = []
    for scene, left_id, offset in samples:
        target_id = left_id + offset
        right_id = left_id + active_stride
        candidates = pseudo_flow_candidates(root, scene, left_id, target_id, right_id, pseudo_flow_root)
        if not any(path.is_file() for path in candidates):
            missing.append(candidates[0])
    return missing


def validate_pseudo_flow_coverage(
    root: Path,
    samples: list[tuple[str, int, int]],
    pseudo_flow_root: Path | None = None,
    active_stride: int = 10,
    preview: int = 8,
) -> None:
    """Fail early when a split is not covered by the LiteFlowNet flow labels."""
    missing = missing_pseudo_flow_paths(root, samples, pseudo_flow_root, active_stride=active_stride)
    if not missing:
        return
    shown = '\n'.join(str(path) for path in missing[:preview])
    suffix = '' if len(missing) <= preview else f'\n... and {len(missing) - preview} more missing pseudo-flow files'
    raise FileNotFoundError(f'Missing {len(missing)} LiteFlowNet flow labels:\n{shown}{suffix}')
