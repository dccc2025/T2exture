"""Configuration helpers shared by training, inference, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class LossWeights:
    """Store the fixed supervision weights used by T2exture."""

    charbonnier: float = 1.0
    css: float = 0.1
    flow: float = 0.001

    def as_dict(self) -> dict[str, float]:
        """Return named weights for logging and loss construction."""
        return {'charbonnier': self.charbonnier, 'css': self.css, 'flow': self.flow}

    @classmethod
    def from_mapping(cls, values: Mapping[str, float] | None) -> 'LossWeights':
        """Build weights from a config mapping while preserving defaults."""
        if values is None:
            return cls()
        defaults = cls().as_dict()
        unknown = set(values) - set(defaults)
        if unknown:
            raise ValueError(f'Unknown loss weight keys: {sorted(unknown)}')
        defaults.update({key: float(value) for key, value in values.items()})
        return cls(**defaults)


@dataclass(frozen=True)
class TrainConfig:
    """Provide the stable defaults for the two-stage fine-tuning schedule."""

    passive_context: int = 5
    active_stride: int = 10
    sample_passive_context: int | None = None
    require_datasets_root: bool = True
    adapter_iterations: int = 10_000
    finetune_iterations: int = 5_000
    layerwise_lr_decay: float = 0.8


def passive_context_ids(center_id: int, context_size: int = 5) -> tuple[int, ...]:
    """Return the centered passive structural context ``C_t`` around ``center_id``."""
    if context_size < 0 or (context_size > 0 and context_size % 2 == 0):
        raise ValueError('passive_context must be zero or a positive odd integer')
    if context_size == 0:
        return ()
    radius = context_size // 2
    return tuple(range(center_id - radius, center_id + radius + 1))


def expected_flow_set(active_stride: int) -> str:
    """Return the canonical pseudo-flow set name for one active-anchor stride."""
    if active_stride <= 1:
        raise ValueError('active_stride must be larger than 1 so an interpolation target exists')
    return f's{active_stride:02d}'


def resolve_sample_passive_context(config: Mapping[str, Any], passive_context: int) -> int:
    """Return the context size used only for sample-window filtering."""
    value = config.get('sample_passive_context')
    sample_context = passive_context if value is None else int(value)
    if sample_context < passive_context:
        raise ValueError('sample_passive_context must be greater than or equal to passive_context')
    passive_context_ids(center_id=10, context_size=sample_context)
    return sample_context


def validate_runtime_config(config: Mapping[str, Any], data_root: Path) -> None:
    """Fail early when a run points at the wrong prepared dataset root."""
    require_datasets_root = config.get('require_datasets_root', config.get('require_dataset_roi', True))
    if bool(require_datasets_root) and data_root.resolve().name != 'datasets':
        raise ValueError('Use --data-root datasets for the prepared T2exture data. Set require_datasets_root: false only for debugging.')
    active_stride = int(config.get('active_stride', 10))
    flow_dir = config.get('pseudo_flow_dir')
    if not flow_dir:
        return
    flow_set = Path(str(flow_dir)).name
    if re.fullmatch(r's\d+', flow_set) and flow_set != expected_flow_set(active_stride):
        raise ValueError(f'pseudo_flow_dir={flow_dir!r} does not match active_stride={active_stride}; expected flow/{expected_flow_set(active_stride)}')
