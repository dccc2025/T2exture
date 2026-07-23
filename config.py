"""Configuration helpers shared by training, inference, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LossWeights:
    """Store the fixed supervision weights used by T2exture."""

    charbonnier: float = 1.0
    css: float = 0.1
    flow: float = 0.001

    def as_dict(self) -> dict[str, float]:
        """Return named weights for logging and loss construction."""
        return {'charbonnier': self.charbonnier, 'css': self.css, 'flow': self.flow}


@dataclass(frozen=True)
class TrainConfig:
    """Provide the stable defaults for the two-stage fine-tuning schedule."""

    passive_context: int = 4
    adapter_iterations: int = 10_000
    finetune_iterations: int = 5_000
    layerwise_lr_decay: float = 0.8


def passive_context_ids(center_id: int, context_size: int = 4) -> tuple[int, ...]:
    """Return equal numbers of passive frame IDs before and after ``center_id``."""
    if context_size <= 0 or context_size % 2:
        raise ValueError('passive_context must be a positive even integer')
    half = context_size // 2
    return tuple(range(center_id - half, center_id)) + tuple(range(center_id + 1, center_id + half + 1))
