"""Train the Ours-L passive-guidance ablation without passive temporal modulation.

This entry point reuses ``train.py`` and the AMT-L wrapper. It only patches the
AMT-L passive encoder inside this process so the passive feature pyramid is
used without adding the learned target-time biases.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from model.t2texture_amt import PassivePyramidEncoder
from train import main as train_main


def forward_without_passive_temporal_modulation(
    self: PassivePyramidEncoder,
    passive_context: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode passive guidance features while ignoring target-time modulation."""
    _ = time
    if passive_context.shape[1] != self.context_size:
        raise ValueError(f'Expected {self.context_size} passive frames, got {passive_context.shape[1]}')
    first = self.blocks[0](passive_context)
    second = self.blocks[1](first)
    third = self.blocks[2](second)
    return first, second, third


def main() -> None:
    PassivePyramidEncoder.forward = forward_without_passive_temporal_modulation
    train_main()


if __name__ == '__main__':
    main()
