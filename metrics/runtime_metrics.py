"""Runtime and model-size metric helpers for deployment tables."""

from __future__ import annotations

import torch.nn as nn

DEPLOYMENT_METRIC_KEYS = ('Latency', 'FLOPs', 'Params', 'Trainable Params')


def count_parameters(model: nn.Module) -> int:
    """Return total parameter count."""
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Return trainable parameter count."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
