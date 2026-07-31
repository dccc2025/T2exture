"""Flow warping primitive required by the vendored AMT-L implementation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample ``image`` at pixel displacements stored in two-channel ``flow``."""
    batch, _, height, width = flow.shape
    x = torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype).view(1, 1, 1, width).expand(batch, -1, height, -1)
    y = torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype).view(1, 1, height, 1).expand(batch, -1, -1, width)
    base = torch.cat((x, y), dim=1)
    normalized = torch.cat((flow[:, :1] / ((width - 1.0) / 2.0), flow[:, 1:2] / ((height - 1.0) / 2.0)), dim=1)
    return F.grid_sample(image, (base + normalized).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=True)
