from __future__ import annotations

import torch
import torch.nn.functional as F


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target).square() + eps * eps).mean()


def _gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = image[..., :, 1:] - image[..., :, :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    return dx, dy


def contrast_structure(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu_p, mu_t = F.avg_pool2d(pred, 7, 1, 3), F.avg_pool2d(target, 7, 1, 3)
    var_p = F.avg_pool2d(pred * pred, 7, 1, 3) - mu_p * mu_p
    var_t = F.avg_pool2d(target * target, 7, 1, 3) - mu_t * mu_t
    cov = F.avg_pool2d(pred * target, 7, 1, 3) - mu_p * mu_t
    cs = (2. * cov + 9e-4) / (var_p + var_t + 9e-4)
    return (1. - cs.clamp(-1., 1.)).mean()


def edge_loss(pred: torch.Tensor, target: torch.Tensor, passive: torch.Tensor) -> torch.Tensor:
    px, py = _gradients(pred)
    tx, ty = _gradients(target)
    ex, ey = _gradients(passive)
    weight_x = 1. + 2. * torch.sigmoid((ex.abs() - ex.abs().flatten(1).quantile(.9, dim=1).view(-1, 1, 1, 1)) / .03)
    weight_y = 1. + 2. * torch.sigmoid((ey.abs() - ey.abs().flatten(1).quantile(.9, dim=1).view(-1, 1, 1, 1)) / .03)
    return (weight_x * (px - tx).abs()).mean() + (weight_y * (py - ty).abs()).mean()


def temporal_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs().mean()
