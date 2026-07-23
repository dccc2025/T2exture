"""Image-quality metrics used by the synthetic-only experiment tables."""

from __future__ import annotations

import torch
import torch.nn.functional as F

MAIN_METRIC_KEYS = ('PSNR', 'SSIM', 'Edge-FI@2px', 'IE', 'NIE')


def _check_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    """Reject metric inputs that cannot be compared sample-by-sample."""
    if prediction.shape != target.shape:
        raise ValueError(f'Metric inputs must have matching shapes, got {prediction.shape} and {target.shape}')
    if prediction.ndim < 3:
        raise ValueError(f'Metric inputs must include spatial dimensions, got {prediction.shape}')


def _spatial_dims(tensor: torch.Tensor) -> tuple[int, ...]:
    """Return all non-batch dimensions for image-shaped tensors."""
    return tuple(range(1, tensor.ndim))


def _as_nchw(tensor: torch.Tensor) -> torch.Tensor:
    """Convert ``[B,H,W]`` or ``[B,C,H,W]`` tensors to grayscale ``[B,1,H,W]``."""
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    if tensor.ndim == 4:
        return tensor.mean(dim=1, keepdim=True)
    raise ValueError(f'Expected [B,H,W] or [B,C,H,W], got {tensor.shape}')


def batch_psnr(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return per-sample PSNR for tensors normalized to ``[0, 1]``."""
    _check_pair(prediction, target)
    dims = _spatial_dims(prediction)
    mse = (prediction - target).pow(2).mean(dim=dims).clamp_min(eps)
    return -10.0 * torch.log10(mse)


def batch_ie(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return AMT-style interpolation error in 8-bit intensity units."""
    _check_pair(prediction, target)
    dims = _spatial_dims(prediction)
    pred_8bit = torch.round(prediction.clamp(0.0, 1.0) * 255.0)
    target_8bit = torch.round(target.clamp(0.0, 1.0) * 255.0)
    return (pred_8bit - target_8bit).abs().mean(dim=dims)


def batch_nie(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return normalized interpolation error in ``[0, 1]`` intensity units."""
    return batch_ie(prediction, target) / 255.0


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a grouped 2-D Gaussian window for SSIM."""
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gaussian = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gaussian = gaussian / gaussian.sum()
    window_2d = torch.outer(gaussian, gaussian)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def batch_ssim(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11, eps: float = 1e-12) -> torch.Tensor:
    """Return per-sample SSIM for tensors normalized to ``[0, 1]``."""
    _check_pair(prediction, target)
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(1)
        target = target.unsqueeze(1)
    if prediction.ndim != 4:
        raise ValueError(f'SSIM expects [B,H,W] or [B,C,H,W], got {prediction.shape}')

    batch, channels, height, width = prediction.shape
    real_size = min(window_size, height, width)
    if real_size % 2 == 0:
        real_size -= 1
    if real_size < 3:
        raise ValueError(f'SSIM needs at least 3x3 spatial inputs, got {(height, width)}')
    padding = real_size // 2
    window = _gaussian_window(real_size, 1.5, channels, prediction.device, prediction.dtype)

    pred_pad = F.pad(prediction, (padding, padding, padding, padding), mode='replicate')
    target_pad = F.pad(target, (padding, padding, padding, padding), mode='replicate')
    mu_pred = F.conv2d(pred_pad, window, groups=channels)
    mu_target = F.conv2d(target_pad, window, groups=channels)
    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_cross = mu_pred * mu_target

    sigma_pred = F.conv2d(F.pad(prediction * prediction, (padding, padding, padding, padding), mode='replicate'), window, groups=channels) - mu_pred_sq
    sigma_target = F.conv2d(F.pad(target * target, (padding, padding, padding, padding), mode='replicate'), window, groups=channels) - mu_target_sq
    sigma_cross = F.conv2d(F.pad(prediction * target, (padding, padding, padding, padding), mode='replicate'), window, groups=channels) - mu_cross

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred + sigma_target + c2)
    ssim_map = numerator / denominator.clamp_min(eps)
    return ssim_map.reshape(batch, -1).mean(dim=1)


def _sobel_magnitude(tensor: torch.Tensor) -> torch.Tensor:
    """Return Sobel gradient magnitude for grayscale ``[B,1,H,W]`` images."""
    kernel_x = tensor.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    kernel_y = tensor.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    grad_x = F.conv2d(F.pad(tensor, (1, 1, 1, 1), mode='replicate'), kernel_x)
    grad_y = F.conv2d(F.pad(tensor, (1, 1, 1, 1), mode='replicate'), kernel_y)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-12)


def batch_edge_fi(
    prediction: torch.Tensor,
    target: torch.Tensor,
    tolerance_px: int = 2,
    edge_quantile: float = 0.90,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return Edge-FI@tolerance as a Sobel-edge F1 score with pixel tolerance."""
    _check_pair(prediction, target)
    if tolerance_px < 0:
        raise ValueError(f'tolerance_px must be non-negative, got {tolerance_px}')
    pred_gray = _as_nchw(prediction.clamp(0.0, 1.0))
    target_gray = _as_nchw(target.clamp(0.0, 1.0))
    pred_mag = _sobel_magnitude(pred_gray)
    target_mag = _sobel_magnitude(target_gray)

    thresholds = target_mag.flatten(1).quantile(edge_quantile, dim=1).view(-1, 1, 1, 1).clamp_min(eps)
    pred_edges = pred_mag >= thresholds
    target_edges = target_mag >= thresholds

    kernel_size = 2 * tolerance_px + 1
    pred_near = F.max_pool2d(pred_edges.float(), kernel_size, stride=1, padding=tolerance_px) > 0
    target_near = F.max_pool2d(target_edges.float(), kernel_size, stride=1, padding=tolerance_px) > 0

    pred_count = pred_edges.flatten(1).sum(dim=1)
    target_count = target_edges.flatten(1).sum(dim=1)
    true_pred = (pred_edges & target_near).flatten(1).sum(dim=1)
    true_target = (target_edges & pred_near).flatten(1).sum(dim=1)

    precision = true_pred / pred_count.clamp_min(eps)
    recall = true_target / target_count.clamp_min(eps)
    score = (2.0 * precision * recall) / (precision + recall).clamp_min(eps)
    no_edges = (pred_count == 0) & (target_count == 0)
    return torch.where(no_edges, torch.ones_like(score), score)


def compute_main_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return the main-table metrics with keys matching the experiment plan."""
    return {
        'PSNR': batch_psnr(prediction, target),
        'SSIM': batch_ssim(prediction, target),
        'Edge-FI@2px': batch_edge_fi(prediction, target, tolerance_px=2),
        'IE': batch_ie(prediction, target),
        'NIE': batch_nie(prediction, target),
    }
