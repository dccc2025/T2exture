"""Image-quality metrics used by the synthetic-only experiment tables."""

from __future__ import annotations

import importlib

import numpy as np
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


def _to_uint8_images(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized grayscale ``[B,1,H,W]`` tensors to uint8 images."""
    array = tensor.detach().float().clamp(0.0, 1.0).cpu().numpy()
    return np.round(array[:, 0] * 255.0).astype(np.uint8)


def _cv2_module():
    """Load OpenCV lazily so importing metric helpers stays lightweight."""
    try:
        return importlib.import_module('cv2')
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'Edge-FI@2px now uses Canny edge maps and requires OpenCV. '
            'Install opencv-python-headless or create the environment from environment.yaml.'
        ) from exc


def canny_edge_map(image: np.ndarray, low_threshold: int = 100, high_threshold: int = 200, aperture_size: int = 3, l2_gradient: bool = True) -> np.ndarray:
    """Return a binary Canny edge map for one uint8 grayscale image."""
    cv2 = _cv2_module()
    edges = cv2.Canny(
        np.ascontiguousarray(image),
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=aperture_size,
        L2gradient=l2_gradient,
    )
    return edges > 0


def edge_f1_from_maps(pred_edges: np.ndarray, target_edges: np.ndarray, tolerance_px: int = 2, eps: float = 1e-12) -> float:
    """Return boundary F1 after allowing matches inside a square pixel tolerance."""
    if pred_edges.shape != target_edges.shape:
        raise ValueError(f'Edge maps must have matching shapes, got {pred_edges.shape} and {target_edges.shape}')
    pred_count = int(pred_edges.sum())
    target_count = int(target_edges.sum())
    if pred_count == 0 and target_count == 0:
        return 1.0
    if tolerance_px > 0:
        cv2 = _cv2_module()
        kernel = np.ones((2 * tolerance_px + 1, 2 * tolerance_px + 1), dtype=np.uint8)
        pred_near = cv2.dilate(pred_edges.astype(np.uint8), kernel, iterations=1) > 0
        target_near = cv2.dilate(target_edges.astype(np.uint8), kernel, iterations=1) > 0
    else:
        pred_near = pred_edges
        target_near = target_edges
    true_pred = int(np.logical_and(pred_edges, target_near).sum())
    true_target = int(np.logical_and(target_edges, pred_near).sum())
    precision = true_pred / max(float(pred_count), eps)
    recall = true_target / max(float(target_count), eps)
    return float((2.0 * precision * recall) / max(precision + recall, eps))


def batch_edge_fi(
    prediction: torch.Tensor,
    target: torch.Tensor,
    tolerance_px: int = 2,
    low_threshold: int = 100,
    high_threshold: int = 200,
    aperture_size: int = 3,
    l2_gradient: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return Edge-FI@tolerance as a Canny boundary F1 score."""
    _check_pair(prediction, target)
    if tolerance_px < 0:
        raise ValueError(f'tolerance_px must be non-negative, got {tolerance_px}')
    if low_threshold < 0 or high_threshold < 0:
        raise ValueError('Canny thresholds must be non-negative.')
    if low_threshold > high_threshold:
        raise ValueError(f'low_threshold must be <= high_threshold, got {low_threshold} > {high_threshold}')
    if aperture_size not in (3, 5, 7):
        raise ValueError(f'aperture_size must be one of 3, 5, or 7, got {aperture_size}')
    pred_gray = _as_nchw(prediction.clamp(0.0, 1.0))
    target_gray = _as_nchw(target.clamp(0.0, 1.0))
    pred_images = _to_uint8_images(pred_gray)
    target_images = _to_uint8_images(target_gray)
    scores = [
        edge_f1_from_maps(
            canny_edge_map(pred_image, low_threshold, high_threshold, aperture_size, l2_gradient),
            canny_edge_map(target_image, low_threshold, high_threshold, aperture_size, l2_gradient),
            tolerance_px,
            eps,
        )
        for pred_image, target_image in zip(pred_images, target_images)
    ]
    return prediction.new_tensor(scores)


def compute_main_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return stable image-quality metrics for reporting and comparison."""
    return {
        'PSNR': batch_psnr(prediction, target),
        'SSIM': batch_ssim(prediction, target),
        'Edge-FI@2px': batch_edge_fi(prediction, target, tolerance_px=2),
        'IE': batch_ie(prediction, target),
        'NIE': batch_nie(prediction, target),
    }
