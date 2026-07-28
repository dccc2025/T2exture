"""No-reference and source-related metrics for real-sequence evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

REAL_METRIC_KEYS = ('En', 'AG', 'SF', 'SD', 'SCD', 'PI')


def _to_numpy(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert one grayscale image-like value to a finite float array in [0, 1]."""
    if torch.is_tensor(image):
        array = image.detach().float().cpu().numpy()
    else:
        array = np.asarray(image, dtype=np.float32)
    array = np.squeeze(array).astype(np.float32)
    if array.ndim != 2:
        raise ValueError(f'Expected one grayscale image, got shape {array.shape}')
    return np.nan_to_num(np.clip(array, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)


def entropy(image: torch.Tensor | np.ndarray) -> float:
    """Return 8-bit Shannon entropy."""
    array = np.round(_to_numpy(image) * 255.0).astype(np.uint8)
    hist = np.bincount(array.reshape(-1), minlength=256).astype(np.float64)
    prob = hist / max(1.0, hist.sum())
    prob = prob[prob > 0.0]
    return float(-(prob * np.log2(prob)).sum())


def average_gradient(image: torch.Tensor | np.ndarray) -> float:
    """Return the average gradient magnitude."""
    array = _to_numpy(image)
    if array.shape[0] < 2 or array.shape[1] < 2:
        return 0.0
    grad_y, grad_x = np.gradient(array)
    return float(np.sqrt((grad_x * grad_x + grad_y * grad_y) * 0.5).mean())


def spatial_frequency(image: torch.Tensor | np.ndarray) -> float:
    """Return spatial frequency from row and column differences."""
    array = _to_numpy(image)
    if array.shape[0] < 2 or array.shape[1] < 2:
        return 0.0
    row_frequency = float(np.sqrt(np.mean((array[1:, :] - array[:-1, :]) ** 2)))
    col_frequency = float(np.sqrt(np.mean((array[:, 1:] - array[:, :-1]) ** 2)))
    return float(np.sqrt(row_frequency * row_frequency + col_frequency * col_frequency))


def standard_deviation(image: torch.Tensor | np.ndarray) -> float:
    """Return image standard deviation in normalized intensity units."""
    return float(np.std(_to_numpy(image)))


def _correlation(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Return Pearson correlation, using zero when either input is constant."""
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denominator <= eps:
        return 0.0
    return float(np.sum(a * b) / denominator)


def sum_correlation_difference(
    prediction: torch.Tensor | np.ndarray,
    source0: torch.Tensor | np.ndarray,
    source1: torch.Tensor | np.ndarray,
) -> float:
    """Return SCD using the two endpoint/source frames and one fused prediction."""
    pred = _to_numpy(prediction)
    left = _to_numpy(source0)
    right = _to_numpy(source1)
    if pred.shape != left.shape or pred.shape != right.shape:
        raise ValueError(f'SCD inputs must share one shape, got {pred.shape}, {left.shape}, {right.shape}')
    return _correlation(pred - right, left) + _correlation(pred - left, right)


def compute_real_metrics(
    prediction: torch.Tensor | np.ndarray,
    source0: torch.Tensor | np.ndarray,
    source1: torch.Tensor | np.ndarray,
    pi_value: float | None = None,
) -> dict[str, float | None]:
    """Return the real-benchmark metric set for one predicted frame."""
    return {
        'En': entropy(prediction),
        'AG': average_gradient(prediction),
        'SF': spatial_frequency(prediction),
        'SD': standard_deviation(prediction),
        'SCD': sum_correlation_difference(prediction, source0, source1),
        'PI': pi_value,
    }


def mean_real_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Average real metrics, preserving ``None`` when every value is missing."""
    if not rows:
        raise ValueError('Cannot average an empty real-metric group')
    summary: dict[str, float | None] = {}
    for key in REAL_METRIC_KEYS:
        values = [float(row[key]) for row in rows if row.get(key) not in (None, '')]
        summary[key] = float(np.mean(values)) if values else None
    return summary
