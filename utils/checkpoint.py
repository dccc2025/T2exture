"""Checkpoint helpers shared by training, evaluation, and inference."""

from __future__ import annotations

import pathlib
from pathlib import Path
from typing import Any

import torch


def torch_load_portable(path: str | Path, device: torch.device | str) -> Any:
    """Load a PyTorch checkpoint saved on either Windows or POSIX hosts."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except NotImplementedError as exc:
        message = str(exc)
        if 'WindowsPath' in message:
            original_windows_path = pathlib.WindowsPath
            pathlib.WindowsPath = pathlib.PosixPath
            try:
                return torch.load(path, map_location=device, weights_only=False)
            finally:
                pathlib.WindowsPath = original_windows_path
        if 'PosixPath' in message:
            original_posix_path = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            try:
                return torch.load(path, map_location=device, weights_only=False)
            finally:
                pathlib.PosixPath = original_posix_path
        raise


def extract_model_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract a model state dict from a training checkpoint or raw state dict."""
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f'Unsupported checkpoint format: {type(checkpoint)!r}')
