"""Time-conditioning layers used by texture and passive-frame encoders."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierTimeEmbedding(nn.Module):
    """Map a scalar interpolation timestamp to a fixed Fourier feature vector."""

    def __init__(self, dim: int = 32) -> None:
        """Create paired sine and cosine frequencies with total width ``dim``."""
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError('Fourier time embedding dimension must be positive and even')
        frequencies = torch.exp(torch.linspace(0.0, math.log(10_000.0), dim // 2))
        self.register_buffer('frequencies', frequencies, persistent=False)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Encode a ``[batch, 1]`` or broadcastable timestamp tensor into ``dim`` features."""
        time = time.reshape(time.shape[0], 1)
        angles = time * self.frequencies.unsqueeze(0) * (2.0 * math.pi)
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class SharedTimeMLP(nn.Module):
    """Project Fourier features into the shared conditioning space used by all adapters."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64) -> None:
        """Build a compact two-layer MLP for time-dependent feature modulation."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Return a shared time-conditioning vector for every sample."""
        return self.net(embedding)
