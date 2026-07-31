"""T2exture conditioning blocks that mirror the paper's main figure."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Encode one interpolation timestamp as fixed Fourier features."""

    def __init__(self, dim: int = 32, max_frequency: float = 10_000.0) -> None:
        super().__init__()
        if dim <= 0 or dim % 2:
            raise ValueError('Fourier feature dimension must be positive and even')
        frequencies = torch.exp(torch.linspace(0.0, math.log(max_frequency), dim // 2))
        self.register_buffer('frequencies', frequencies, persistent=False)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Return features shaped ``[B, dim, 1, 1]`` for Conv2d modulation."""
        time = time.reshape(time.shape[0], 1)
        angles = time * self.frequencies.unsqueeze(0) * (2.0 * math.pi)
        features = torch.cat((angles.sin(), angles.cos()), dim=1)
        return features.unsqueeze(-1).unsqueeze(-1)


class FourierConv2d(nn.Module):
    """Relative positional encoding block: Fourier -> Conv2d -> SiLU -> Conv2d."""

    def __init__(self, out_channels: int, fourier_dim: int = 32, hidden_channels: int = 64) -> None:
        super().__init__()
        self.fourier = FourierFeatures(fourier_dim)
        self.net = nn.Sequential(
            nn.Conv2d(fourier_dim, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Return a broadcastable time-conditioned feature map."""
        return self.net(self.fourier(time))


class DuplicateChannels(nn.Module):
    """Duplicate one texture channel into the RGB channel count expected by AMT."""

    def __init__(self, out_channels: int = 3) -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError('out_channels must be positive')
        self.out_channels = out_channels

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] != 1:
            raise ValueError(f'DuplicateChannels expects one input channel, got {image.shape[1]}')
        return image.repeat(1, self.out_channels, 1, 1)


class T2VAdapter(nn.Module):
    """Texture-to-visual adapter: duplicate one texture channel, then refine it."""

    def __init__(self, channels: int = 3, hidden_channels: int = 16) -> None:
        super().__init__()
        self.duplicate = DuplicateChannels(channels)
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Adapt one texture frame while preserving AMT's pretrained input scale at start."""
        duplicated = self.duplicate(image)
        return duplicated + self.adapter(duplicated)


class ConvP(nn.Module):
    """Passive guidance block used three times in the main figure."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.PReLU(out_channels),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.block(image)


class PassiveGuidancePyramid(nn.Module):
    """Encode passive frames into the three AMT decoder guidance fields."""

    def __init__(
        self,
        context_size: int,
        decoder_channels: tuple[int, int, int],
        temporal_modulation: bool = True,
    ) -> None:
        super().__init__()
        if context_size <= 0:
            raise ValueError('context_size must be positive when passive guidance is enabled')
        self.context_size = context_size
        self.temporal_modulation = temporal_modulation
        c1, c2, c3 = decoder_channels
        self.blocks = nn.ModuleList(
            [
                ConvP(context_size, c1),
                ConvP(c1, c2),
                ConvP(c2, c3),
            ]
        )
        self.relative_position = FourierConv2d(c1) if temporal_modulation else None

    def forward(self, passive_context: torch.Tensor, time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``G_t^1``, ``G_t^2``, and ``G_t^3`` passive guidance fields."""
        if passive_context.shape[1] != self.context_size:
            raise ValueError(f'Expected {self.context_size} passive frames, got {passive_context.shape[1]}')
        first = self.blocks[0](passive_context)
        if self.relative_position is not None:
            first = first + self.relative_position(time)
        second = self.blocks[1](first)
        third = self.blocks[2](second)
        return first, second, third


__all__ = [
    'ConvP',
    'DuplicateChannels',
    'FourierConv2d',
    'FourierFeatures',
    'PassiveGuidancePyramid',
    'T2VAdapter',
]
