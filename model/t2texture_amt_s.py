"""T2exture-S wrapper built on the official AMT-S backbone."""

from __future__ import annotations

from pathlib import Path

from model.conditioning import PassiveGuidancePyramid, T2VAdapter
from model.t2texture_base import AMT_S_SPEC, T2textureAMTBase


class T2textureAMTS(T2textureAMTBase):
    """T2exture-S: compact AMT-S plus T2V adapter and passive guidance."""

    def __init__(self, pretrained: str | Path | None, passive_context: int = 5) -> None:
        super().__init__(AMT_S_SPEC, pretrained, passive_context)


TextureAdapter = T2VAdapter
PassivePyramidEncoderS = PassiveGuidancePyramid

__all__ = ['PassivePyramidEncoderS', 'T2textureAMTS', 'TextureAdapter']
