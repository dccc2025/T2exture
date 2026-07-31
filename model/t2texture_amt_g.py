"""T2exture-G wrapper built on the official AMT-G backbone."""

from __future__ import annotations

from pathlib import Path

from model.conditioning import PassiveGuidancePyramid, T2VAdapter
from model.t2texture_base import AMT_G_SPEC, T2textureAMTBase


class T2textureAMTG(T2textureAMTBase):
    """T2exture-G: large AMT-G plus T2V adapter and passive guidance."""

    def __init__(self, pretrained: str | Path | None, passive_context: int = 5) -> None:
        super().__init__(AMT_G_SPEC, pretrained, passive_context)


TextureAdapter = T2VAdapter
PassivePyramidEncoderG = PassiveGuidancePyramid

__all__ = ['PassivePyramidEncoderG', 'T2textureAMTG', 'TextureAdapter']
