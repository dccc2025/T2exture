"""T2exture-L wrapper built on the official AMT-L backbone."""

from __future__ import annotations

from pathlib import Path

from model.conditioning import PassiveGuidancePyramid, T2VAdapter
from model.t2texture_base import AMT_L_SPEC, T2textureAMTBase


class T2textureAMTL(T2textureAMTBase):
    """T2exture-L: AMT-L plus T2V adapter and passive guidance."""

    def __init__(self, pretrained: str | Path | None, passive_context: int = 5) -> None:
        super().__init__(AMT_L_SPEC, pretrained, passive_context)


TextureAdapter = T2VAdapter
PassivePyramidEncoder = PassiveGuidancePyramid

__all__ = ['PassivePyramidEncoder', 'T2textureAMTL', 'TextureAdapter']
