"""T2exture modules layered on top of vendored AMT backbones."""

from pathlib import Path

import torch.nn as nn

from .t2texture_amt_g import T2textureAMTG
from .t2texture_amt_s import T2textureAMTS
from .t2texture_amt import T2textureAMTL

T2TEXTURE_MODELS = {
    'amt-s': T2textureAMTS,
    'amt-l': T2textureAMTL,
    'amt-g': T2textureAMTG,
}


def build_t2texture_model(backbone: str, pretrained: str | Path, passive_context: int = 4) -> nn.Module:
    """Build a T2exture wrapper for one AMT backbone name."""
    key = backbone.lower()
    if key not in T2TEXTURE_MODELS:
        raise ValueError(f'Unknown T2exture backbone {backbone!r}; expected one of {sorted(T2TEXTURE_MODELS)}')
    return T2TEXTURE_MODELS[key](pretrained, passive_context)


__all__ = ['T2TEXTURE_MODELS', 'T2textureAMTG', 'T2textureAMTL', 'T2textureAMTS', 'build_t2texture_model']
