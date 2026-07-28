"""Shared AMT wrapper used by T2exture-S, T2exture-L, and T2exture-G."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from model.conditioning import PassiveGuidancePyramid, T2VAdapter


@dataclass(frozen=True)
class AMTBackboneSpec:
    """Describe one vendored AMT backbone variant."""

    name: str
    filename: str
    decoder_channels: tuple[int, int, int]
    finetune_modules: tuple[str, ...]


AMT_S_SPEC = AMTBackboneSpec(
    name='amt-s',
    filename='AMT-S.py',
    decoder_channels=(20, 32, 44),
    finetune_modules=('decoder2', 'update2', 'decoder1', 'comb_block'),
)
AMT_L_SPEC = AMTBackboneSpec(
    name='amt-l',
    filename='AMT-L.py',
    decoder_channels=(48, 64, 72),
    finetune_modules=('decoder2', 'update2', 'decoder1', 'comb_block'),
)
AMT_G_SPEC = AMTBackboneSpec(
    name='amt-g',
    filename='AMT-G.py',
    decoder_channels=(84, 96, 112),
    finetune_modules=('decoder2', 'update2_low', 'update2_high', 'decoder1', 'comb_block'),
)


def load_amt_class(filename: str, module_name: str) -> type[nn.Module]:
    """Load an official AMT ``Model`` class without editing third-party code."""
    package_root = Path(__file__).parents[1] / 'third_party' / 'AMT_official'
    path = package_root / 'networks' / filename
    if not path.is_file():
        raise FileNotFoundError(f'Missing AMT backbone source: {path}')
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load AMT backbone from {path}')
    sys.path.insert(0, str(package_root))
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == str(package_root):
            sys.path.pop(0)
    return module.Model


class T2textureAMTBase(nn.Module):
    """T2exture wrapper around one pretrained AMT backbone."""

    def __init__(
        self,
        spec: AMTBackboneSpec,
        pretrained: str | Path | None,
        passive_context: int = 4,
        passive_temporal_modulation: bool = True,
    ) -> None:
        super().__init__()
        if passive_context < 0 or passive_context % 2:
            raise ValueError('passive_context must be zero or a positive even integer')
        self.spec = spec
        self.passive_context = passive_context
        self.t2v_adapter = T2VAdapter()
        self.texture_adapter = self.t2v_adapter
        self.passive_encoder = (
            PassiveGuidancePyramid(passive_context, spec.decoder_channels, temporal_modulation=passive_temporal_modulation)
            if passive_context
            else None
        )
        self.guidance_projections = (
            nn.ModuleList([nn.Conv2d(channels, channels, kernel_size=1) for channels in spec.decoder_channels])
            if passive_context
            else nn.ModuleList()
        )
        self.zero_convs = self.guidance_projections
        for layer in self.guidance_projections:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        backbone_class = load_amt_class(spec.filename, f't2texture_{spec.name.replace("-", "_")}_backbone')
        self.backbone = backbone_class()
        if pretrained is not None:
            state: dict[str, Any] = torch.load(pretrained, map_location='cpu', weights_only=False)
            self.backbone.load_state_dict(state.get('state_dict', state), strict=True)

        self._control: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._hooks = [
            self.backbone.decoder4.register_forward_hook(self._inject_level3),
            self.backbone.decoder3.register_forward_hook(self._inject_level2),
            self.backbone.decoder2.register_forward_hook(self._inject_level1),
        ]
        self.set_phase('adapter')

    def _inject(self, output: tuple[torch.Tensor, torch.Tensor, torch.Tensor], index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add one passive guidance field to an AMT decoder feature output."""
        if self._control is None:
            return output
        flow0, flow1, feature = output
        guidance = self.guidance_projections[index](self._control[index])
        return flow0, flow1, feature + guidance

    def _inject_level3(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._inject(output, 2)

    def _inject_level2(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._inject(output, 1)

    def _inject_level1(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._inject(output, 0)

    def set_phase(self, phase: str) -> None:
        """Freeze AMT or unfreeze the rear refinement modules for stage two."""
        if phase not in {'adapter', 'finetune'}:
            raise ValueError("phase must be 'adapter' or 'finetune'")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if phase == 'finetune':
            for module_name in self.spec.finetune_modules:
                module = getattr(self.backbone, module_name)
                for parameter in module.parameters():
                    parameter.requires_grad = True
        trainable_modules: list[nn.Module] = [self.t2v_adapter, self.guidance_projections]
        if self.passive_encoder is not None:
            trainable_modules.append(self.passive_encoder)
        for module in trainable_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def forward(
        self,
        texture0: torch.Tensor,
        texture1: torch.Tensor,
        time: torch.Tensor,
        passive_context: torch.Tensor,
        return_flow: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Predict the target texture frame ``I_t``."""
        if self.passive_encoder is None:
            if passive_context.shape[1] != 0:
                raise ValueError(f'Expected no passive frames, got {passive_context.shape[1]}')
            self._control = None
        else:
            self._control = self.passive_encoder(passive_context, time)
        try:
            output = self.backbone(
                self.t2v_adapter(texture0, time),
                self.t2v_adapter(texture1, time),
                time.view(time.shape[0], 1, 1, 1),
                eval=not return_flow,
            )
        finally:
            self._control = None
        prediction = output['imgt_pred'].mean(dim=1, keepdim=True)
        return {'prediction': prediction, **({key: value for key, value in output.items() if key != 'imgt_pred'})}


__all__ = [
    'AMTBackboneSpec',
    'AMT_G_SPEC',
    'AMT_L_SPEC',
    'AMT_S_SPEC',
    'T2textureAMTBase',
    'load_amt_class',
]
