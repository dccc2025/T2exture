"""AMT-S T2exture wrapper with passive-context conditioning."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from model.conditioning import FourierTimeEmbedding, SharedTimeMLP


def _load_amt_s() -> type[nn.Module]:
    """Load the vendored official AMT-S class without editing third-party code."""
    package_root = Path(__file__).parents[1] / 'third_party' / 'AMT_official'
    path = package_root / 'networks' / 'AMT-S.py'
    spec = importlib.util.spec_from_file_location('t2texture_amt_s_backbone', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load AMT-S from {path}')
    sys.path.insert(0, str(package_root))
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == str(package_root):
            sys.path.pop(0)
    return module.Model


class TextureAdapter(nn.Module):
    """Adapt one-channel texture frames to AMT-S RGB input space."""

    def __init__(self, condition_dim: int = 64) -> None:
        super().__init__()
        self.projection = nn.Conv2d(1, 3, kernel_size=1)
        nn.init.ones_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.time_bias = nn.Linear(condition_dim, 3)
        nn.init.zeros_(self.time_bias.weight)
        nn.init.zeros_(self.time_bias.bias)

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if image.shape[1] != 1:
            raise ValueError(f'TextureAdapter expects one input channel, got {image.shape[1]}')
        bias = self.time_bias(condition).unsqueeze(-1).unsqueeze(-1)
        return self.projection(image) + bias


class PassivePyramidEncoderS(nn.Module):
    """Encode passive context into AMT-S decoder feature widths."""

    def __init__(self, context_size: int, condition_dim: int = 64) -> None:
        super().__init__()
        self.context_size = context_size
        self.level1 = nn.Sequential(nn.Conv2d(context_size, 20, 3, 2, 1), nn.PReLU(20), nn.Conv2d(20, 20, 3, 1, 1), nn.PReLU(20))
        self.level2 = nn.Sequential(nn.Conv2d(20, 32, 3, 2, 1), nn.PReLU(32), nn.Conv2d(32, 32, 3, 1, 1), nn.PReLU(32))
        self.level3 = nn.Sequential(nn.Conv2d(32, 44, 3, 2, 1), nn.PReLU(44), nn.Conv2d(44, 44, 3, 1, 1), nn.PReLU(44))
        self.time_biases = nn.ModuleList([nn.Linear(condition_dim, channels) for channels in (20, 32, 44)])
        for layer in self.time_biases:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, passive_context: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if passive_context.shape[1] != self.context_size:
            raise ValueError(f'Expected {self.context_size} passive frames, got {passive_context.shape[1]}')
        first = self.level1(passive_context) + self.time_biases[0](condition).unsqueeze(-1).unsqueeze(-1)
        second = self.level2(first) + self.time_biases[1](condition).unsqueeze(-1).unsqueeze(-1)
        third = self.level3(second) + self.time_biases[2](condition).unsqueeze(-1).unsqueeze(-1)
        return first, second, third


class T2textureAMTS(nn.Module):
    """Fine-tune AMT-S using texture endpoints and passive observations."""

    def __init__(self, pretrained: str | Path, passive_context: int = 4) -> None:
        super().__init__()
        if passive_context <= 0 or passive_context % 2:
            raise ValueError('passive_context must be a positive even integer')
        self.passive_context = passive_context
        self.time_embedding = FourierTimeEmbedding(dim=32)
        self.time_mlp = SharedTimeMLP(input_dim=32, hidden_dim=64)
        self.texture_adapter = TextureAdapter(condition_dim=64)
        self.passive_encoder = PassivePyramidEncoderS(passive_context, condition_dim=64)
        self.zero_convs = nn.ModuleList([nn.Conv2d(channels, channels, 1) for channels in (20, 32, 44)])
        for layer in self.zero_convs:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.backbone = _load_amt_s()()
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
        if self._control is None:
            return output
        flow0, flow1, feature = output
        control = self._control[index]
        return flow0, flow1, feature + self.zero_convs[index](control)

    def _inject_level3(self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._inject(output, 2)

    def _inject_level2(self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._inject(output, 1)

    def _inject_level1(self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._inject(output, 0)

    def set_phase(self, phase: str) -> None:
        if phase not in {'adapter', 'finetune'}:
            raise ValueError("phase must be 'adapter' or 'finetune'")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if phase == 'finetune':
            for module in (self.backbone.decoder2, self.backbone.update2, self.backbone.decoder1, self.backbone.comb_block):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        for module in (self.time_embedding, self.time_mlp, self.texture_adapter, self.passive_encoder, self.zero_convs):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def forward(self, texture0: torch.Tensor, texture1: torch.Tensor, time: torch.Tensor, passive_context: torch.Tensor, return_flow: bool = False) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        condition = self.time_mlp(self.time_embedding(time))
        self._control = self.passive_encoder(passive_context, condition)
        try:
            output = self.backbone(
                self.texture_adapter(texture0, condition),
                self.texture_adapter(texture1, condition),
                time.view(time.shape[0], 1, 1, 1),
                eval=not return_flow,
            )
        finally:
            self._control = None
        prediction = output['imgt_pred'].mean(dim=1, keepdim=True)
        return {'prediction': prediction, **({key: value for key, value in output.items() if key != 'imgt_pred'})}
