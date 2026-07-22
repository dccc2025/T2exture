from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class EdgeControlEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('sobel_x', torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3))
        self.level1 = nn.Sequential(nn.Conv2d(4, 48, 3, 2, 1), nn.PReLU(48), nn.Conv2d(48, 48, 3, 1, 1), nn.PReLU(48))
        self.level2 = nn.Sequential(nn.Conv2d(48, 64, 3, 2, 1), nn.PReLU(64), nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(64))
        self.level3 = nn.Sequential(nn.Conv2d(64, 72, 3, 2, 1), nn.PReLU(72), nn.Conv2d(72, 72, 3, 1, 1), nn.PReLU(72))

    def forward(self, passive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gx = F.conv2d(passive, self.sobel_x, padding=1)
        gy = F.conv2d(passive, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)
        threshold = magnitude.flatten(1).quantile(0.90, dim=1).view(-1, 1, 1, 1)
        mask = torch.sigmoid((magnitude - threshold) / 0.03)
        first = self.level1(torch.cat([passive, gx, gy, mask], dim=1))
        second = self.level2(first)
        third = self.level3(second)
        return first, second, third


class EdgeConditionedAMT(nn.Module):
    def __init__(self, checkpoint: str | Path | None = None, load_pretrained: bool = True):
        super().__init__()
        module_path = Path(__file__).parent / 'networks' / 'AMT-L.py'
        spec = importlib.util.spec_from_file_location('amt_l', module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Cannot load AMT-L from {module_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.backbone = module.Model()
        if load_pretrained:
            checkpoint = Path(checkpoint or Path(__file__).parent / 'pretrained' / 'amt-l.pth')
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
            self.backbone.load_state_dict(state, strict=True)
        self.edge_encoder = EdgeControlEncoder()
        self.z1 = _zero(nn.Conv2d(48, 48, 1))
        self.z2 = _zero(nn.Conv2d(64, 64, 1))
        self.z3 = _zero(nn.Conv2d(72, 72, 1))
        self.final_delta = _zero(nn.ConvTranspose2d(48, 40, 4, 2, 1))
        self._control: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._hooks = [self.backbone.decoder4.register_forward_hook(self._hook4),
                       self.backbone.decoder3.register_forward_hook(self._hook3),
                       self.backbone.decoder2.register_forward_hook(self._hook2),
                       self.backbone.decoder1.register_forward_hook(self._hook1)]
        self.set_phase('adapter')

    def _hook4(self, _module, _inputs, output):
        if self._control is None: return output
        f0, f1, feature = output
        return f0, f1, feature + self.z3(self._control[2])

    def _hook3(self, _module, _inputs, output):
        if self._control is None: return output
        f0, f1, feature = output
        return f0, f1, feature + self.z2(self._control[1])

    def _hook2(self, _module, _inputs, output):
        if self._control is None: return output
        f0, f1, feature = output
        return f0, f1, feature + self.z1(self._control[0])

    def _hook1(self, _module, _inputs, output):
        if self._control is None: return output
        flow0, flow1, mask, residual = output
        delta = self.final_delta(self._control[0])
        d0, d1, dm, dr = torch.split(delta, [10, 10, 5, 15], dim=1)
        return flow0 + d0, flow1 + d1, torch.clamp(mask + dm, 0., 1.), residual + dr

    def adapter_parameters(self):
        for module in (self.edge_encoder, self.z1, self.z2, self.z3, self.final_delta):
            yield from module.parameters()

    def set_phase(self, phase: str) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.adapter_parameters():
            parameter.requires_grad = True
        if phase == 'refine':
            for module in (self.backbone.decoder1, self.backbone.decoder2, self.backbone.update2, self.backbone.comb_block):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        elif phase != 'adapter':
            raise ValueError(f'Unknown phase {phase}')

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, time: torch.Tensor, passive: torch.Tensor) -> torch.Tensor:
        self._control = self.edge_encoder(passive)
        try:
            return self.backbone(x0, x1, time, eval=True)['imgt_pred']
        finally:
            self._control = None


class ThermalInputAdapter(nn.Module):
    """A trainable single-channel to AMT-RGB adapter, initialized as replication."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(1, 3, 1)
        nn.init.ones_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] != 1:
            raise ValueError(f'ThermalInputAdapter expects one channel, got {image.shape[1]}')
        return self.proj(image)


def _backwarp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample ``image`` at pixel offsets ``flow`` (dx, dy), with stable borders."""
    batch, _, height, width = image.shape
    if flow.shape != (batch, 2, height, width):
        raise ValueError(f'Expected flow {(batch, 2, height, width)}, got {tuple(flow.shape)}')
    yy, xx = torch.meshgrid(
        torch.arange(height, device=image.device, dtype=image.dtype),
        torch.arange(width, device=image.device, dtype=image.dtype), indexing='ij')
    grid_x = (xx.unsqueeze(0) + flow[:, 0]) * (2.0 / max(width - 1, 1)) - 1.0
    grid_y = (yy.unsqueeze(0) + flow[:, 1]) * (2.0 / max(height - 1, 1)) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)


class PassiveGeometryFusion(nn.Module):
    """Use passive temporal context only to estimate endpoint geometry and fusion gates."""

    def __init__(self, max_displacement: float = 32.0, context_radius: int = 1):
        super().__init__()
        if context_radius < 0:
            raise ValueError('context_radius must be non-negative')
        self.max_displacement = max_displacement
        self.context_radius = int(context_radius)
        self.context_channels = 2 * self.context_radius + 1
        self.register_buffer('sobel_x', torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3))
        self.encoder = nn.Sequential(
            nn.Conv2d(self.context_channels + 6, 32, 3, 2, 1), nn.PReLU(32),
            nn.Conv2d(32, 48, 3, 1, 1), nn.PReLU(48),
            nn.Conv2d(48, 48, 3, 1, 1), nn.PReLU(48),
        )
        self.flow_head = nn.ConvTranspose2d(48, 4, 4, 2, 1)
        self.gate_head = nn.ConvTranspose2d(48, 2, 4, 2, 1)
        _zero(self.flow_head)
        _zero(self.gate_head)

    def forward(self, p0: torch.Tensor, p1: torch.Tensor, pt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for passive in (p0, p1):
            if passive.shape[1] != 1:
                raise ValueError('PassiveGeometryFusion expects one-channel passive endpoints')
        if pt.shape[1] == 1:
            context = pt.repeat(1, self.context_channels, 1, 1)
        elif pt.shape[1] == self.context_channels:
            context = pt
        else:
            raise ValueError(f'PassiveGeometryFusion expects pt with one channel or {self.context_channels} context channels')
        p_cur = context[:, self.context_radius:self.context_radius + 1]
        gx = F.conv2d(p_cur, self.sobel_x, padding=1)
        gy = F.conv2d(p_cur, self.sobel_y, padding=1)
        features = self.encoder(torch.cat((
            p0, p1,
            context,
            p_cur - p0, p_cur - p1,
            gx, gy,
        ), dim=1))
        flow = self.max_displacement * torch.tanh(self.flow_head(features))
        # tanh makes the zero-initialized branch exactly neutral while retaining
        # a non-zero derivative for learning a spatially varying fusion weight.
        gates = torch.tanh(self.gate_head(features))
        return flow, gates


class GeometryGatedAMT(nn.Module):
    """Frozen AMT-L plus passive-only endpoint alignment and learned fusion gates."""

    def __init__(self, checkpoint: str | Path | None = None, load_pretrained: bool = True,
                 context_radius: int = 1, use_passive: bool = True,
                 refine_modules: str = 'all'):
        super().__init__()
        self.use_passive = use_passive
        self.refine_modules = refine_modules
        module_path = Path(__file__).parent / 'networks' / 'AMT-L.py'
        spec = importlib.util.spec_from_file_location('amt_l_geometry_gated', module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f'Cannot load AMT-L from {module_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.backbone = module.Model()
        if load_pretrained:
            checkpoint = Path(checkpoint or Path(__file__).parent / 'pretrained' / 'amt-l.pth')
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
            self.backbone.load_state_dict(state, strict=True)
        self.input_adapter = ThermalInputAdapter()
        self.geometry = PassiveGeometryFusion(context_radius=context_radius)
        self.set_phase('adapter')

    def adapter_parameters(self):
        yield from self.input_adapter.parameters()
        if self.use_passive:
            yield from self.geometry.parameters()

    def set_phase(self, phase: str) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.adapter_parameters():
            parameter.requires_grad = True
        if phase == 'refine':
            selected = self._refine_modules()
            for module in selected:
                for parameter in module.parameters():
                    parameter.requires_grad = True
        elif phase != 'adapter':
            raise ValueError(f'Unknown phase {phase}')

    def _refine_modules(self) -> tuple[nn.Module, ...]:
        available = {
            'decoder1': (self.backbone.decoder1,),
            'decoder2': (self.backbone.decoder2,),
            'update2': (self.backbone.update2,),
            'comb': (self.backbone.comb_block,),
            'decoder12': (self.backbone.decoder1, self.backbone.decoder2),
            'all': (self.backbone.decoder1, self.backbone.decoder2,
                    self.backbone.update2, self.backbone.comb_block),
            'none': (),
        }
        modules: list[nn.Module] = []
        for name in (item.strip() for item in self.refine_modules.split(',') if item.strip()):
            if name not in available:
                raise ValueError(f'Unknown refine module {name}; choose from {sorted(available)}')
            modules.extend(available[name])
        return tuple(dict.fromkeys(modules))

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, time: torch.Tensor,
                p0: torch.Tensor, p1: torch.Tensor, pt: torch.Tensor) -> torch.Tensor:
        x0_rgb, x1_rgb = self.input_adapter(x0), self.input_adapter(x1)
        base = self.backbone(x0_rgb, x1_rgb, time, eval=True)['imgt_pred']
        if not self.use_passive:
            return base
        flow, gates = self.geometry(p0, p1, pt)
        warp0 = _backwarp(x0_rgb, flow[:, :2])
        warp1 = _backwarp(x1_rgb, flow[:, 2:])
        # P contributes only the flows and weights. Texture values come solely
        # from AMT's two-endpoint prediction and the warped X endpoints.
        return base + gates[:, :1] * (warp0 - base) + gates[:, 1:] * (warp1 - base)
