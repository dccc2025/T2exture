"""Profile AMT vanilla and T2exture wrappers for deployment Table 5."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics import count_parameters, count_trainable_parameters
from model import build_t2texture_model
from scripts.eval_amt_vanilla import load_model as load_amt_model


PRETRAINED = {
    'amt-s': Path('pretrained/amt-s.pth'),
    'amt-l': Path('pretrained/amt-l.pth'),
    'amt-g': Path('pretrained/amt-g.pth'),
}


def parse_args() -> argparse.Namespace:
    """Read model and profiling options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', choices=['amt-s', 'amt-l', 'amt-g'], required=True)
    parser.add_argument('--setting', choices=['vanilla', 't2texture'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('train.yaml'))
    parser.add_argument('--data-root', type=Path, default=Path('dataset_roi'))
    parser.add_argument('--pretrained', type=Path, default=None)
    parser.add_argument('--checkpoint', type=Path, default=None, help='Optional trained T2exture checkpoint; latency/FLOPs do not require it.')
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Expected a YAML mapping in {path}')
    return data


def roi_shape(data_root: Path, height: int | None, width: int | None) -> tuple[int, int]:
    """Resolve the profiling input shape from CLI or ROI manifest."""
    if height is not None and width is not None:
        return height, width
    manifest_path = data_root / 'roi_manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        shape = manifest.get('output_shape_hw')
        if isinstance(shape, list) and len(shape) == 2:
            return int(shape[0]), int(shape[1])
    return height or 640, width or 960


def load_t2texture(backbone: str, pretrained: Path, checkpoint: Path | None, passive_context: int, device: torch.device) -> nn.Module:
    """Load a T2exture wrapper and put it in its fine-tuning parameter state."""
    model = build_t2texture_model(backbone, pretrained, passive_context).to(device)
    model.set_phase('finetune')
    if checkpoint is not None and checkpoint.is_file():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        state_dict = state['model'] if isinstance(state, dict) and 'model' in state else state
        model.load_state_dict(state_dict, strict=True)
    return model.eval()


def rgb(image: torch.Tensor) -> torch.Tensor:
    """Replicate one-channel inputs for AMT's RGB interface."""
    return image.repeat(1, 3, 1, 1)


def build_profile_target(args: argparse.Namespace, device: torch.device, config: dict[str, Any], height: int, width: int) -> tuple[nn.Module, Callable[[], Any]]:
    """Build one model and a no-argument forward callback."""
    pretrained = args.pretrained or PRETRAINED[args.backbone]
    passive_context = int(config.get('passive_context', 4))
    x0 = torch.rand(1, 1, height, width, device=device)
    x1 = torch.rand(1, 1, height, width, device=device)
    t = torch.tensor([0.5], device=device, dtype=torch.float32)
    context = torch.rand(1, passive_context, height, width, device=device)

    if args.setting == 'vanilla':
        model = load_amt_model(args.backbone, pretrained, device)
        for parameter in model.parameters():
            parameter.requires_grad = False

        def forward() -> Any:
            return model(rgb(x0), rgb(x1), t.view(1, 1, 1, 1), eval=True)

        return model, forward

    model = load_t2texture(args.backbone, pretrained, args.checkpoint, passive_context, device)

    def forward() -> Any:
        return model(x0, x1, t, context, return_flow=False)

    return model, forward


def sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize()


def measure_latency(forward: Callable[[], Any], device: torch.device, warmup: int, repeats: int) -> float:
    """Return average forward latency in milliseconds."""
    if warmup < 0 or repeats <= 0:
        raise ValueError('warmup must be non-negative and repeats must be positive')
    with torch.no_grad():
        for _ in range(warmup):
            forward()
        sync(device)
        if device.type == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                forward()
            end.record()
            torch.cuda.synchronize()
            return float(start.elapsed_time(end) / repeats)
        start_time = time.perf_counter()
        for _ in range(repeats):
            forward()
        return float((time.perf_counter() - start_time) * 1000.0 / repeats)


def profiler_flops(forward: Callable[[], Any], device: torch.device) -> int:
    """Try to collect FLOPs from torch.profiler."""
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.no_grad(), torch.profiler.profile(activities=activities, with_flops=True) as profile:
            forward()
        total = sum(int(event.flops or 0) for event in profile.key_averages())
        return total
    except Exception:
        return 0


def hook_flops(model: nn.Module, forward: Callable[[], Any]) -> int:
    """Fallback Conv2d/Linear FLOPs counter, counting multiply-add as two FLOPs."""
    total = 0

    def conv_hook(module: nn.Conv2d, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        if not torch.is_tensor(output):
            return
        batch, out_channels, out_h, out_w = output.shape
        kernel_h, kernel_w = module.kernel_size
        in_channels = module.in_channels // module.groups
        total += int(batch * out_channels * out_h * out_w * in_channels * kernel_h * kernel_w * 2)

    def linear_hook(module: nn.Linear, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        if not torch.is_tensor(output):
            return
        elements = output.numel()
        total += int(elements * module.in_features * 2)

    handles = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    try:
        with torch.no_grad():
            forward()
    finally:
        for handle in handles:
            handle.remove()
    return total


def measure_flops(model: nn.Module, forward: Callable[[], Any], device: torch.device) -> tuple[float, str]:
    """Return GFLOPs and the profiling method used."""
    flops = profiler_flops(forward, device)
    method = 'torch.profiler'
    if flops <= 0:
        flops = hook_flops(model, forward)
        method = 'conv_linear_hooks'
    return float(flops / 1e9), method


def main() -> None:
    """Write one runtime.json artifact for Table 5."""
    args = parse_args()
    config = read_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

    height, width = roi_shape(args.data_root, args.height, args.width)
    model, forward = build_profile_target(args, device, config, height, width)
    latency_ms = measure_latency(forward, device, args.warmup, args.repeats)
    gflops, flops_method = measure_flops(model, forward, device)

    overall = {
        'Latency': latency_ms,
        'FLOPs': gflops,
        'Params': count_parameters(model) / 1e6,
        'Trainable Params': count_trainable_parameters(model) / 1e6,
    }
    runtime = {
        'backbone': args.backbone,
        'setting': args.setting,
        'overall': overall,
        'units': {
            'Latency': 'ms per frame, batch=1',
            'FLOPs': 'GFLOPs per forward',
            'Params': 'M',
            'Trainable Params': 'M',
        },
        'input_shape': {'batch': 1, 'channels': 1 if args.setting == 't2texture' else 3, 'height': height, 'width': width},
        'device': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'warmup': args.warmup,
        'repeats': args.repeats,
        'flops_method': flops_method,
        'config': str(args.config.resolve()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(overall, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
