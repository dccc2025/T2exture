"""Evaluate official GIMM-VFI-F checkpoints on the T2exture dataset split."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import resolve_sample_passive_context, validate_runtime_config
from data import TextureDataset
from eval import _batch_ints, _batch_strings, _mean_rows, _move_batch, _relative, _sample_name, _save_png
from metrics import MAIN_METRIC_KEYS, compute_main_metrics


class _CfgNode(dict):
    """Minimal yacs.CfgNode substitute needed by GIMM FlowFormer config."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def clone(self):
        return copy.deepcopy(self)


class _EasyDict(dict):
    """Minimal easydict.EasyDict substitute for GIMM config loading."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def install_config_stubs() -> None:
    if 'easydict' not in sys.modules:
        easydict_module = types.ModuleType('easydict')
        easydict_module.EasyDict = _EasyDict
        sys.modules['easydict'] = easydict_module
    if 'yacs' not in sys.modules:
        yacs_module = types.ModuleType('yacs')
        yacs_config_module = types.ModuleType('yacs.config')
        yacs_config_module.CfgNode = _CfgNode
        yacs_module.config = yacs_config_module
        sys.modules['yacs'] = yacs_module
        sys.modules['yacs.config'] = yacs_config_module
    if 'loguru' not in sys.modules:
        loguru_module = types.ModuleType('loguru')

        class _Logger:
            def __getattr__(self, _name: str):
                return lambda *args, **kwargs: None

        loguru_module.logger = _Logger()
        sys.modules['loguru'] = loguru_module
    try:
        import timm.models.helpers as timm_helpers
    except Exception:
        return
    if not hasattr(timm_helpers, 'overlay_external_default_cfg'):
        timm_helpers.overlay_external_default_cfg = lambda *args, **kwargs: None
    try:
        import timm.layers as timm_layers
        import timm.models.layers as legacy_timm_layers
    except Exception:
        return
    if not hasattr(legacy_timm_layers, 'activations') and hasattr(timm_layers, 'activations'):
        legacy_timm_layers.activations = timm_layers.activations


def install_cupy_compat() -> None:
    """Restore the CuPy API expected by GIMM's vendored softsplat kernel."""
    try:
        import cupy
        import cupy.cuda.compiler as cupy_compiler
    except Exception:
        return
    if hasattr(cupy.cuda, 'compile_with_cache'):
        return
    if not hasattr(cupy_compiler, '_compile_module_with_cache'):
        return

    def compile_with_cache(
        source,
        options=(),
        *,
        arch=None,
        extra_source=None,
        backend='nvrtc',
        enable_cooperative_groups=False,
        name_expressions=None,
        log_stream=None,
        jitify=False,
        to_ltoir=False,
    ):
        return cupy_compiler._compile_module_with_cache(
            source,
            options,
            arch=arch,
            extra_source=extra_source,
            backend=backend,
            enable_cooperative_groups=enable_cooperative_groups,
            name_expressions=name_expressions,
            log_stream=log_stream,
            jitify=jitify,
            to_ltoir=to_ltoir,
        )

    cupy.cuda.compile_with_cache = compile_with_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('pretrained/GIMM-VFI-F/gimmvfi_f_arb.pt'))
    parser.add_argument('--flowformer', type=Path, default=Path('pretrained/GIMM-VFI-F/flowformer_sintel.pth'))
    parser.add_argument('--model-config', type=Path, default=Path('third_party/GIMM-VFI/configs/gimmvfi/gimmvfi_f_arb.yaml'))
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='test')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('train.yaml'))
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--ds-factor', type=float, default=1.0)
    return parser.parse_args()


def texture_to_rgb(texture: torch.Tensor) -> torch.Tensor:
    if texture.shape[1] != 1:
        raise ValueError(f'Expected one-channel texture tensor, got {texture.shape}')
    return texture.repeat(1, 3, 1, 1)


def loader_kwargs(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    workers = int(config.get('num_workers', 0))
    kwargs: dict[str, Any] = {
        'num_workers': workers,
        'pin_memory': bool(config.get('pin_memory', False)) and device.type == 'cuda',
    }
    if workers > 0:
        kwargs['persistent_workers'] = bool(config.get('persistent_workers', False))
        kwargs['prefetch_factor'] = int(config.get('prefetch_factor', 2))
    return kwargs


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, Any, type]:
    package_src = ROOT / 'third_party' / 'GIMM-VFI' / 'src'
    isolated_names = [
        name
        for name in sys.modules
        if name == 'models'
        or name.startswith('models.')
        or name == 'utils'
        or name.startswith('utils.')
    ]
    previous_modules = {name: sys.modules[name] for name in isolated_names}
    for name in isolated_names:
        sys.modules.pop(name, None)

    install_config_stubs()
    install_cupy_compat()
    sys.path.insert(0, str(package_src))
    try:
        config_module = importlib.import_module('utils.config')
        model_module = importlib.import_module('models')
        padder_module = importlib.import_module('utils.utils')
        submission_module = importlib.import_module('models.generalizable_INR.flowformer.configs.submission')
        submission_module._CN.model = str(args.flowformer.resolve())

        config = config_module.augment_defaults(config_module.load_config(str(args.model_config.resolve())))
        model, _ = model_module.create_model(config.arch)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        result = model.load_state_dict(checkpoint['state_dict'], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f'GIMM-VFI-F checkpoint mismatch: missing={len(result.missing_keys)}, '
                f'unexpected={len(result.unexpected_keys)}'
            )
        model = model.to(device).eval()
    finally:
        if sys.path and sys.path[0] == str(package_src):
            sys.path.pop(0)
        loaded_names = [
            name
            for name in sys.modules
            if name == 'models'
            or name.startswith('models.')
            or name == 'utils'
            or name.startswith('utils.')
        ]
        for name in loaded_names:
            sys.modules.pop(name, None)
        sys.modules.update(previous_modules)
    return model, config, padder_module.InputPadder


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError('GIMM-VFI-F runner currently uses batch_size=1 because each sample has its own timestep.')

    runtime_config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    validate_runtime_config(runtime_config, args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    passive_context = int(runtime_config.get('passive_context', 4))
    active_stride = int(runtime_config.get('active_stride', 10))
    sample_passive_context = resolve_sample_passive_context(runtime_config, passive_context)
    pseudo_flow_root = Path(runtime_config['pseudo_flow_dir']) if runtime_config.get('pseudo_flow_dir') else None
    dataset = TextureDataset(
        args.data_root,
        args.data_root / f'{args.split}.txt',
        passive_context=passive_context,
        crop_size=None,
        random_crop=False,
        pseudo_flow_root=pseudo_flow_root,
        active_stride=active_stride,
        sample_passive_context=sample_passive_context,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, **loader_kwargs(runtime_config, device))
    model, gimm_config, input_padder = load_model(args, device)

    frame_rows: list[dict[str, Any]] = []
    scene_rows: dict[str, list[dict[str, float]]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            image0 = texture_to_rgb(batch['texture0'])
            image1 = texture_to_rgb(batch['texture1'])
            padder = input_padder(image0.shape, 32)
            image0, image1 = padder.pad(image0, image1)
            xs = torch.cat((image0.unsqueeze(2), image1.unsqueeze(2)), dim=2)
            timestep = float(batch['time'][0].detach().cpu())
            coord_inputs = [
                (
                    model.sample_coord_input(
                        xs.shape[0],
                        xs.shape[-2:],
                        [timestep],
                        device=xs.device,
                        upsample_ratio=args.ds_factor,
                    ),
                    None,
                )
            ]
            timesteps = [torch.tensor([timestep], device=xs.device, dtype=torch.float32)]
            output = model(xs, coord_inputs, t=timesteps, ds_factor=args.ds_factor)
            prediction = padder.unpad(output['imgt_pred'][0]).mean(dim=1, keepdim=True).clamp(0.0, 1.0)
            target = batch['target'].clamp(0.0, 1.0)
            metric_tensors = compute_main_metrics(prediction, target)

            scenes = _batch_strings(batch['scene'])
            left_ids = _batch_ints(batch['left_id'])
            target_ids = _batch_ints(batch['target_id'])
            right_ids = _batch_ints(batch['right_id'])
            scene = scenes[0]
            left_id = left_ids[0]
            target_id = target_ids[0]
            right_id = right_ids[0]
            name = _sample_name(left_id, target_id, right_id)
            pred_path = args.output_dir / 'pred' / scene / f'{name}.png'
            err_path = args.output_dir / 'err' / scene / f'{name}.png'
            _save_png(prediction[0], pred_path)
            _save_png((prediction[0] - target[0]).abs(), err_path)
            metrics = {key: float(metric_tensors[key][0].detach().cpu()) for key in MAIN_METRIC_KEYS}
            row = {
                'scene': scene,
                'left_id': left_id,
                'target_id': target_id,
                'right_id': right_id,
                **metrics,
                'pred_path': _relative(pred_path, args.output_dir),
                'err_path': _relative(err_path, args.output_dir),
            }
            frame_rows.append(row)
            scene_rows[scene].append(metrics)

    if not frame_rows:
        raise ValueError(f'No samples were evaluated for split {args.split!r}')

    overall = _mean_rows([{key: float(row[key]) for key in MAIN_METRIC_KEYS} for row in frame_rows])
    scene_summary = [{'scene': scene, 'frames': len(rows), **_mean_rows(rows)} for scene, rows in sorted(scene_rows.items())]
    write_csv(args.output_dir / 'frame.csv', frame_rows, ['scene', 'left_id', 'target_id', 'right_id', *MAIN_METRIC_KEYS, 'pred_path', 'err_path'])
    write_csv(args.output_dir / 'scene.csv', scene_summary, ['scene', 'frames', *MAIN_METRIC_KEYS])

    metrics = {
        'method': 'GIMM-VFI-F',
        'split': args.split,
        'frames': len(frame_rows),
        'overall': overall,
        'by_scene': scene_summary,
    }
    manifest = {
        **metrics,
        'data_root': str(args.data_root.resolve()),
        'checkpoint': str(args.checkpoint.resolve()),
        'flowformer': str(args.flowformer.resolve()),
        'model_config': str(args.model_config.resolve()),
        'ds_factor': args.ds_factor,
        'gimm_arch_type': str(gimm_config.arch.type),
        'config': runtime_config,
        'artifacts': {
            'pred': 'pred/',
            'err': 'err/',
            'frame_csv': 'frame.csv',
            'scene_csv': 'scene.csv',
            'metrics_json': 'metrics.json',
        },
    }
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metrics['overall'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
