"""Generate pseudo-flow labels for T2exture synthetic scenes.

This mirrors AMT's ``flow_generation/gen_flow.py``: raw synthetic frames stay in
``dataset/sim/<scene>/texture`` and derived flow labels are cached separately in
``dataset/flow/<flow_set>/<scene>``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from config import passive_context_ids
from flow_generation import DEFAULT_FLOW_SET, flow_file_name, resolve_pseudo_flow_root


def parse_args() -> argparse.Namespace:
    """Read dataset and split inputs for pseudo-flow generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--flow-set', default=DEFAULT_FLOW_SET, help='Pseudo-flow set name under DATA_ROOT/flow.')
    parser.add_argument('--output-dir', type=Path, default=None, help='Explicit pseudo-flow root. Overrides --flow-set.')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid'])
    parser.add_argument('--active-stride', type=int, default=10)
    parser.add_argument('--context-size', type=int, default=4)
    parser.add_argument('--liteflownet', type=Path, default=Path('third_party/AMT_official/flow_generation/liteflownet/run.py'))
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def read_split(path: Path) -> list[str]:
    """Read non-empty scene names from one split file."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def ensure_cupy_compile_with_cache() -> None:
    """Provide the legacy CuPy API expected by LiteFlowNet's correlation layer."""
    try:
        import cupy
    except ImportError as exc:
        raise ImportError('LiteFlowNet flow generation requires cupy-cuda12x in the active environment') from exc

    if hasattr(cupy.cuda, 'compile_with_cache'):
        return

    class _RawModuleWithFunction:
        def __init__(self, code: str):
            self.module = cupy.RawModule(code=code)

        def get_function(self, name: str):
            return self.module.get_function(name)

    cupy.cuda.compile_with_cache = _RawModuleWithFunction  # type: ignore[attr-defined]


def load_liteflownet(path: Path) -> ModuleType:
    """Import the vendored LiteFlowNet script without letting it parse our CLI flags."""
    ensure_cupy_compile_with_cache()
    original_argv = sys.argv[:]
    package_root = path.parent.parent
    try:
        sys.argv = [str(path)]
        sys.path.insert(0, str(package_root))
        return importlib.import_module(f'{path.parent.name}.run')
    finally:
        sys.argv = original_argv
        if sys.path and sys.path[0] == str(package_root):
            sys.path.pop(0)


def load_texture(path: Path) -> torch.Tensor:
    """Load one grayscale texture npy and replicate it to LiteFlowNet's RGB input."""
    image = np.load(path).astype(np.float32)
    if image.ndim != 2:
        raise ValueError(f'Expected 2-D texture frame at {path}, got {image.shape}')
    maximum = float(image.max())
    minimum = float(image.min())
    if maximum > minimum:
        image = (image - minimum) / (maximum - minimum)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    return torch.from_numpy(np.repeat(image[None, ...], 3, axis=0))


def texture_path(root: Path, scene: str, frame_id: int) -> Path:
    """Return the canonical synthetic texture frame path."""
    return root / 'sim' / scene / 'texture' / f'{frame_id:03d}.npy'


def output_flow_path(output_root: Path, scene: str, left_id: int, target_id: int, right_id: int) -> Path:
    """Return the AMT-style pseudo-flow cache path for one target."""
    return output_root / scene / flow_file_name(left_id, target_id, right_id)


@torch.no_grad()
def generate_scene(root: Path, output_root: Path, scene: str, estimate, active_stride: int, context_size: int) -> tuple[int, int]:
    """Generate all target-to-endpoint pseudo flows for one scene."""
    written = 0
    skipped = 0
    if active_stride <= 1:
        raise ValueError('active_stride must be larger than 1 so an interpolation target exists')
    for left_id in range(1, 181 - active_stride, active_stride):
        right_id = left_id + active_stride
        texture0 = load_texture(texture_path(root, scene, left_id))
        texture1 = load_texture(texture_path(root, scene, right_id))
        for offset in range(1, active_stride):
            target_id = left_id + offset
            if min(passive_context_ids(target_id, context_size)) < 1 or max(passive_context_ids(target_id, context_size)) > 180:
                skipped += 1
                continue
            out_path = output_flow_path(output_root, scene, left_id, target_id, right_id)
            if out_path.is_file():
                skipped += 1
                continue
            target = load_texture(texture_path(root, scene, target_id))
            flow0 = estimate(target, texture0).cpu().numpy().astype(np.float32)
            flow1 = estimate(target, texture1).cpu().numpy().astype(np.float32)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_path, flow0=flow0, flow1=flow1)
            written += 1
    return written, skipped


def main() -> None:
    """Generate pseudo-flow labels for the requested splits."""
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('LiteFlowNet generation requested CUDA, but CUDA is not available')
    output_root = resolve_pseudo_flow_root(args.data_root, args.output_dir or Path('flow') / args.flow_set)
    liteflownet = load_liteflownet(args.liteflownet)
    scenes = []
    for split in args.splits:
        scenes.extend(read_split(args.data_root / f'{split}.txt'))
    scenes = sorted(set(scenes))
    written = 0
    skipped = 0
    for scene in scenes:
        scene_written, scene_skipped = generate_scene(args.data_root, output_root, scene, liteflownet.estimate, args.active_stride, args.context_size)
        written += scene_written
        skipped += scene_skipped
        print(json.dumps({'scene': scene, 'written': scene_written, 'skipped': scene_skipped}), flush=True)
    manifest = {
        'source': 'LiteFlowNet via third_party/AMT_official/flow_generation/liteflownet',
        'data_root': str(args.data_root),
        'output_root': str(output_root),
        'flow_set': args.flow_set,
        'splits': args.splits,
        'scenes': scenes,
        'active_stride': args.active_stride,
        'context_size': args.context_size,
        'format': 'flow/<flow_set>/<scene>/<left>_<target>_<right>.npz with flow0 and flow1, each shaped [2,H,W], target-to-endpoint pixel flow',
        'written': written,
        'skipped': skipped,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'flow_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
