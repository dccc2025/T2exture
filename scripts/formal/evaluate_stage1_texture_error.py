"""Evaluate Stage-1 passive interpolation as texture-extraction error.

For each test-frame triplet, Stage 1 predicts the middle passive frame from
its two neighboring passive frames. The predicted texture is then extracted as

    X_hat_t = clamp(A_t - P_hat_t, 0), where A_t = X_t + P_t.

Metrics are reported between normalized X_hat_t and the ground-truth texture
X_t. This script is appendix-only and does not alter the main evaluation code.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import pathlib
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import _load_frame, _normalise, read_split
from eval import _state_dict, _torch_load
from flow_generation.generate_liteflownet_flow import load_liteflownet
from metrics import MAIN_METRIC_KEYS, batch_ie, compute_main_metrics
from scripts.baselines.evaluate_amt_vanilla import load_model as load_amt_model
from scripts.baselines.evaluate_amt_vanilla import texture_to_rgb


METHODS = ("liteflownet", "raft", "gma", "amt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--method", choices=[*METHODS, "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final/appendix_stage1_texture_error"))
    parser.add_argument("--appendix-figure-dir", type=Path, default=Path("appendix/figures"))
    parser.add_argument("--amt-pretrained", type=Path, default=Path("pretrained/amt-l.pth"))
    parser.add_argument("--amt-backbone", choices=["amt-s", "amt-l", "amt-g"], default="amt-l")
    parser.add_argument("--liteflownet", type=Path, default=Path("third_party/AMT_official/flow_generation/liteflownet/run.py"))
    parser.add_argument("--raft-root", type=Path, default=Path("third_party/RAFT"))
    parser.add_argument("--gma-root", type=Path, default=Path("third_party/GMA"))
    parser.add_argument("--raft-checkpoint", type=Path, default=None)
    parser.add_argument("--gma-checkpoint", type=Path, default=None)
    parser.add_argument("--raft-iters", type=int, default=20)
    parser.add_argument("--gma-iters", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit per method.")
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args()


def torch_device(requested: str) -> torch.device:
    return torch.device(requested if requested == "cpu" or torch.cuda.is_available() else "cpu")


def save_gray_png(image: np.ndarray, path: Path, normalise: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = image.astype(np.float32, copy=False)
    if normalise:
        array = _normalise(array)
    array = np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def save_error_png(error: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = error.astype(np.float32, copy=False)
    scale = float(np.percentile(array, 99.0))
    if scale <= 0.0:
        scale = float(array.max())
    if scale <= 0.0:
        scale = 1.0
    array = np.round(np.clip(array / scale, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def sample_name(target_id: int) -> str:
    return f"{target_id - 1:03d}_{target_id:03d}_{target_id + 1:03d}"


def shared_minmax(*images: np.ndarray) -> tuple[float, float]:
    minimum = min(float(image.min()) for image in images)
    maximum = max(float(image.max()) for image in images)
    if maximum <= minimum:
        maximum = minimum + 1.0
    return minimum, maximum


def normalize_with_range(image: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    return ((image.astype(np.float32) - minimum) / max(maximum - minimum, 1e-12)).clip(0.0, 1.0)


def to_u8_pair(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    minimum, maximum = shared_minmax(first, second)
    return (
        np.round(normalize_with_range(first, minimum, maximum) * 255.0).astype(np.uint8),
        np.round(normalize_with_range(second, minimum, maximum) * 255.0).astype(np.uint8),
        minimum,
        maximum,
    )


def to_rgb_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = np.repeat(image[None, ...].astype(np.float32), 3, axis=0)
    return torch.from_numpy(rgb.copy()).to(device=device)


def numpy_to_metric_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(_normalise(image).copy()).to(device=device).view(1, 1, *image.shape)


def backward_warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    if flow.shape[:2] != image.shape or flow.shape[2] != 2:
        raise ValueError(f"Flow shape {flow.shape} does not match image shape {image.shape}")
    height, width = image.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    warped = cv2.remap(
        image.astype(np.float32),
        xx - flow[..., 0].astype(np.float32),
        yy - flow[..., 1].astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped.astype(np.float32)


def halfway_flow_blend(left_raw: np.ndarray, right_raw: np.ndarray, estimate_flow: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    left_u8, right_u8, _, _ = to_u8_pair(left_raw, right_raw)
    forward = estimate_flow(left_u8, right_u8)
    backward = estimate_flow(right_u8, left_u8)
    left_mid = backward_warp(left_raw, 0.5 * forward)
    right_mid = backward_warp(right_raw, 0.5 * backward)
    return (0.5 * (left_mid + right_mid)).astype(np.float32)


@dataclass
class Predictor:
    method: str
    predict: Callable[[np.ndarray, np.ndarray], np.ndarray]
    description: str


def load_amt_predictor(args: argparse.Namespace, device: torch.device) -> Predictor:
    clear_third_party_flow_modules()
    model = load_amt_model(args.amt_backbone, args.amt_pretrained, device)

    @torch.no_grad()
    def predict(left_raw: np.ndarray, right_raw: np.ndarray) -> np.ndarray:
        minimum, maximum = shared_minmax(left_raw, right_raw)
        left = torch.from_numpy(normalize_with_range(left_raw, minimum, maximum).copy()).view(1, 1, *left_raw.shape).to(device)
        right = torch.from_numpy(normalize_with_range(right_raw, minimum, maximum).copy()).view(1, 1, *right_raw.shape).to(device)
        time = torch.full((1, 1, 1, 1), 0.5, device=device, dtype=left.dtype)
        output = model(texture_to_rgb(left), texture_to_rgb(right), time, eval=True)
        pred_norm = output["imgt_pred"].mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        pred = pred_norm[0, 0].detach().cpu().numpy().astype(np.float32)
        return pred * (maximum - minimum) + minimum

    return Predictor("amt", predict, f"{args.amt_backbone.upper()} direct passive frame interpolation at t=0.5")


def load_liteflownet_predictor(args: argparse.Namespace, device: torch.device) -> Predictor:
    if device.type != "cuda":
        raise RuntimeError("LiteFlowNet predictor requires CUDA in the vendored implementation.")
    liteflownet = load_liteflownet(args.liteflownet)

    @torch.no_grad()
    def estimate_flow(left_u8: np.ndarray, right_u8: np.ndarray) -> np.ndarray:
        left = to_rgb_tensor(left_u8.astype(np.float32) / 255.0, device)
        right = to_rgb_tensor(right_u8.astype(np.float32) / 255.0, device)
        flow = liteflownet.estimate(left.detach().cpu(), right.detach().cpu()).numpy().astype(np.float32)
        return np.moveaxis(flow, 0, -1)

    return Predictor(
        "liteflownet",
        lambda left, right: halfway_flow_blend(left, right, estimate_flow),
        "LiteFlowNet bidirectional optical flow with half-step warping and averaging",
    )


def clear_third_party_flow_modules() -> None:
    names = []
    for name, module in list(sys.modules.items()):
        path = (getattr(module, "__file__", "") or "").replace("\\", "/")
        if "/baselines/RAFT/core/" in path or "/baselines/GMA/core/" in path:
            names.append(name)
        elif name in {"raft", "network", "gma", "corr", "extractor", "update", "utils"} or name.startswith("utils."):
            names.append(name)
    for name in names:
        sys.modules.pop(name, None)


class AttrDict(dict):
    def __getattr__(self, key: str) -> Any:
        return self[key]


def resolve_first(candidates: list[Path], explicit: Path | None = None) -> Path:
    if explicit is not None:
        path = explicit if explicit.is_absolute() else ROOT / explicit
        if path.is_file():
            return path
        raise FileNotFoundError(path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("None of the candidate checkpoints exists:\n" + "\n".join(str(item) for item in candidates))


def load_raft_predictor(args: argparse.Namespace, device: torch.device) -> Predictor:
    raft_root = args.raft_root if args.raft_root.is_absolute() else ROOT / args.raft_root
    checkpoint = resolve_first(
        [
            raft_root / "models" / "raft-sintel.pth",
            raft_root / "models" / "models" / "raft-sintel.pth",
            raft_root / "models" / "raft-things.pth",
            raft_root / "models" / "models" / "raft-things.pth",
        ],
        args.raft_checkpoint,
    )
    clear_third_party_flow_modules()
    core_path = str(raft_root / "core")
    sys.path.insert(0, core_path)
    try:
        from raft import RAFT  # type: ignore
        from utils.utils import InputPadder  # type: ignore
    finally:
        if sys.path and sys.path[0] == core_path:
            sys.path.pop(0)
    model = torch.nn.DataParallel(RAFT(AttrDict(small=False, mixed_precision=False, alternate_corr=False, dropout=0.0)))
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model = model.module.to(device).eval()

    @torch.no_grad()
    def estimate_flow(left_u8: np.ndarray, right_u8: np.ndarray) -> np.ndarray:
        left = torch.from_numpy(np.repeat(left_u8[None, ...], 3, axis=0).copy()).float().view(1, 3, *left_u8.shape).to(device)
        right = torch.from_numpy(np.repeat(right_u8[None, ...], 3, axis=0).copy()).float().view(1, 3, *right_u8.shape).to(device)
        padder = InputPadder(left.shape)
        left_pad, right_pad = padder.pad(left, right)
        _, flow = model(left_pad, right_pad, iters=args.raft_iters, test_mode=True)
        flow = padder.unpad(flow)[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        return flow

    return Predictor("raft", lambda left, right: halfway_flow_blend(left, right, estimate_flow), f"RAFT Sintel flow, {args.raft_iters} iterations")


def load_gma_predictor(args: argparse.Namespace, device: torch.device) -> Predictor:
    gma_root = args.gma_root if args.gma_root.is_absolute() else ROOT / args.gma_root
    checkpoint = resolve_first(
        [
            gma_root / "checkpoints" / "gma-sintel.pth",
            gma_root / "checkpoints" / "gma-things.pth",
        ],
        args.gma_checkpoint,
    )
    clear_third_party_flow_modules()
    core_path = str(gma_root / "core")
    sys.path.insert(0, core_path)
    try:
        from network import RAFTGMA  # type: ignore
        from utils.utils import InputPadder  # type: ignore
    finally:
        if sys.path and sys.path[0] == core_path:
            sys.path.pop(0)
    model_args = AttrDict(
        dropout=0.0,
        mixed_precision=False,
        num_heads=1,
        position_only=False,
        position_and_content=False,
        corr_levels=4,
        corr_radius=4,
    )
    model = torch.nn.DataParallel(RAFTGMA(model_args))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if state and not any(str(key).startswith("module.") for key in state):
        state = {f"module.{key}": value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.module.to(device).eval()

    @torch.no_grad()
    def estimate_flow(left_u8: np.ndarray, right_u8: np.ndarray) -> np.ndarray:
        left = torch.from_numpy(np.repeat(left_u8[None, ...], 3, axis=0).copy()).float().view(1, 3, *left_u8.shape).to(device)
        right = torch.from_numpy(np.repeat(right_u8[None, ...], 3, axis=0).copy()).float().view(1, 3, *right_u8.shape).to(device)
        padder = InputPadder(left.shape)
        left_pad, right_pad = padder.pad(left, right)
        _, flow = model(left_pad, right_pad, iters=args.gma_iters, test_mode=True)
        flow = padder.unpad(flow)[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        return flow

    return Predictor("gma", lambda left, right: halfway_flow_blend(left, right, estimate_flow), f"GMA Sintel flow, {args.gma_iters} iterations")


def load_predictor(method: str, args: argparse.Namespace, device: torch.device) -> Predictor:
    if method == "amt":
        return load_amt_predictor(args, device)
    if method == "liteflownet":
        return load_liteflownet_predictor(args, device)
    if method == "raft":
        return load_raft_predictor(args, device)
    if method == "gma":
        return load_gma_predictor(args, device)
    raise ValueError(method)


def metric_row(pred_texture_raw: np.ndarray, gt_texture_raw: np.ndarray, pred_passive_raw: np.ndarray, gt_passive_raw: np.ndarray, device: torch.device) -> dict[str, float]:
    pred_texture = numpy_to_metric_tensor(pred_texture_raw, device)
    gt_texture = numpy_to_metric_tensor(gt_texture_raw, device)
    metrics = {key: float(value[0].detach().cpu()) for key, value in compute_main_metrics(pred_texture, gt_texture).items()}
    pred_passive = torch.from_numpy(normalize_with_range(pred_passive_raw, *shared_minmax(gt_passive_raw, pred_passive_raw)).copy()).to(device).view(1, 1, *gt_passive_raw.shape)
    gt_passive = torch.from_numpy(normalize_with_range(gt_passive_raw, *shared_minmax(gt_passive_raw, pred_passive_raw)).copy()).to(device).view(1, 1, *gt_passive_raw.shape)
    metrics["Passive-IE"] = float(batch_ie(pred_passive, gt_passive)[0].detach().cpu())
    return metrics


def mean_rows(rows: list[dict[str, float]], keys: tuple[str, ...] | list[str]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot average empty metrics")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


@torch.no_grad()
def evaluate_method(args: argparse.Namespace, method: str, scenes: list[str], device: torch.device) -> dict[str, Any]:
    predictor = load_predictor(method, args, device)
    output_dir = args.output_dir / method / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, Any]] = []
    scene_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    metric_keys = [*MAIN_METRIC_KEYS, "Passive-IE"]

    for scene in scenes:
        for target_id in range(2, 180):
            left_id = target_id - 1
            right_id = target_id + 1
            name = sample_name(target_id)
            left_passive = _load_frame(args.data_root, scene, "passive", left_id)
            gt_passive = _load_frame(args.data_root, scene, "passive", target_id)
            right_passive = _load_frame(args.data_root, scene, "passive", right_id)
            gt_texture = _load_frame(args.data_root, scene, "texture", target_id)
            active_raw = gt_texture + gt_passive

            pred_passive = predictor.predict(left_passive, right_passive).astype(np.float32)
            pred_texture = np.maximum(active_raw - pred_passive, 0.0).astype(np.float32)
            texture_error = np.abs(_normalise(pred_texture) - _normalise(gt_texture)).astype(np.float32)
            passive_error = np.abs(pred_passive - gt_passive).astype(np.float32)

            pred_passive_npy = output_dir / "pred_passive_npy" / scene / f"{name}.npy"
            pred_texture_npy = output_dir / "pred_texture_npy" / scene / f"{name}.npy"
            pred_passive_npy.parent.mkdir(parents=True, exist_ok=True)
            pred_texture_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(pred_passive_npy, pred_passive)
            np.save(pred_texture_npy, pred_texture)

            pred_passive_png = output_dir / "pred_passive" / scene / f"{name}.png"
            pred_texture_png = output_dir / "pred_texture" / scene / f"{name}.png"
            gt_texture_png = output_dir / "gt_texture" / scene / f"{name}.png"
            err_passive_png = output_dir / "err_passive" / scene / f"{name}.png"
            err_texture_png = output_dir / "err_texture" / scene / f"{name}.png"
            save_gray_png(pred_passive, pred_passive_png)
            save_gray_png(pred_texture, pred_texture_png)
            save_gray_png(gt_texture, gt_texture_png)
            save_error_png(passive_error, err_passive_png)
            save_error_png(texture_error, err_texture_png)

            metrics = metric_row(pred_texture, gt_texture, pred_passive, gt_passive, device)
            row = {
                "scene": scene,
                "left_id": left_id,
                "target_id": target_id,
                "right_id": right_id,
                **metrics,
                "pred_passive_path": relative(pred_passive_png, output_dir),
                "pred_texture_path": relative(pred_texture_png, output_dir),
                "gt_texture_path": relative(gt_texture_png, output_dir),
                "err_passive_path": relative(err_passive_png, output_dir),
                "err_texture_path": relative(err_texture_png, output_dir),
            }
            frame_rows.append(row)
            scene_rows[scene].append(metrics)

            if len(frame_rows) % 50 == 0:
                print(json.dumps({"method": method, "frames": len(frame_rows), "latest": f"{scene}/{name}"}), flush=True)
            if args.limit is not None and len(frame_rows) >= args.limit:
                break
        if args.limit is not None and len(frame_rows) >= args.limit:
            break

    if not frame_rows:
        raise ValueError(f"No frames evaluated for {method}")

    overall = mean_rows([{key: float(row[key]) for key in metric_keys} for row in frame_rows], metric_keys)
    scene_summary = [
        {"scene": scene, "frames": len(rows), **mean_rows(rows, metric_keys)}
        for scene, rows in sorted(scene_rows.items())
    ]
    frame_fields = [
        "scene",
        "left_id",
        "target_id",
        "right_id",
        *metric_keys,
        "pred_passive_path",
        "pred_texture_path",
        "gt_texture_path",
        "err_passive_path",
        "err_texture_path",
    ]
    write_csv(output_dir / "frame.csv", frame_rows, frame_fields)
    write_csv(output_dir / "scene.csv", scene_summary, ["scene", "frames", *metric_keys])

    metrics_json = {
        "method": method,
        "description": predictor.description,
        "split": args.split,
        "frames": len(frame_rows),
        "overall": overall,
        "by_scene": scene_summary,
    }
    manifest = {
        **metrics_json,
        "diagnostic": "stage1_passive_interpolation_texture_extraction_error",
        "data_root": str(args.data_root.resolve()),
        "formula": "A_t = X_t + P_t; X_hat_t = clamp(A_t - P_hat_t, 0); metrics compare normalized X_hat_t and X_t.",
        "input_rule": "P_{t-1} and P_{t+1} are used to estimate P_t on test scenes, frames 002..179.",
        "artifacts": {
            "pred_passive": "pred_passive/",
            "pred_texture": "pred_texture/",
            "gt_texture": "gt_texture/",
            "err_passive": "err_passive/",
            "err_texture": "err_texture/",
            "frame_csv": "frame.csv",
            "scene_csv": "scene.csv",
            "metrics_json": "metrics.json",
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2) + "\n", encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return metrics_json


def load_frame_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


def _as_pil_gray(image: np.ndarray) -> Image.Image:
    return Image.fromarray(np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")


def _square_bbox(mask: np.ndarray, margin: int = 34) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    height, width = mask.shape
    if len(xs) == 0:
        return (0, 0, min(height, width), min(height, width))
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(width, int(xs.max()) + 1 + margin)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(height, int(ys.max()) + 1 + margin)
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2
    side = max(x1 - x0, y1 - y0)
    side = min(side, height, width)
    x0 = max(0, min(width - side, center_x - side // 2))
    y0 = max(0, min(height - side, center_y - side // 2))
    return (x0, y0, x0 + side, y0 + side)


def _stage1_object_box(images: list[np.ndarray]) -> tuple[int, int, int, int]:
    stack = np.maximum.reduce([_normalise(image).astype(np.float32) for image in images])
    threshold = max(0.035, float(np.percentile(stack, 85.0)) * 0.25)
    return _square_bbox(stack > threshold, margin=34)


def _robust01(image: np.ndarray, limits: tuple[float, float] | None = None) -> np.ndarray:
    image = image.astype(np.float32)
    if limits is None:
        lo, hi = np.percentile(image, [0.5, 99.7])
    else:
        lo, hi = limits
    if hi <= lo:
        hi = float(image.max() if image.max() > lo else lo + 1.0)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def _stage1_limits(images: list[np.ndarray], obj_box: tuple[int, int, int, int]) -> tuple[float, float]:
    values = np.concatenate([_crop(image, obj_box).reshape(-1) for image in images])
    lo, hi = np.percentile(values, [0.5, 99.7])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if intersection == 0:
        return 0.0
    area_a = max(1, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1, (bx1 - bx0) * (by1 - by0))
    return intersection / float(area_a + area_b - intersection)


def _center_distance(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    return math.hypot((ax0 + ax1 - bx0 - bx1) * 0.5, (ay0 + ay1 - by0 - by1) * 0.5)


def _fallback_roi(
    first: tuple[int, int, int, int],
    obj_box: tuple[int, int, int, int],
    size: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = obj_box
    fx0, fy0, fx1, fy1 = first
    fc_x, fc_y = (fx0 + fx1) * 0.5, (fy0 + fy1) * 0.5
    target_x = x0 + int(0.72 * (x1 - x0)) if fc_x < (x0 + x1) * 0.5 else x0 + int(0.28 * (x1 - x0))
    target_y = y0 + int(0.72 * (y1 - y0)) if fc_y < (y0 + y1) * 0.5 else y0 + int(0.28 * (y1 - y0))
    half = size // 2
    rx0 = max(x0, min(x1 - size, target_x - half))
    ry0 = max(y0, min(y1 - size, target_y - half))
    return (rx0, ry0, rx0 + size, ry0 + size)


def _stage1_rois(
    left: np.ndarray,
    gt: np.ndarray,
    right: np.ndarray,
    obj_box: tuple[int, int, int, int],
    size: int = 122,
) -> list[tuple[int, int, int, int]]:
    x0, y0, x1, y1 = obj_box
    size = min(size, x1 - x0, y1 - y0)
    half = size // 2
    left_n, gt_n, right_n = (_normalise(image).astype(np.float32) for image in (left, gt, right))
    motion = np.abs(left_n - gt_n) + np.abs(right_n - gt_n)
    grad_x = cv2.Sobel(gt_n, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gt_n, cv2.CV_32F, 0, 1, ksize=3)
    detail = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    foreground = gt_n > max(0.04, float(np.percentile(_crop(gt_n, obj_box), 65.0)) * 0.35)

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for center_y in range(y0 + half, y1 - half + 1, 6):
        for center_x in range(x0 + half, x1 - half + 1, 6):
            box = (center_x - half, center_y - half, center_x - half + size, center_y - half + size)
            bx0, by0, bx1, by1 = box
            density = float(foreground[by0:by1, bx0:bx1].mean())
            if density < 0.08:
                continue
            score = float((motion[by0:by1, bx0:bx1] + 0.20 * detail[by0:by1, bx0:bx1]).sum()) * (0.6 + density)
            candidates.append((score, box))

    selected: list[tuple[int, int, int, int]] = []
    for _, box in sorted(candidates, key=lambda item: item[0], reverse=True):
        if all(_box_iou(box, previous) < 0.02 and _center_distance(box, previous) > size * 1.1 for previous in selected):
            selected.append(box)
            if len(selected) == 2:
                break

    if not selected:
        selected.append((x0 + (x1 - x0 - size) // 2, y0 + (y1 - y0 - size) // 2, x0 + (x1 - x0 - size) // 2 + size, y0 + (y1 - y0 - size) // 2 + size))
    if len(selected) == 1:
        selected.append(_fallback_roi(selected[0], obj_box, size))
    return selected[:2]


def _tile_roi_rects(
    obj_box: tuple[int, int, int, int],
    rois: list[tuple[int, int, int, int]],
    tile_size: int,
) -> list[tuple[int, int, int, int]]:
    ox0, oy0, _, _ = obj_box
    scale = tile_size / float(obj_box[2] - obj_box[0])
    return [
        (
            int((rx0 - ox0) * scale),
            int((ry0 - oy0) * scale),
            int((rx1 - ox0) * scale),
            int((ry1 - oy0) * scale),
        )
        for rx0, ry0, rx1, ry1 in rois
    ]


def _draw_stage1_roi_rects(
    draw: ImageDraw.ImageDraw,
    obj_box: tuple[int, int, int, int],
    rois: list[tuple[int, int, int, int]],
    tile_size: int,
    width: int,
) -> None:
    for rect, color in zip(_tile_roi_rects(obj_box, rois, tile_size), ["#D55E00", "#009E73"], strict=True):
        draw.rectangle(rect, outline=color, width=width)


def _draw_stage1_insets(
    full: Image.Image,
    image: np.ndarray,
    rois: list[tuple[int, int, int, int]],
    inset_size: int,
) -> None:
    draw = ImageDraw.Draw(full)
    tile_size = full.size[0]
    pad = 8
    positions = [(pad, pad), (tile_size - inset_size - pad, tile_size - inset_size - pad)]
    for roi, color, (x0, y0) in zip(rois, ["#D55E00", "#009E73"], positions, strict=True):
        inset = _as_pil_gray(_crop(image, roi)).resize((inset_size, inset_size), Image.Resampling.LANCZOS).convert("RGB")
        draw.rectangle([x0 - 5, y0 - 5, x0 + inset_size + 4, y0 + inset_size + 4], fill="white")
        full.paste(inset, (x0, y0))
        draw.rectangle([x0 - 2, y0 - 2, x0 + inset_size, y0 + inset_size], outline=color, width=5)


def _stage1_tile(
    image: np.ndarray,
    obj_box: tuple[int, int, int, int],
    rois: list[tuple[int, int, int, int]],
    tile_size: int,
    inset_size: int,
    with_insets: bool,
    highlight: bool = False,
) -> np.ndarray:
    full = _as_pil_gray(_crop(image, obj_box)).resize((tile_size, tile_size), Image.Resampling.LANCZOS).convert("RGB")
    draw = ImageDraw.Draw(full)
    _draw_stage1_roi_rects(draw, obj_box, rois, tile_size, width=5 if with_insets else 3)
    if with_insets:
        _draw_stage1_insets(full, image, rois, inset_size)
    if highlight:
        draw = ImageDraw.Draw(full)
        draw.rectangle([1, 1, tile_size - 2, tile_size - 2], outline="#D55E00", width=9)
    return np.asarray(full)


def build_stage1_figure(args: argparse.Namespace) -> Path | None:
    method_dirs = {method: args.output_dir / method / args.split for method in METHODS}
    if not all((method_dir / "pred_passive_npy").is_dir() for method_dir in method_dirs.values()):
        return None

    source_csv = method_dirs["amt"] / "passive_frame.csv"
    if not source_csv.is_file():
        source_csv = method_dirs["amt"] / "frame.csv"
    rows = load_frame_csv(source_csv)
    used = {
        ("beetle", 3),
        ("beetle", 4),
        ("beetle", 5),
        ("beetle", 6),
        ("beetle", 7),
        ("boombox", 5),
        ("teapot", 5),
        ("vintage_video_camera", 5),
    }
    passive_rows_by_method: dict[str, dict[tuple[str, int], dict[str, str]]] = {}
    for method, method_dir in method_dirs.items():
        csv_path = method_dir / "passive_frame.csv"
        if not csv_path.is_file():
            continue
        passive_rows_by_method[method] = {
            (row["scene"], int(row["target_id"])): row
            for row in load_frame_csv(csv_path)
        }

    def case_score(row: dict[str, str]) -> float:
        key = (row["scene"], int(row["target_id"]))
        values = [
            float(method_rows[key]["IE"])
            for method_rows in passive_rows_by_method.values()
            if key in method_rows
        ]
        return float(np.mean(values)) if values else float(row["IE"])

    row_by_key = {(row["scene"], int(row["target_id"])): row for row in rows}
    preferred_cases = [
        ("vintage_video_camera", 36),
        ("beetle", 92),
        ("boombox", 172),
    ]
    candidates = [row_by_key[key] for key in preferred_cases if key in row_by_key]
    seen_scenes: set[str] = set()
    for row in candidates:
        seen_scenes.add(row["scene"])
    for row in sorted(rows, key=case_score, reverse=True):
        if len(candidates) >= 3:
            break
        scene = row["scene"]
        target_id = int(row["target_id"])
        if (scene, target_id) in used:
            continue
        if any(scene == candidate["scene"] and target_id == int(candidate["target_id"]) for candidate in candidates):
            continue
        if scene in seen_scenes and len(seen_scenes) < 3:
            continue
        candidates.append(row)
        seen_scenes.add(scene)
    if len(candidates) < 3:
        candidates = sorted(rows, key=lambda item: float(item["IE"]), reverse=True)[:3]

    method_labels = {
        "liteflownet": "LiteFlowNet",
        "raft": "RAFT",
        "gma": "GMA",
        "amt": "AMT",
    }
    columns = [
        (r"$P_{t-1}$", "input_left", None),
        (r"$P_{t+1}$", "input_right", None),
        *[(method_labels[method], "pred", method) for method in METHODS],
        (r"GT $P_t$", "gt", None),
    ]

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 6.8,
            "axes.titlesize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
        }
    )
    tile_size = 520
    inset_size = 152
    spacer_after = 2
    width_ratios = [1.0] * spacer_after + [0.13] + [1.0] * (len(columns) - spacer_after)
    fig, axes = plt.subplots(
        len(candidates),
        len(columns) + 1,
        figsize=(7.05, 2.45),
        gridspec_kw={"width_ratios": width_ratios},
    )
    if len(candidates) == 1:
        axes = np.expand_dims(axes, 0)

    for row_index, row in enumerate(candidates):
        scene = row["scene"]
        target_id = int(row["target_id"])
        name = sample_name(target_id)
        left = _load_frame(args.data_root, scene, "passive", target_id - 1)
        gt = _load_frame(args.data_root, scene, "passive", target_id)
        right = _load_frame(args.data_root, scene, "passive", target_id + 1)
        predictions = {
            method: np.load(method_dirs[method] / "pred_passive_npy" / scene / f"{name}.npy").astype(np.float32)
            for method in METHODS
        }
        all_images = [left, right, *predictions.values(), gt]
        obj_box = _stage1_object_box(all_images)
        limits = _stage1_limits(all_images, obj_box)
        normalized = {
            "input_left": _robust01(left, limits),
            "input_right": _robust01(right, limits),
            "gt": _robust01(gt, limits),
            **{method: _robust01(prediction, limits) for method, prediction in predictions.items()},
        }
        rois = _stage1_rois(left, gt, right, obj_box)

        sep_ax = axes[row_index, spacer_after]
        sep_ax.set_xlim(0, 1)
        sep_ax.set_ylim(0, 1)
        sep_ax.axvline(0.5, ymin=0.0, ymax=1.0, color="#CFCFCF", linewidth=0.8)
        sep_ax.axis("off")

        for display_col, spec in enumerate(columns):
            title, kind, method = spec
            ax_col = display_col if display_col < spacer_after else display_col + 1
            ax = axes[row_index, ax_col]
            if kind == "input_left":
                tile = _stage1_tile(normalized["input_left"], obj_box, rois, tile_size, inset_size, with_insets=False)
            elif kind == "input_right":
                tile = _stage1_tile(normalized["input_right"], obj_box, rois, tile_size, inset_size, with_insets=False)
            elif kind == "gt":
                tile = _stage1_tile(normalized["gt"], obj_box, rois, tile_size, inset_size, with_insets=True)
            else:
                assert method is not None
                tile = _stage1_tile(normalized[method], obj_box, rois, tile_size, inset_size, with_insets=True)
            ax.imshow(tile, interpolation="none")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_index == 0:
                ax.set_title(title, pad=2.4, color="#222222", fontweight="normal")
            if display_col == 0:
                label = scene.replace("_", " ")
                if label == "vintage video camera":
                    label = "video camera"
                ax.set_ylabel(label, rotation=0, labelpad=18, va="center", fontsize=7.2)

    fig.subplots_adjust(left=0.064, right=0.998, top=0.84, bottom=0.015, wspace=0.026, hspace=0.035)

    args.appendix_figure_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.appendix_figure_dir / "fig_stage1_passive_interpolation.pdf"
    png_path = args.appendix_figure_dir / "fig_stage1_passive_interpolation.png"
    roi_pdf_path = args.appendix_figure_dir / "fig_stage1_passive_interpolation_roi.pdf"
    roi_png_path = args.appendix_figure_dir / "fig_stage1_passive_interpolation_roi.png"
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    fig.savefig(roi_pdf_path, dpi=600)
    fig.savefig(roi_png_path, dpi=600)
    plt.close(fig)

    cases_path = args.appendix_figure_dir / "fig_stage1_passive_interpolation_cases.json"
    cases_path.write_text(json.dumps(candidates, indent=2) + "\n", encoding="utf-8")
    return pdf_path


def write_summary(args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    merged: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        metrics_path = args.output_dir / method / args.split / "metrics.json"
        if metrics_path.is_file():
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            merged[method] = data
    for result in results:
        merged[result["method"]] = result
    rows = []
    for method in METHODS:
        if method not in merged:
            continue
        result = merged[method]
        row = {
            "method": result["method"],
            "frames": result["frames"],
            **result["overall"],
        }
        rows.append(row)
    fields = ["method", "frames", *MAIN_METRIC_KEYS, "Passive-IE"]
    write_csv(args.output_dir / "summary.csv", rows, fields)
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = torch_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    scenes = read_split(args.data_root / f"{args.split}.txt")
    methods = METHODS if args.method == "all" else (args.method,)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for method in methods:
        result = evaluate_method(args, method, scenes, device)
        results.append(result)
        write_summary(args, results)
        print(json.dumps({"completed": method, "overall": result["overall"]}, ensure_ascii=False), flush=True)

    if not args.no_figure and (args.method in {"all", "amt"}):
        figure_path = build_stage1_figure(args)
        if figure_path is not None:
            print(json.dumps({"figure": str(figure_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
