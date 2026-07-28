"""Summarize Stage-1 passive prediction metrics from saved artifacts.

This post-processing script does not run interpolation models. It reads the
``pred_passive_npy`` artifacts produced by ``evaluate_stage1_texture_error.py``
and reports metrics directly between predicted and ground-truth passive frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import _load_frame
from metrics import MAIN_METRIC_KEYS, compute_main_metrics


METHODS = ("liteflownet", "raft", "gma", "amt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final/appendix_stage1_texture_error"))
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def shared_minmax(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    minimum = min(float(first.min()), float(second.min()))
    maximum = max(float(first.max()), float(second.max()))
    if maximum <= minimum:
        maximum = minimum + 1.0
    return minimum, maximum


def normalize_with_range(image: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    return ((image.astype(np.float32) - minimum) / max(maximum - minimum, 1e-12)).clip(0.0, 1.0)


def tensor_pair(prediction: np.ndarray, target: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    minimum, maximum = shared_minmax(prediction, target)
    pred = normalize_with_range(prediction, minimum, maximum)
    gt = normalize_with_range(target, minimum, maximum)
    return (
        torch.from_numpy(pred.copy()).to(device=device).view(1, 1, *pred.shape),
        torch.from_numpy(gt.copy()).to(device=device).view(1, 1, *gt.shape),
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot average an empty metric group")
    return {key: float(np.mean([row[key] for row in rows])) for key in MAIN_METRIC_KEYS}


def sample_name(target_id: int) -> str:
    return f"{target_id - 1:03d}_{target_id:03d}_{target_id + 1:03d}"


def preserve_texture_summary(output_dir: Path) -> None:
    for suffix in ("csv", "json"):
        source = output_dir / f"summary.{suffix}"
        target = output_dir / f"summary_texture.{suffix}"
        if source.is_file() and not target.is_file():
            shutil.copy2(source, target)


@torch.no_grad()
def summarize_method(args: argparse.Namespace, method: str, device: torch.device) -> dict[str, Any]:
    method_dir = args.output_dir / method / args.split
    source_rows = read_csv(method_dir / "frame.csv")
    frame_rows: list[dict[str, Any]] = []
    scene_rows: dict[str, list[dict[str, float]]] = defaultdict(list)

    for row in source_rows:
        scene = row["scene"]
        target_id = int(row["target_id"])
        name = sample_name(target_id)
        prediction = np.load(method_dir / "pred_passive_npy" / scene / f"{name}.npy").astype(np.float32)
        target = _load_frame(args.data_root, scene, "passive", target_id)
        pred_tensor, target_tensor = tensor_pair(prediction, target, device)
        metrics = {key: float(value[0].detach().cpu()) for key, value in compute_main_metrics(pred_tensor, target_tensor).items()}
        out_row = {
            "scene": scene,
            "left_id": int(row["left_id"]),
            "target_id": target_id,
            "right_id": int(row["right_id"]),
            **metrics,
            "pred_passive_path": row["pred_passive_path"],
            "err_passive_path": row["err_passive_path"],
        }
        frame_rows.append(out_row)
        scene_rows[scene].append(metrics)

    overall = mean_rows([{key: float(row[key]) for key in MAIN_METRIC_KEYS} for row in frame_rows])
    by_scene = [
        {"scene": scene, "frames": len(rows), **mean_rows(rows)}
        for scene, rows in sorted(scene_rows.items())
    ]
    write_csv(
        method_dir / "passive_frame.csv",
        frame_rows,
        ["scene", "left_id", "target_id", "right_id", *MAIN_METRIC_KEYS, "pred_passive_path", "err_passive_path"],
    )
    write_csv(method_dir / "passive_scene.csv", by_scene, ["scene", "frames", *MAIN_METRIC_KEYS])

    metrics_path = method_dir / "metrics.json"
    metrics_json = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_json["passive_overall"] = overall
    metrics_json["passive_by_scene"] = by_scene
    metrics_json["passive_metric_normalization"] = "shared min-max normalization over predicted and ground-truth passive frames"
    metrics_path.write_text(json.dumps(metrics_json, indent=2) + "\n", encoding="utf-8")

    manifest_path = method_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["passive_metric_normalization"] = metrics_json["passive_metric_normalization"]
        manifest.setdefault("artifacts", {})
        manifest["artifacts"]["passive_frame_csv"] = "passive_frame.csv"
        manifest["artifacts"]["passive_scene_csv"] = "passive_scene.csv"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {"method": method, "frames": len(frame_rows), **overall}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    preserve_texture_summary(args.output_dir)
    summary = [summarize_method(args, method, device) for method in METHODS]
    fields = ["method", "frames", *MAIN_METRIC_KEYS]
    write_csv(args.output_dir / "summary_passive.csv", summary, fields)
    (args.output_dir / "summary_passive.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "summary.csv", summary, fields)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
