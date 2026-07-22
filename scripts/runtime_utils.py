from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import torch


class InferenceTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.milliseconds: list[float] = []

    def measure(self, fn: Callable[[], Any]) -> Any:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            torch.cuda.synchronize(self.device)
            self.milliseconds.append(float(start.elapsed_time(end)))
            return result

        start_time = time.perf_counter()
        result = fn()
        self.milliseconds.append((time.perf_counter() - start_time) * 1000.0)
        return result

    def summary(
        self,
        method: str,
        frames: int,
        warmup_iters: int,
        *,
        params_m: float | None = None,
        flops_t: float | None = None,
        flops_error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.milliseconds:
            raise RuntimeError("No timed inference calls were recorded.")
        latency_ms_per_frame = float(sum(self.milliseconds)) / max(1, int(frames))
        summary: dict[str, Any] = {
            "method": method,
            "frames": int(frames),
            "warmup_iters": int(warmup_iters),
            "device": str(self.device),
            "Latency(ms/f)": round(float(latency_ms_per_frame), 4),
            "Params(M)": round(float(params_m), 4) if params_m is not None else None,
            "FLOPs(T)": round(float(flops_t), 4) if flops_t is not None else None,
            "excludes_io_and_metrics": True,
        }
        if flops_error:
            summary["flops_error"] = flops_error
        if extra:
            summary.update(extra)
        return summary


def cuda_warmup(device: torch.device, warmup_iters: int, fn: Callable[[], Any]) -> None:
    for _ in range(max(0, warmup_iters)):
        _ = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_parameters_m(module: Any) -> float | None:
    if module is None or not hasattr(module, "parameters"):
        return None
    return float(sum(parameter.numel() for parameter in module.parameters()) / 1e6)


def profile_flops_t(forward_fn: Callable[[], Any], device: torch.device) -> tuple[float | None, str | None]:
    try:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
            torch.cuda.synchronize(device)
        with torch.inference_mode():
            with profile(activities=activities, with_flops=True) as prof:
                _ = forward_fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        flops = float(sum((getattr(event, "flops", 0) or 0) for event in prof.key_averages()))
        if flops <= 0:
            return None, "torch.profiler reported zero FLOPs for the representative forward pass"
        return flops / 1e12, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def efficiency_from_runtime(runtime: dict[str, Any] | None) -> dict[str, Any]:
    runtime = runtime or {}
    return {
        "Latency(ms/f)": runtime.get("Latency(ms/f)"),
        "Params(M)": runtime.get("Params(M)"),
        "FLOPs(T)": runtime.get("FLOPs(T)"),
    }


def make_table_row(
    method: str,
    frames: int,
    mean_metrics: dict[str, Any],
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "Method": method,
        "Frames": int(frames),
        "PSNR": mean_metrics.get("psnr"),
        "SSIM": mean_metrics.get("ssim"),
        "MAE": mean_metrics.get("mae"),
        "LPIPS": mean_metrics.get("lpips"),
    }
    row.update(efficiency_from_runtime(runtime))
    return row


def write_runtime(path: Path, runtime: dict[str, Any]) -> None:
    import json

    path.write_text(json.dumps(runtime, indent=2) + "\n")
