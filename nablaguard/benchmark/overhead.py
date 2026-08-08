"""Representative runtime and memory overhead measurements by guard mode."""

from __future__ import annotations

import copy
import json
import platform
import statistics
import time
import tracemalloc
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

import torch

from nablaguard.sanitize import guard

BenchmarkMode = Literal["light", "standard", "deep"]
WORKLOAD_NAMES = (
    "tiny_mlp",
    "cnn",
    "transformer",
    "mixed_precision",
    "large_synthetic",
)


class OverheadConfigError(ValueError):
    """Raised when an overhead benchmark selection is invalid."""


@dataclass(frozen=True, slots=True)
class _Measurement:
    wall_seconds: float
    cpu_seconds: float
    python_heap_peak_bytes: int
    gpu_peak_bytes: int | None


@dataclass(frozen=True, slots=True)
class OverheadReport:
    """Measured overhead matrix; values are never performance guarantees."""

    device: str
    repeats: int
    warmups: int
    workloads: dict[str, dict[str, Any]]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": {"name": "nablaguard.overhead", "version": 1},
            "device": self.device,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "repeats": self.repeats,
            "warmups": self.warmups,
            "elapsed_seconds": self.elapsed_seconds,
            "workloads": self.workloads,
            "gpu_utilization_change": None,
            "gpu_utilization_note": (
                "GPU utilization requires an external device sampler and was not inferred."
            ),
            "disk_io_bytes": 0,
            "disk_io_note": "Observation-only guard modes do not persist artifacts.",
        }

    def format(self) -> str:
        lines = [
            "NablaGuard overhead benchmark",
            "=" * 40,
            f"Device: {self.device}",
            f"Repeats: {self.repeats} after {self.warmups} warmup(s)",
            "",
            "Workload                 Light       Standard    Deep",
            "-" * 62,
        ]
        for name, values in self.workloads.items():
            ratios = [
                values.get(mode, {}).get("wall_clock_ratio")
                for mode in ("light", "standard", "deep")
            ]
            rendered = ["N/A" if value is None else f"{value:.3f}x" for value in ratios]
            lines.append(
                f"{name:<25}{rendered[0]:<12}{rendered[1]:<12}{rendered[2]:<12}"
            )
        lines.extend(
            [
                "",
                "Ratios are measurements for this machine and workload, not guarantees.",
            ]
        )
        return "\n".join(lines)


def run_overhead_benchmark(
    *,
    quick: bool = False,
    device: str | torch.device | None = None,
    workloads: Iterable[str] | None = None,
    modes: Iterable[BenchmarkMode] = ("light", "standard", "deep"),
) -> OverheadReport:
    """Measure representative training-step overhead with persistent guards."""

    started = time.perf_counter()
    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise OverheadConfigError("CUDA was requested but is unavailable")
    selected_workloads = tuple(workloads or WORKLOAD_NAMES)
    unknown_workloads = set(selected_workloads) - set(WORKLOAD_NAMES)
    if unknown_workloads:
        raise OverheadConfigError(f"unknown overhead workloads: {sorted(unknown_workloads)}")
    selected_modes = tuple(modes)
    unknown_modes = set(selected_modes) - {"light", "standard", "deep"}
    if unknown_modes:
        raise OverheadConfigError(f"unknown guard modes: {sorted(unknown_modes)}")
    repeats = 2 if quick else 7
    warmups = 1 if quick else 3
    results: dict[str, dict[str, Any]] = {}
    for workload_name in selected_workloads:
        model, step = _build_workload(workload_name, selected_device)
        baseline = _measure(step, repeats=repeats, warmups=warmups, device=selected_device)
        workload_result: dict[str, Any] = {"baseline": _measurement_dict(baseline)}
        for mode in selected_modes:
            mode_model = copy.deepcopy(model)
            mode_step = _step_for_model(workload_name, mode_model, selected_device)
            with guard(mode_model, mode=mode, capture_source=False):
                measured = _measure(
                    mode_step, repeats=repeats, warmups=warmups, device=selected_device
                )
            workload_result[mode] = {
                **_measurement_dict(measured),
                "wall_clock_ratio": _ratio(measured.wall_seconds, baseline.wall_seconds),
                "cpu_time_ratio": _ratio(measured.cpu_seconds, baseline.cpu_seconds),
                "python_heap_overhead_bytes": max(
                    0, measured.python_heap_peak_bytes - baseline.python_heap_peak_bytes
                ),
                "gpu_memory_overhead_bytes": (
                    None
                    if measured.gpu_peak_bytes is None or baseline.gpu_peak_bytes is None
                    else max(0, measured.gpu_peak_bytes - baseline.gpu_peak_bytes)
                ),
            }
        results[workload_name] = workload_result
    return OverheadReport(
        device=str(selected_device),
        repeats=repeats,
        warmups=warmups,
        workloads=results,
        elapsed_seconds=time.perf_counter() - started,
    )


def json_dumps(report: OverheadReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False)


def _measure(
    callback: Callable[[], None], *, repeats: int, warmups: int, device: torch.device
) -> _Measurement:
    for _ in range(warmups):
        callback()
    wall_samples: list[float] = []
    cpu_samples: list[float] = []
    heap_peaks: list[int] = []
    gpu_peaks: list[int] = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            gpu_start = torch.cuda.memory_allocated(device)
        else:
            gpu_start = None
        tracemalloc.start()
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        callback()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        cpu_samples.append(time.process_time() - cpu_started)
        wall_samples.append(time.perf_counter() - wall_started)
        _, heap_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        heap_peaks.append(heap_peak)
        if gpu_start is not None:
            gpu_peaks.append(max(0, torch.cuda.max_memory_allocated(device) - gpu_start))
    return _Measurement(
        wall_seconds=statistics.median(wall_samples),
        cpu_seconds=statistics.median(cpu_samples),
        python_heap_peak_bytes=max(heap_peaks),
        gpu_peak_bytes=max(gpu_peaks) if gpu_peaks else None,
    )


def _build_workload(
    name: str, device: torch.device
) -> tuple[torch.nn.Module, Callable[[], None]]:
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(7123)
        if name == "tiny_mlp":
            model = torch.nn.Sequential(
                torch.nn.Linear(32, 64), torch.nn.GELU(), torch.nn.Linear(64, 16)
            ).to(device)
        elif name == "cnn":
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 16, 3, padding=1),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(16, 8),
            ).to(device)
        elif name == "transformer":
            model = torch.nn.TransformerEncoderLayer(
                d_model=32,
                nhead=4,
                dim_feedforward=64,
                dropout=0.0,
                batch_first=True,
            ).to(device)
        elif name == "mixed_precision":
            model = torch.nn.Sequential(
                torch.nn.Linear(64, 128), torch.nn.ReLU(), torch.nn.Linear(128, 32)
            ).to(device)
        elif name == "large_synthetic":
            layers: list[torch.nn.Module] = []
            for _ in range(8):
                layers.extend((torch.nn.Linear(256, 256), torch.nn.ReLU()))
            model = torch.nn.Sequential(*layers).to(device)
        else:
            raise OverheadConfigError(f"unknown overhead workload: {name}")
    return model, _step_for_model(name, model, device)


def _step_for_model(
    name: str, model: torch.nn.Module, device: torch.device
) -> Callable[[], None]:
    generator = torch.Generator(device=device).manual_seed(9181)
    if name == "tiny_mlp":
        inputs = torch.randn(32, 32, generator=generator, device=device)
    elif name == "cnn":
        inputs = torch.randn(8, 3, 16, 16, generator=generator, device=device)
    elif name == "transformer":
        inputs = torch.randn(4, 16, 32, generator=generator, device=device)
    elif name == "mixed_precision":
        inputs = torch.randn(32, 64, generator=generator, device=device)
    elif name == "large_synthetic":
        inputs = torch.randn(16, 256, generator=generator, device=device)
    else:
        raise OverheadConfigError(f"unknown overhead workload: {name}")

    def step() -> None:
        model.zero_grad(set_to_none=True)
        if name == "mixed_precision":
            autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                loss = model(inputs).float().square().mean()
        else:
            loss = model(inputs).square().mean()
        loss.backward()

    return step


def _measurement_dict(value: _Measurement) -> dict[str, Any]:
    return {
        "wall_seconds": value.wall_seconds,
        "cpu_seconds": value.cpu_seconds,
        "python_heap_peak_bytes": value.python_heap_peak_bytes,
        "gpu_peak_bytes": value.gpu_peak_bytes,
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None
