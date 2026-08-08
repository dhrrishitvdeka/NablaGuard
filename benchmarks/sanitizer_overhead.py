"""Measure eager sanitizer runtime and Python-heap overhead by mode."""

from __future__ import annotations

import argparse
import statistics
import time
import tracemalloc
from collections.abc import Callable

import torch

import nablaguard as ng


def workload(value: torch.Tensor, iterations: int) -> None:
    for _ in range(iterations):
        value = torch.logsumexp(value, dim=-1, keepdim=True).expand_as(value) / 8


def measure(callback: Callable[[], None], repeats: int) -> tuple[float, int]:
    samples = []
    peak = 0
    for _ in range(repeats):
        tracemalloc.start()
        started = time.perf_counter()
        callback()
        samples.append(time.perf_counter() - started)
        _, observed_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak = max(peak, observed_peak)
    return statistics.median(samples), peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    value = torch.randn(args.size, dtype=torch.float32)

    baseline, baseline_peak = measure(lambda: workload(value, args.iterations), args.repeats)
    print(f"baseline: {baseline:.6f}s, Python heap peak {baseline_peak} bytes")
    for mode in ("light", "standard", "deep"):

        def guarded(selected_mode=mode) -> None:
            with ng.guard(mode=selected_mode, capture_source=False):
                workload(value, args.iterations)

        elapsed, peak = measure(guarded, args.repeats)
        print(f"{mode}: {elapsed:.6f}s ({elapsed / baseline:.2f}x), Python heap peak {peak} bytes")


if __name__ == "__main__":
    main()
