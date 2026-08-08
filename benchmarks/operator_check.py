"""Small reproducible overhead benchmark; numbers are never hard-coded in claims."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

import nablaguard as ng


def workload(x: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(x, dim=-1)


def timed(callback, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        callback()
        samples.append(time.perf_counter() - started)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()
    value = torch.randn(args.size, dtype=torch.float64, requires_grad=True)

    def baseline() -> None:
        local = value.detach().clone().requires_grad_(True)
        workload(local).sum().backward()

    def checked() -> None:
        ng.check.operator(candidate=workload, reference=workload, inputs=[value])

    baseline_seconds = statistics.median(timed(baseline, args.repeats))
    checked_seconds = statistics.median(timed(checked, args.repeats))
    print(f"baseline median: {baseline_seconds:.6f}s")
    print(f"checked median:  {checked_seconds:.6f}s")
    print(f"measured overhead: {checked_seconds / baseline_seconds:.2f}x")


if __name__ == "__main__":
    main()
