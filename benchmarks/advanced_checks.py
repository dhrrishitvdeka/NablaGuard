"""Measure incremental cost of opt-in derivative verification layers."""

from __future__ import annotations

import argparse
import statistics
import time

import torch

import nablaguard as ng


def candidate(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x) * x.square()


def measure(repeats: int, **options) -> float:
    samples = []
    for _ in range(repeats + 1):
        started = time.perf_counter()
        result = ng.check.operator(
            candidate=candidate,
            reference=candidate,
            inputs=[ng.tensor(shape=(16,), dtype=torch.float64)],
            **options,
        )
        assert result.passed
        samples.append(time.perf_counter() - started)
    return statistics.median(samples[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    configurations = {
        "forward+VJP": {},
        "+random VJP": {"vjp_cotangent": "random"},
        "+JVP": {"check_jvp": True},
        "+double backward": {"check_double_backward": True},
        "+finite difference": {"check_finite_difference": True},
        "+determinism": {"check_determinism": True},
    }
    for name, options in configurations.items():
        print(f"{name}: {measure(args.repeats, **options):.6f}s")


if __name__ == "__main__":
    main()
