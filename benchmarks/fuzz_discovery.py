"""Measure discovery and shrinking cost for a controlled boundary bug."""

from __future__ import annotations

import argparse
import statistics

import torch

import nablaguard as ng


def candidate(x: torch.Tensor) -> torch.Tensor:
    return x + 1 if x.shape[-1] == 17 else x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    strategy = ng.tensor(
        shape=ng.shapes(ranks=(1, 2), dimensions=(7, 8, 16, 17, 31, 32)),
        dtype=[torch.float64],
        distribution=["normal"],
        layout=["contiguous"],
    )
    discovery_trials = []
    elapsed = []
    for seed in range(args.runs):
        result = ng.check.fuzz(
            candidate=candidate,
            reference=lambda x: x,
            inputs=[strategy],
            trials=args.trials,
            seed=seed,
        )
        if result.failures:
            discovery_trials.append(result.failures[0].trial + 1)
        elapsed.append(result.elapsed_seconds)
    print(f"discovery rate: {len(discovery_trials) / args.runs:.1%}")
    if discovery_trials:
        print(f"median trials to discovery: {statistics.median(discovery_trials):.1f}")
    print(f"median run seconds: {statistics.median(elapsed):.6f}")


if __name__ == "__main__":
    main()
