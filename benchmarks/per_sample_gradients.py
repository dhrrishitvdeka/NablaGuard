"""Measure selected-parameter per-sample gradient cost and retained elements."""

from __future__ import annotations

import argparse

import torch

import nablaguard as ng


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--microbatch-size", type=int, default=8)
    args = parser.parse_args()
    model = torch.nn.Sequential(
        torch.nn.Linear(args.features, args.features),
        torch.nn.Tanh(),
        torch.nn.Linear(args.features, 1),
    )
    inputs = torch.randn(args.batch_size, args.features)
    report = ng.trace.samples(
        model,
        lambda output: output.squeeze(-1),
        inputs,
        layers=["2.*"],
        microbatch_size=args.microbatch_size,
    )
    print(f"selected parameter elements: {report.parameter_elements}")
    print(f"retained gradient elements: {report.gradient_elements}")
    print(f"measured seconds: {report.elapsed_seconds:.6f}")
    print(f"conflicting samples: {len(report.conflicting_samples)}")


if __name__ == "__main__":
    main()
