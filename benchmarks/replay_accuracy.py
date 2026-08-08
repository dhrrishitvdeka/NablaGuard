"""Measure exact replay accuracy and runtime on a deterministic CPU fixture."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

import nablaguard as ng


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        torch.manual_seed(91)
        model = torch.nn.Linear(8, 4, dtype=torch.float64)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with ng.capture(
            model,
            optimizer,
            root=Path(directory),
            checkpoint_every=max(args.steps // 5, 1),
        ) as recorder:
            for step in range(1, args.steps + 1):
                value = torch.randn(16, 8, dtype=torch.float64)
                loss = model(value).square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                recorder.record_step(
                    step=step,
                    tensors={"weight": model.weight, "bias": model.bias},
                )

        replay_model = torch.nn.Linear(8, 4, dtype=torch.float64)
        replay_optimizer = torch.optim.SGD(replay_model.parameters(), lr=0.01)

        def step_fn(step: int, metadata: dict):
            del step, metadata
            value = torch.randn(16, 8, dtype=torch.float64)
            loss = replay_model(value).square().mean()
            replay_optimizer.zero_grad()
            loss.backward()
            replay_optimizer.step()
            return {"weight": replay_model.weight, "bias": replay_model.bias}

        result = ng.replay(
            recorder.run_path,
            model=replay_model,
            optimizer=replay_optimizer,
            step_fn=step_fn,
            to_step=args.steps,
        )
        matched = sum(step.status == "MATCH" for step in result.steps)
        print(f"exact fingerprint matches: {matched}/{args.steps}")
        print(f"replay seconds: {result.elapsed_seconds:.6f}")
        print(f"first divergence: {result.first_divergence}")


if __name__ == "__main__":
    main()
