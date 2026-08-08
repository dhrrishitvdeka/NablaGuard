"""Reproducible end-to-end benchmark matrix for NablaGuard release candidates."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

import nablaguard as ng
from nablaguard.bisect import first_bad


class WrongSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(value)
        return value.square()

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> torch.Tensor:
        (value,) = ctx.saved_tensors
        return gradient * value


def measure(callback: Callable[[], None], repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        callback()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def detection_metrics() -> dict[str, Any]:
    spec = [ng.tensor(shape=(16,), dtype=torch.float64)]
    correct = ng.check.operator(
        candidate=torch.sin,
        reference=torch.sin,
        inputs=spec,
        seed=11,
    ).passed
    wrong_backward = not ng.check.operator(
        candidate=WrongSquare.apply,
        reference=lambda value: value.square(),
        inputs=spec,
        seed=11,
    ).passed
    with ng.guard(mode="standard", capture_source=False) as safe_guard:
        torch.logsumexp(torch.tensor([1.0, 2.0]), dim=0)
    with ng.guard(mode="standard", capture_source=False) as broken_guard:
        torch.exp(torch.tensor([100.0], dtype=torch.float32))
    cases = {
        "correct_operator_passed": correct,
        "safe_sanitizer_had_no_issues": not safe_guard.issues,
        "wrong_backward_detected": wrong_backward,
        "overflow_risk_detected": any(
            issue.category == "OVERFLOW_RISK" for issue in broken_guard.issues
        ),
    }
    safe = (cases["correct_operator_passed"], cases["safe_sanitizer_had_no_issues"])
    broken = (cases["wrong_backward_detected"], cases["overflow_risk_detected"])
    return {
        "cases": cases,
        "controlled_false_positive_rate": 1 - sum(safe) / len(safe),
        "controlled_false_negative_rate": 1 - sum(broken) / len(broken),
        "scope": "controlled fixtures only; not a population effectiveness estimate",
    }


def overhead_metrics(repeats: int, iterations: int) -> dict[str, Any]:
    value = torch.randn(512, dtype=torch.float32)

    def workload() -> None:
        current = value
        for _ in range(iterations):
            current = torch.logsumexp(current, dim=0, keepdim=True).expand_as(current) / 8

    baseline = measure(workload, repeats)

    def guarded() -> None:
        with ng.guard(mode="standard", capture_source=False):
            workload()

    standard = measure(guarded, repeats)
    result: dict[str, Any] = {
        "baseline_median_seconds": baseline,
        "standard_median_seconds": standard,
        "standard_runtime_ratio": standard / baseline,
        "repeats": repeats,
        "iterations": iterations,
    }
    if torch.cuda.is_available():
        cuda_value = value.cuda()

        def cuda_workload() -> None:
            torch.logsumexp(cuda_value, dim=0)
            torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        cuda_workload()
        baseline_peak = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with ng.guard(mode="standard", capture_source=False):
            cuda_workload()
        guarded_peak = torch.cuda.max_memory_allocated()
        result["gpu_peak_memory_overhead_bytes"] = guarded_peak - baseline_peak
    else:
        result["gpu_peak_memory_overhead_bytes"] = None
        result["gpu_note"] = "CUDA unavailable; GPU memory overhead not measured"
    return result


def replay_metrics(steps: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        torch.manual_seed(91)
        model = torch.nn.Linear(4, 2, dtype=torch.float64)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        with ng.capture(
            model,
            optimizer,
            root=Path(directory),
            checkpoint_every=max(steps, 1),
        ) as recorder:
            for step in range(1, steps + 1):
                value = torch.randn(8, 4, dtype=torch.float64)
                loss = model(value).square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                recorder.record_step(
                    step=step,
                    tensors={"weight": model.weight, "bias": model.bias},
                )
        replay_model = torch.nn.Linear(4, 2, dtype=torch.float64)
        replay_optimizer = torch.optim.SGD(replay_model.parameters(), lr=0.01)

        def replay_step(step: int, metadata: dict[str, Any]) -> dict[str, torch.Tensor]:
            del step, metadata
            value = torch.randn(8, 4, dtype=torch.float64)
            loss = replay_model(value).square().mean()
            replay_optimizer.zero_grad()
            loss.backward()
            replay_optimizer.step()
            return {"weight": replay_model.weight, "bias": replay_model.bias}

        result = ng.replay(
            recorder.run_path,
            model=replay_model,
            optimizer=replay_optimizer,
            step_fn=replay_step,
            to_step=steps,
        )
        matches = sum(step.status == "MATCH" for step in result.steps)
        return {
            "matched_steps": matches,
            "total_steps": steps,
            "accuracy": matches / steps,
            "elapsed_seconds": result.elapsed_seconds,
        }


def search_metrics(steps: int) -> dict[str, Any]:
    boundary = max(1, steps * 3 // 5)
    result = first_bad(0, steps, lambda step: step >= boundary)
    evaluations = len(result.probes) + 2
    return {
        "expected_first_bad": boundary,
        "observed_first_bad": result.first_bad_step,
        "accurate": result.first_bad_step == boundary,
        "predicate_evaluations": evaluations,
        "ceiling_plus_endpoints": math.ceil(math.log2(steps)) + 2,
    }


def fuzz_metrics(runs: int, trials: int) -> dict[str, Any]:
    def candidate(value: torch.Tensor) -> torch.Tensor:
        return value + 1 if value.shape[-1] == 17 else value

    strategy = ng.tensor(
        shape=ng.shapes(ranks=(1,), dimensions=(16, 17, 32)),
        dtype=[torch.float64],
        distribution=["normal"],
        layout=["contiguous"],
    )
    discovered = 0
    minimized = 0
    for seed in range(runs):
        result = ng.check.fuzz(
            candidate=candidate,
            reference=lambda value: value,
            inputs=[strategy],
            trials=trials,
            seed=seed,
        )
        if result.failures:
            discovered += 1
            minimized += result.failures[0].minimal_specs[0].shape == (17,)
    return {
        "runs": runs,
        "trials_per_run": trials,
        "discovery_rate": discovered / runs,
        "known_minimum_recovery_rate": minimized / discovered if discovered else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repeats, iterations, steps, runs, trials = (
        (3, 3, 5, 3, 12) if args.quick else (10, 20, 20, 20, 50)
    )
    result = {
        "format_version": 1,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "detection": detection_metrics(),
        "overhead": overhead_metrics(repeats, iterations),
        "replay": replay_metrics(steps),
        "bisect": search_metrics(max(steps * 100, 2)),
        "fuzzing": fuzz_metrics(runs, trials),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
