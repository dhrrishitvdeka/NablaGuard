"""Command-line entry point for NablaGuard artifacts and package metadata."""

from __future__ import annotations

import argparse
import importlib
import json
import runpy
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch

from nablaguard import __version__
from nablaguard.bisect import bisect as bisect_run
from nablaguard.bisect import metric_greater_than, metric_less_than, metric_nonfinite
from nablaguard.check import FuzzResult, OperatorCheckResult, fuzz, operator
from nablaguard.check.specs import TensorSpec, TensorStrategy, shapes, tensor
from nablaguard.replay import replay
from nablaguard.sanitize import guard


def build_parser() -> argparse.ArgumentParser:
    """Build the stable CLI parser."""

    parser = argparse.ArgumentParser(
        prog="nabla", description="Verification and debugging for differentiable programs."
    )
    parser.add_argument("--version", action="version", version=f"NablaGuard {__version__}")
    subcommands = parser.add_subparsers(dest="command")
    inspect_parser = subcommands.add_parser("inspect", help="inspect a failure artifact")
    inspect_parser.add_argument("artifact", type=Path)
    check_parser = subcommands.add_parser("check", help="verify an importable PyTorch callable")
    check_parser.add_argument("candidate", help="candidate as module:qualified_name")
    check_parser.add_argument(
        "--reference", required=True, help="reference as module:qualified_name"
    )
    check_parser.add_argument("--shape", default="32", help="comma-separated dimensions")
    check_parser.add_argument(
        "--dtype", choices=("float64", "float32", "bfloat16", "float16"), default="float64"
    )
    check_parser.add_argument("--trials", type=int, default=1)
    check_parser.add_argument("--seed", type=int, default=81927183)
    check_parser.add_argument("--artifact-dir", type=Path)
    sanitize_parser = subcommands.add_parser(
        "sanitize", help="run a Python script under numerical instrumentation"
    )
    sanitize_parser.add_argument("script", type=Path)
    sanitize_parser.add_argument(
        "--mode", choices=("light", "standard", "deep"), default="standard"
    )
    sanitize_parser.add_argument("--shadow", action="store_true")
    sanitize_parser.add_argument("script_args", nargs=argparse.REMAINDER)
    replay_parser = subcommands.add_parser(
        "replay", help="restore and verify a captured training run"
    )
    replay_parser.add_argument("run", type=Path)
    replay_parser.add_argument("--model-factory", required=True)
    replay_parser.add_argument("--step-function", required=True)
    replay_parser.add_argument("--optimizer-factory")
    replay_parser.add_argument("--from-step", type=int, default=0)
    replay_parser.add_argument("--to-step", type=int)
    replay_parser.add_argument("--continue-on-divergence", action="store_true")
    bisect_parser = subcommands.add_parser(
        "bisect", help="locate a monotonic failure transition in captured metadata"
    )
    bisect_parser.add_argument("run", type=Path)
    bisect_parser.add_argument("--metric", required=True)
    bisect_thresholds = bisect_parser.add_mutually_exclusive_group(required=True)
    bisect_thresholds.add_argument("--greater-than", type=float)
    bisect_thresholds.add_argument("--less-than", type=float)
    bisect_thresholds.add_argument("--nonfinite", action="store_true")
    bisect_parser.add_argument("--known-good", type=int, default=0)
    bisect_parser.add_argument("--known-bad", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the NablaGuard CLI."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "inspect":
        metadata_path = arguments.artifact / "metadata.json"
        if not metadata_path.is_file():
            parser.error(f"not a NablaGuard artifact: {arguments.artifact}")
        print(json.dumps(json.loads(metadata_path.read_text(encoding="utf-8")), indent=2))
        return 0
    if arguments.command == "check":
        candidate = _load_callable(arguments.candidate)
        reference = _load_callable(arguments.reference)
        shape = _parse_shape(arguments.shape)
        dtype = getattr(torch, arguments.dtype)
        result: OperatorCheckResult | FuzzResult
        if arguments.trials == 1:
            result = operator(
                candidate=candidate,
                reference=reference,
                inputs=[TensorSpec(shape, dtype)],
                seed=arguments.seed,
                artifact_dir=arguments.artifact_dir,
            )
        else:
            strategy = tensor(
                shape=shapes(shape),
                dtype=[dtype],
                distribution=["normal", "zeros", "ones", "mixed_magnitude"],
                layout=["contiguous", "transposed", "strided"],
            )
            assert isinstance(strategy, TensorStrategy)
            result = fuzz(
                candidate=candidate,
                reference=reference,
                inputs=[strategy],
                trials=arguments.trials,
                seed=arguments.seed,
                artifact_dir=arguments.artifact_dir,
            )
        result.print()
        return 0 if result.passed else 1
    if arguments.command == "sanitize":
        if not arguments.script.is_file():
            parser.error(f"script does not exist: {arguments.script}")
        original_argv = sys.argv
        sys.argv = [str(arguments.script), *arguments.script_args]
        monitor = guard(mode=arguments.mode, shadow=arguments.shadow or None)
        try:
            with monitor:
                runpy.run_path(str(arguments.script), run_name="__main__")
        finally:
            sys.argv = original_argv
        monitor.print()
        return 1 if monitor.issues else 0
    if arguments.command == "replay":
        model_factory = _load_callable(arguments.model_factory)
        step_function = _load_callable(arguments.step_function)
        optimizer_factory = (
            _load_callable(arguments.optimizer_factory) if arguments.optimizer_factory else None
        )
        model = model_factory()
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model factory must return torch.nn.Module")
        optimizer = optimizer_factory(model) if optimizer_factory else None

        def invoke_step(step: int, metadata: dict[str, Any]) -> Mapping[str, torch.Tensor] | None:
            return cast(
                Mapping[str, torch.Tensor] | None,
                step_function(model, optimizer, step, metadata),
            )

        replay_result = replay(
            arguments.run,
            model=model,
            optimizer=optimizer,
            step_fn=invoke_step,
            from_step=arguments.from_step,
            to_step=arguments.to_step,
            stop_on_divergence=not arguments.continue_on_divergence,
        )
        replay_result.print()
        return 0 if replay_result.passed else 1
    if arguments.command == "bisect":
        if arguments.greater_than is not None:
            predicate = metric_greater_than(arguments.metric, arguments.greater_than)
        elif arguments.less_than is not None:
            predicate = metric_less_than(arguments.metric, arguments.less_than)
        else:
            predicate = metric_nonfinite(arguments.metric)
        bisect_result = bisect_run(
            arguments.run,
            predicate,
            known_good=arguments.known_good,
            known_bad=arguments.known_bad,
        )
        bisect_result.print()
        return 0
    parser.print_help()
    return 0


def _load_callable(target: str) -> Any:
    if ":" not in target:
        raise ValueError("callable targets must use module:qualified_name")
    module_name, path = target.split(":", 1)
    value: Any = importlib.import_module(module_name)
    for component in path.split("."):
        value = getattr(value, component)
    if not callable(value):
        raise TypeError(f"{target!r} is not callable")
    return value


def _parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(dimension) for dimension in value.split(",") if dimension)
    except ValueError as error:
        raise ValueError("shape dimensions must be integers") from error
    if any(dimension < 0 for dimension in shape):
        raise ValueError("shape dimensions cannot be negative")
    return shape


if __name__ == "__main__":
    raise SystemExit(main())
