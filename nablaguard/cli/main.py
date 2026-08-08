"""Command-line entry point for NablaGuard artifacts and package metadata."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from nablaguard import __version__
from nablaguard.check import FuzzResult, OperatorCheckResult, fuzz, operator
from nablaguard.check.specs import TensorSpec, TensorStrategy, shapes, tensor


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
