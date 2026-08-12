"""Command-line entry point for NablaGuard artifacts and package metadata."""

from __future__ import annotations

import argparse
import importlib
import runpy
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch

from nablaguard import __version__
from nablaguard.artifact import (
    ArtifactPolicy,
    inspect_artifact,
    migrate_legacy_artifact,
    sanitize_artifact,
)
from nablaguard.benchmark import (
    BugBenchConfigError,
    OverheadConfigError,
    run_bugbench,
    run_overhead_benchmark,
)
from nablaguard.benchmark.bugbench import json_dumps as bugbench_json
from nablaguard.benchmark.overhead import json_dumps as overhead_json
from nablaguard.bisect import bisect as bisect_run
from nablaguard.bisect import metric_greater_than, metric_less_than, metric_nonfinite
from nablaguard.check import FuzzResult, OperatorCheckResult, fuzz, operator
from nablaguard.check.specs import TensorSpec, TensorStrategy, shapes, tensor
from nablaguard.core.serialization import atomic_write_text, dumps_json
from nablaguard.replay import replay
from nablaguard.report import dumps as json_report
from nablaguard.report import html as html_report
from nablaguard.report import junit as junit_report
from nablaguard.sanitize import guard

# Exit taxonomy: 0 success / pass, 1 check-fail, 2 usage, 3 internal/config error.
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2
EXIT_ERROR = 3


def build_parser() -> argparse.ArgumentParser:
    """Build the stable CLI parser."""

    parser = argparse.ArgumentParser(
        prog="nabla", description="Verification and debugging for differentiable programs."
    )
    parser.add_argument("--version", action="version", version=f"NablaGuard {__version__}")
    subcommands = parser.add_subparsers(dest="command")
    inspect_parser = subcommands.add_parser(
        "inspect",
        help="inspect a failure artifact (alias of `artifact inspect`)",
    )
    inspect_parser.add_argument("artifact", type=Path)
    inspect_parser.add_argument("--no-verify-hashes", action="store_true")
    artifact_parser = subcommands.add_parser("artifact", help="inspect and sanitize NGF artifacts")
    artifact_commands = artifact_parser.add_subparsers(dest="artifact_command", required=True)
    artifact_inspect = artifact_commands.add_parser("inspect", help="validate an NGF artifact")
    artifact_inspect.add_argument("artifact", type=Path)
    artifact_inspect.add_argument("--no-verify-hashes", action="store_true")
    artifact_sanitize = artifact_commands.add_parser(
        "sanitize", help="create a new metadata-only sanitized NGF artifact"
    )
    artifact_sanitize.add_argument("artifact", type=Path)
    artifact_sanitize.add_argument("--output-root", type=Path)
    artifact_migrate = artifact_commands.add_parser(
        "migrate", help="migrate legacy metadata without loading tensor pickle files"
    )
    artifact_migrate.add_argument("artifact", type=Path)
    artifact_migrate.add_argument("--output-root", type=Path, required=True)
    run_parser = subcommands.add_parser("run", help="run a Python script with optional monitoring")
    run_parser.add_argument("script", type=Path)
    run_parser.add_argument("--capture", action="store_true", help="capture a numerical report")
    run_parser.add_argument("--mode", choices=("light", "standard", "deep"), default="standard")
    _add_report_options(run_parser)
    trace_parser = subcommands.add_parser(
        "trace", help="run a Python script and capture sensitive eager tensor operations"
    )
    trace_parser.add_argument("script", type=Path)
    trace_parser.add_argument("--mode", choices=("standard", "deep"), default="standard")
    _add_report_options(trace_parser)
    check_parser = subcommands.add_parser("check", help="verify an importable PyTorch callable")
    check_parser.add_argument("candidate", help="candidate as module:qualified_name")
    check_parser.add_argument("--reference", help="reference as module:qualified_name")
    check_parser.add_argument("--shape", default="32", help="comma-separated dimensions")
    check_parser.add_argument(
        "--dtype", choices=("float64", "float32", "bfloat16", "float16"), default="float64"
    )
    check_parser.add_argument("--trials", type=int, default=1)
    check_parser.add_argument("--seed", type=int, default=81927183)
    check_parser.add_argument("--artifact-dir", type=Path)
    check_parser.add_argument("--artifact-raw-tensors", action="store_true")
    check_parser.add_argument("--artifact-max-size", default="500MB")
    check_parser.add_argument("--artifact-max-tensors", type=int, default=16)
    check_parser.add_argument("--random-vjp", action="store_true")
    check_parser.add_argument("--jvp", action="store_true")
    check_parser.add_argument("--double-backward", action="store_true")
    check_parser.add_argument("--finite-difference", action="store_true")
    check_parser.add_argument("--determinism", action="store_true")
    _add_report_options(check_parser)
    sanitize_parser = subcommands.add_parser(
        "sanitize", help="run a Python script under numerical instrumentation"
    )
    sanitize_parser.add_argument("script", type=Path)
    sanitize_parser.add_argument(
        "--mode", choices=("light", "standard", "deep"), default="standard"
    )
    sanitize_parser.add_argument("--shadow", action="store_true")
    _add_report_options(sanitize_parser)
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
    replay_parser.add_argument(
        "--i-trust-this-run",
        action="store_true",
        help="required: acknowledge that checkpoints are pickle and must be local/trusted",
    )
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
    benchmark_parser = subcommands.add_parser("benchmark", help="run reproducible benchmarks")
    benchmark_suites = benchmark_parser.add_subparsers(dest="benchmark_suite", required=True)
    bugbench_parser = benchmark_suites.add_parser(
        "bugbench", help="run the ground-truth ML bug benchmark"
    )
    bugbench_parser.add_argument("--root", type=Path, default=Path("benchmarks/bugbench"))
    bugbench_parser.add_argument("--seed", type=int, default=81927183)
    bugbench_parser.add_argument("--category", action="append", dest="categories")
    bugbench_parser.add_argument("--format", choices=("console", "json"), default="console")
    bugbench_parser.add_argument("--output", type=Path)
    overhead_parser = benchmark_suites.add_parser(
        "overhead", help="measure guard-mode overhead on representative workloads"
    )
    overhead_parser.add_argument("--quick", action="store_true")
    overhead_parser.add_argument("--device")
    overhead_parser.add_argument("--workload", action="append", dest="workloads")
    overhead_parser.add_argument("--format", choices=("console", "json"), default="console")
    overhead_parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the NablaGuard CLI."""

    parser = build_parser()
    try:
        arguments, remaining = parser.parse_known_args(argv)
    except SystemExit as error:
        code = error.code
        if code is None:
            return EXIT_OK
        return int(code) if int(code) in {EXIT_OK, EXIT_USAGE} else EXIT_USAGE
    if remaining and arguments.command not in {"run", "sanitize", "trace"}:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")
    if arguments.command == "inspect":
        inspection = inspect_artifact(
            arguments.artifact, verify_hashes=not arguments.no_verify_hashes
        )
        print(dumps_json(inspection.to_dict()))
        return EXIT_OK if inspection.valid else EXIT_FAIL
    if arguments.command == "artifact":
        if arguments.artifact_command == "inspect":
            inspection = inspect_artifact(
                arguments.artifact, verify_hashes=not arguments.no_verify_hashes
            )
            print(dumps_json(inspection.to_dict()))
            return EXIT_OK if inspection.valid else EXIT_FAIL
        if arguments.artifact_command == "sanitize":
            try:
                sanitized = sanitize_artifact(arguments.artifact, arguments.output_root)
            except (OSError, ValueError, TypeError) as error:
                print(f"Artifact sanitization failed: {error}", file=sys.stderr)
                return EXIT_ERROR
            print(sanitized)
            return EXIT_OK
        if arguments.artifact_command == "migrate":
            try:
                migrated = migrate_legacy_artifact(
                    arguments.artifact, arguments.output_root
                )
            except (OSError, ValueError, TypeError) as error:
                print(f"Artifact migration failed: {error}", file=sys.stderr)
                return EXIT_ERROR
            print(migrated)
            return EXIT_OK
    if arguments.command in {"run", "sanitize", "trace"}:
        if not arguments.script.is_file():
            parser.error(f"script does not exist: {arguments.script}")
        if arguments.command == "run" and not arguments.capture:
            _execute_script(arguments.script, remaining)
            return EXIT_OK
        monitor = guard(
            mode=arguments.mode,
            shadow=getattr(arguments, "shadow", False) or None,
        )
        with monitor:
            _execute_script(arguments.script, remaining)
        _emit_report(monitor, arguments.format, arguments.output, console=monitor.format())
        return EXIT_FAIL if monitor.issues else EXIT_OK
    if arguments.command == "check":
        candidate_path = Path(arguments.candidate)
        if arguments.reference is None and candidate_path.suffix == ".py":
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", str(candidate_path)],
                check=False,
            )
            return _map_pytest_exit(completed.returncode)
        if arguments.reference is None:
            parser.error("--reference is required for an importable callable")
        try:
            candidate = _load_callable(arguments.candidate)
            reference = _load_callable(arguments.reference)
            shape = _parse_shape(arguments.shape)
            dtype = getattr(torch, arguments.dtype)
            ArtifactPolicy.create(
                raw_tensors=arguments.artifact_raw_tensors,
                max_size=arguments.artifact_max_size,
                max_stored_tensors=arguments.artifact_max_tensors,
            )
        except (ValueError, TypeError, ModuleNotFoundError, AttributeError) as error:
            print(f"Invalid check configuration: {error}", file=sys.stderr)
            return EXIT_ERROR
        result: OperatorCheckResult | FuzzResult
        try:
            if arguments.trials == 1:
                result = operator(
                    candidate=candidate,
                    reference=reference,
                    inputs=[TensorSpec(shape, dtype)],
                    seed=arguments.seed,
                    vjp_cotangent="random" if arguments.random_vjp else "ones",
                    check_jvp=arguments.jvp,
                    check_double_backward=arguments.double_backward,
                    check_finite_difference=arguments.finite_difference,
                    check_determinism=arguments.determinism,
                    artifact_dir=arguments.artifact_dir,
                    artifact_raw_tensors=arguments.artifact_raw_tensors,
                    artifact_max_size=arguments.artifact_max_size,
                    artifact_max_tensors=arguments.artifact_max_tensors,
                )
            else:
                if any(
                    (
                        arguments.random_vjp,
                        arguments.jvp,
                        arguments.double_backward,
                        arguments.finite_difference,
                        arguments.determinism,
                    )
                ):
                    parser.error("advanced derivative flags currently require --trials 1")
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
                    artifact_raw_tensors=arguments.artifact_raw_tensors,
                    artifact_max_size=arguments.artifact_max_size,
                    artifact_max_tensors=arguments.artifact_max_tensors,
                )
        except ValueError as error:
            print(f"Invalid check configuration: {error}", file=sys.stderr)
            return EXIT_ERROR
        _emit_report(result, arguments.format, arguments.output, console=result.format())
        return EXIT_OK if result.passed else EXIT_FAIL
    if arguments.command == "replay":
        try:
            _require_trusted_run(arguments.run, trusted=arguments.i_trust_this_run)
            model_factory = _load_callable(arguments.model_factory)
            step_function = _load_callable(arguments.step_function)
            optimizer_factory = (
                _load_callable(arguments.optimizer_factory)
                if arguments.optimizer_factory
                else None
            )
            model = model_factory()
            if not isinstance(model, torch.nn.Module):
                raise TypeError("model factory must return torch.nn.Module")
            optimizer = optimizer_factory(model) if optimizer_factory else None
        except (ValueError, TypeError, ModuleNotFoundError, AttributeError, OSError) as error:
            print(f"Invalid replay configuration: {error}", file=sys.stderr)
            return EXIT_ERROR

        def invoke_step(step: int, metadata: dict[str, Any]) -> Mapping[str, torch.Tensor] | None:
            return cast(
                Mapping[str, torch.Tensor] | None,
                step_function(model, optimizer, step, metadata),
            )

        try:
            replay_result = replay(
                arguments.run,
                model=model,
                optimizer=optimizer,
                step_fn=invoke_step,
                from_step=arguments.from_step,
                to_step=arguments.to_step,
                stop_on_divergence=not arguments.continue_on_divergence,
            )
        except (ValueError, FileNotFoundError, RuntimeError) as error:
            print(f"Replay failed: {error}", file=sys.stderr)
            return EXIT_ERROR
        replay_result.print()
        return EXIT_OK if replay_result.passed else EXIT_FAIL
    if arguments.command == "bisect":
        if arguments.greater_than is not None:
            predicate = metric_greater_than(arguments.metric, arguments.greater_than)
        elif arguments.less_than is not None:
            predicate = metric_less_than(arguments.metric, arguments.less_than)
        else:
            predicate = metric_nonfinite(arguments.metric)
        try:
            bisect_result = bisect_run(
                arguments.run,
                predicate,
                known_good=arguments.known_good,
                known_bad=arguments.known_bad,
            )
        except (ValueError, FileNotFoundError, RuntimeError) as error:
            print(f"Bisect failed: {error}", file=sys.stderr)
            return EXIT_ERROR
        bisect_result.print()
        return EXIT_OK if bisect_result.passed else EXIT_FAIL
    if arguments.command == "benchmark" and arguments.benchmark_suite == "bugbench":
        try:
            benchmark_result = run_bugbench(
                arguments.root,
                seed=arguments.seed,
                categories=arguments.categories,
            )
        except BugBenchConfigError as error:
            print(f"Invalid BugBench configuration: {error}", file=sys.stderr)
            return EXIT_ERROR
        rendered = (
            bugbench_json(benchmark_result)
            if arguments.format == "json"
            else benchmark_result.format()
        )
        if arguments.output is None:
            print(rendered)
        else:
            atomic_write_text(arguments.output, rendered)
        if benchmark_result.exit_code == 0:
            return EXIT_OK
        if benchmark_result.metrics["internal_errors"]:
            return EXIT_ERROR
        return EXIT_FAIL
    if arguments.command == "benchmark" and arguments.benchmark_suite == "overhead":
        try:
            overhead_result = run_overhead_benchmark(
                quick=arguments.quick,
                device=arguments.device,
                workloads=arguments.workloads,
            )
        except OverheadConfigError as error:
            print(f"Invalid overhead benchmark configuration: {error}", file=sys.stderr)
            return EXIT_ERROR
        rendered = (
            overhead_json(overhead_result)
            if arguments.format == "json"
            else overhead_result.format()
        )
        if arguments.output is None:
            print(rendered)
        else:
            atomic_write_text(arguments.output, rendered)
        return EXIT_OK
    parser.print_help()
    return EXIT_OK


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


def _map_pytest_exit(code: int) -> int:
    """Map pytest exit codes into NablaGuard's 0/1/2/3 contract."""

    if code == 0:
        return EXIT_OK
    if code == 1:
        return EXIT_FAIL
    if code == 5:
        # pytest: no tests collected
        return EXIT_FAIL
    return EXIT_ERROR


def _require_trusted_run(run: Path, *, trusted: bool) -> None:
    """Refuse unacknowledged pickle-based checkpoint replay."""

    run_path = run.expanduser().resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"run path does not exist: {run_path}")
    if not trusted:
        raise ValueError(
            "refusing to load capture checkpoints without --i-trust-this-run; "
            "checkpoints are pickle-based and must not come from untrusted sources"
        )
    print(
        "WARNING: capture checkpoints use pickle (torch.load weights_only=False). "
        "Replay only trusted local runs produced by this process or an equivalent "
        "trusted capture. NGF inspect never loads .pt files.",
        file=sys.stderr,
    )


def _add_report_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=("console", "json", "html", "junit"),
        default="console",
        help="report encoding",
    )
    parser.add_argument("--output", type=Path, help="write the report to a file")


def _execute_script(script: Path, script_args: Sequence[str]) -> None:
    original_argv = sys.argv
    sys.argv = [str(script), *script_args]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = original_argv


def _emit_report(
    report: Any,
    selected_format: str,
    output: Path | None,
    *,
    console: str,
) -> None:
    if selected_format == "json":
        rendered = json_report(report)
    elif selected_format == "html":
        rendered = html_report(report)
    elif selected_format == "junit":
        rendered = junit_report(report)
    else:
        rendered = console
    if output is None:
        print(rendered)
        return
    atomic_write_text(output, rendered)


if __name__ == "__main__":
    raise SystemExit(main())
