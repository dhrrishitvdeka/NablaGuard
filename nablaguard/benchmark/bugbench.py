"""Ground-truth-driven NablaGuard BugBench execution and metrics."""

from __future__ import annotations

import importlib
import json
import platform
import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import torch


class BugBenchConfigError(ValueError):
    """Raised when BugBench ground truth is missing or malformed."""


@dataclass(frozen=True, slots=True)
class ExpectedOutcome:
    """Machine-readable expected detector outcome for one fixture."""

    detected: bool
    category: str | None
    module: str | None
    stage: str | None


@dataclass(frozen=True, slots=True)
class BugBenchGroundTruth:
    """Versioned ground truth loaded from a fixture ``.yaml`` file.

    BugBench v1 files use JSON syntax, which is a strict subset of YAML 1.2. This
    keeps the benchmark dependency-free while remaining readable by YAML tools.
    """

    identifier: str
    benchmark_category: str
    description: str
    implementation: str
    expected: ExpectedOutcome
    tags: tuple[str, ...]
    requirements: Mapping[str, Any]
    path: Path

    @classmethod
    def load(cls, path: Path) -> BugBenchGroundTruth:
        """Load and strictly validate one BugBench v1 ground-truth record."""

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BugBenchConfigError(
                f"cannot read BugBench ground truth {path}: {error}"
            ) from error
        if not isinstance(raw, dict):
            raise BugBenchConfigError(f"BugBench ground truth must be an object: {path}")
        schema = raw.get("schema")
        if schema != {"name": "nablaguard.bugbench.case", "version": 1}:
            raise BugBenchConfigError(f"unsupported BugBench schema in {path}")
        expected = raw.get("expected")
        if not isinstance(expected, dict) or not isinstance(expected.get("detected"), bool):
            raise BugBenchConfigError(f"expected.detected must be Boolean in {path}")
        identifier = _required_string(raw, "id", path)
        benchmark_category = _required_string(raw, "benchmark_category", path)
        if path.parent.name != benchmark_category:
            raise BugBenchConfigError(
                f"benchmark_category {benchmark_category!r} does not match directory "
                f"{path.parent.name!r}"
            )
        tags = raw.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(value, str) for value in tags):
            raise BugBenchConfigError(f"tags must be a string list in {path}")
        requirements = raw.get("requirements", {})
        if not isinstance(requirements, dict):
            raise BugBenchConfigError(f"requirements must be an object in {path}")
        outcome = ExpectedOutcome(
            detected=expected["detected"],
            category=_optional_string(expected, "category", path),
            module=_optional_string(expected, "module", path),
            stage=_optional_string(expected, "stage", path),
        )
        if outcome.detected and outcome.category is None:
            raise BugBenchConfigError(f"bug fixtures require expected.category in {path}")
        return cls(
            identifier=identifier,
            benchmark_category=benchmark_category,
            description=_required_string(raw, "description", path),
            implementation=_required_string(raw, "implementation", path),
            expected=outcome,
            tags=tuple(tags),
            requirements=requirements,
            path=path,
        )


@dataclass(frozen=True, slots=True)
class CaseContext:
    """Stable execution context passed to BugBench fixture implementations."""

    identifier: str
    seed: int
    fixture_path: Path


@dataclass(frozen=True, slots=True)
class BugBenchObservation:
    """Detector output and optional measured evidence from one fixture."""

    detected: bool
    category: str | None = None
    module: str | None = None
    stage: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    skip_reason: str | None = None
    baseline_seconds: float | None = None
    instrumented_seconds: float | None = None
    gpu_memory_overhead_bytes: int | None = None
    original_failure_size: int | None = None
    minimized_failure_size: int | None = None
    replay_fidelity: str | None = None

    def __post_init__(self) -> None:
        if self.detected and self.category is None:
            raise ValueError("detected BugBench observations require a category")
        if self.skip_reason is not None and self.detected:
            raise ValueError("skipped BugBench observations cannot be detected")
        if (self.baseline_seconds is None) != (self.instrumented_seconds is None):
            raise ValueError("baseline and instrumented timings must be supplied together")
        if self.baseline_seconds is not None and self.baseline_seconds <= 0:
            raise ValueError("baseline timing must be positive")
        if self.instrumented_seconds is not None and self.instrumented_seconds < 0:
            raise ValueError("instrumented timing cannot be negative")
        if (self.original_failure_size is None) != (self.minimized_failure_size is None):
            raise ValueError("original and minimized failure sizes must be supplied together")
        if self.original_failure_size is not None:
            if self.original_failure_size <= 0 or self.minimized_failure_size is None:
                raise ValueError("failure sizes must be positive")
            if not 0 < self.minimized_failure_size <= self.original_failure_size:
                raise ValueError("minimized failure size must be in (0, original]")
        if self.replay_fidelity not in {None, "L0", "L1", "L2", "L3", "L4"}:
            raise ValueError("replay_fidelity must be L0 through L4")


@dataclass(frozen=True, slots=True)
class BugBenchCaseResult:
    """Comparison between one observation and its declared ground truth."""

    ground_truth: BugBenchGroundTruth
    observation: BugBenchObservation
    elapsed_seconds: float
    error: str | None = None

    @property
    def skipped(self) -> bool:
        return self.observation.skip_reason is not None

    @property
    def internal_error(self) -> bool:
        return self.error is not None

    @property
    def true_positive(self) -> bool:
        return (
            not self.skipped
            and self.ground_truth.expected.detected
            and self.observation.detected
        )

    @property
    def false_negative(self) -> bool:
        return (
            not self.skipped
            and self.ground_truth.expected.detected
            and not self.observation.detected
        )

    @property
    def false_positive(self) -> bool:
        return (
            not self.skipped
            and not self.ground_truth.expected.detected
            and self.observation.detected
        )

    @property
    def true_negative(self) -> bool:
        return (
            not self.skipped
            and not self.internal_error
            and not self.ground_truth.expected.detected
            and not self.observation.detected
        )

    @property
    def diagnostic_correct(self) -> bool:
        return (
            self.true_positive
            and self.observation.category == self.ground_truth.expected.category
        )

    @property
    def localization_correct(self) -> bool:
        expected = self.ground_truth.expected
        return (
            self.true_positive
            and self.observation.module == expected.module
            and self.observation.stage == expected.stage
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe case result."""

        return {
            "id": self.ground_truth.identifier,
            "benchmark_category": self.ground_truth.benchmark_category,
            "description": self.ground_truth.description,
            "ground_truth_path": str(self.ground_truth.path),
            "expected": asdict(self.ground_truth.expected),
            "observed": asdict(self.observation),
            "elapsed_seconds": self.elapsed_seconds,
            "skipped": self.skipped,
            "internal_error": self.internal_error,
            "error": self.error,
            "classification": _classification(self),
            "diagnostic_correct": self.diagnostic_correct,
            "localization_correct": self.localization_correct,
        }


@dataclass(frozen=True, slots=True)
class BugBenchReport:
    """A reproducible BugBench run with metrics derived only from case results."""

    root: Path
    seed: int
    cases: tuple[BugBenchCaseResult, ...]
    elapsed_seconds: float

    @property
    def metrics(self) -> dict[str, Any]:
        """Compute benchmark metrics without favorable defaults or imputation."""

        true_positives = sum(case.true_positive for case in self.cases)
        false_negatives = sum(case.false_negative for case in self.cases)
        false_positives = sum(case.false_positive for case in self.cases)
        true_negatives = sum(case.true_negative for case in self.cases)
        detected_bugs = true_positives
        bug_denominator = true_positives + false_negatives
        control_denominator = true_negatives + false_positives
        overhead_ratios = [
            case.observation.instrumented_seconds / case.observation.baseline_seconds
            for case in self.cases
            if case.observation.instrumented_seconds is not None
            and case.observation.baseline_seconds is not None
        ]
        gpu_overheads = [
            case.observation.gpu_memory_overhead_bytes
            for case in self.cases
            if case.observation.gpu_memory_overhead_bytes is not None
        ]
        detection_times = [
            case.elapsed_seconds for case in self.cases if case.true_positive
        ]
        minimizations = [
            (
                case.observation.original_failure_size,
                case.observation.minimized_failure_size,
            )
            for case in self.cases
            if case.observation.original_failure_size is not None
            and case.observation.minimized_failure_size is not None
        ]
        fidelity = [
            case.observation.replay_fidelity
            for case in self.cases
            if case.observation.replay_fidelity is not None
        ]
        return {
            "cases": len(self.cases),
            "executed": sum(not case.skipped for case in self.cases),
            "skipped": sum(case.skipped for case in self.cases),
            "internal_errors": sum(case.internal_error for case in self.cases),
            "correctly_detected": true_positives,
            "missed": false_negatives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "detection_rate": _rate(true_positives, bug_denominator),
            "false_negative_rate": _rate(false_negatives, bug_denominator),
            "false_positive_rate": _rate(false_positives, control_denominator),
            "localization_accuracy": _rate(
                sum(case.localization_correct for case in self.cases), detected_bugs
            ),
            "diagnostic_accuracy": _rate(
                sum(case.diagnostic_correct for case in self.cases), detected_bugs
            ),
            "runtime_overhead": _distribution(overhead_ratios, unit="ratio"),
            "gpu_memory_overhead": _distribution(gpu_overheads, unit="bytes"),
            "time_to_detection": _distribution(detection_times, unit="seconds"),
            "failure_minimization_effectiveness": _minimization_metrics(minimizations),
            "replay_fidelity": {
                "available": bool(fidelity),
                "levels": {
                    level: fidelity.count(level) for level in ("L0", "L1", "L2", "L3", "L4")
                },
                "note": None if fidelity else "No fixture produced a rigorous L0-L4 replay result.",
            },
        }

    @property
    def passed(self) -> bool:
        metrics = self.metrics
        return bool(
            metrics["missed"] == 0
            and metrics["false_positives"] == 0
            and metrics["internal_errors"] == 0
        )

    @property
    def exit_code(self) -> int:
        """Return the CI exit taxonomy for this benchmark run."""

        if self.metrics["internal_errors"]:
            return 2
        return 0 if self.passed else 1

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned BugBench report schema."""

        return {
            "schema": {"name": "nablaguard.bugbench.report", "version": 1},
            "seed": self.seed,
            "root": str(self.root),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "elapsed_seconds": self.elapsed_seconds,
            "passed": self.passed,
            "metrics": self.metrics,
            "cases": [case.to_dict() for case in self.cases],
        }

    def format(self) -> str:
        """Render an honest compact terminal report."""

        metrics = self.metrics
        lines = [
            "NablaGuard BugBench",
            "=" * 32,
            f"Seed                          {self.seed}",
            f"Cases                         {metrics['cases']}",
            f"Executed                      {metrics['executed']}",
            f"Skipped                       {metrics['skipped']}",
            f"Internal errors               {metrics['internal_errors']}",
            "",
            f"Correctly detected            {metrics['correctly_detected']}",
            f"Missed                        {metrics['missed']}",
            f"False positives               {metrics['false_positives']}",
            "",
            f"Detection rate                {_percent(metrics['detection_rate'])}",
            f"False negative rate           {_percent(metrics['false_negative_rate'])}",
            f"False positive rate           {_percent(metrics['false_positive_rate'])}",
            f"Localization accuracy         {_percent(metrics['localization_accuracy'])}",
            f"Diagnostic accuracy           {_percent(metrics['diagnostic_accuracy'])}",
            "",
        ]
        runtime = metrics["runtime_overhead"]
        gpu = metrics["gpu_memory_overhead"]
        minimization = metrics["failure_minimization_effectiveness"]
        lines.append(
            "Median runtime overhead       "
            + (f"{runtime['median']:.3f}x" if runtime["available"] else "UNAVAILABLE")
        )
        lines.append(
            "Median GPU memory overhead    "
            + (f"{gpu['median']:.0f} bytes" if gpu["available"] else "UNAVAILABLE")
        )
        lines.append(
            "Failure minimization          "
            + (
                f"{minimization['mean_reduction']:.2%} mean size reduction"
                if minimization["available"]
                else "UNAVAILABLE"
            )
        )
        lines.extend(["", f"Result: {'PASS' if self.passed else 'FAIL'}"])
        skipped = [case for case in self.cases if case.skipped]
        if skipped:
            lines.extend(["", "Skipped cases:"])
            lines.extend(
                f"  {case.ground_truth.identifier}: {case.observation.skip_reason}"
                for case in skipped
            )
        failures = [
            case
            for case in self.cases
            if case.false_negative or case.false_positive or case.internal_error
        ]
        if failures:
            lines.extend(["", "Incorrect outcomes:"])
            lines.extend(
                f"  {case.ground_truth.identifier}: {_classification(case)}"
                + (f" ({case.error})" if case.error else "")
                for case in failures
            )
        return "\n".join(lines)


FixtureFunction = Callable[[CaseContext], BugBenchObservation]


def run_bugbench(
    root: str | Path,
    *,
    seed: int = 81927183,
    categories: Iterable[str] | None = None,
) -> BugBenchReport:
    """Discover, execute, and score BugBench ground-truth fixtures."""

    started = time.perf_counter()
    suite_root = Path(root).resolve()
    if not suite_root.is_dir():
        raise BugBenchConfigError(f"BugBench root does not exist: {suite_root}")
    selected = set(categories or ())
    paths = sorted(suite_root.glob("*/*.yaml"))
    if selected:
        paths = [path for path in paths if path.parent.name in selected]
        unknown = selected - {path.parent.name for path in paths}
        if unknown:
            raise BugBenchConfigError(f"unknown or empty BugBench categories: {sorted(unknown)}")
    if not paths:
        raise BugBenchConfigError(f"no BugBench ground truth found below {suite_root}")
    ground_truths = [BugBenchGroundTruth.load(path) for path in paths]
    identifiers = [truth.identifier for truth in ground_truths]
    duplicates = sorted({value for value in identifiers if identifiers.count(value) > 1})
    if duplicates:
        raise BugBenchConfigError(f"duplicate BugBench IDs: {duplicates}")
    results = tuple(
        _execute_case(truth, seed=seed + index) for index, truth in enumerate(ground_truths)
    )
    return BugBenchReport(suite_root, seed, results, time.perf_counter() - started)


def _execute_case(ground_truth: BugBenchGroundTruth, *, seed: int) -> BugBenchCaseResult:
    started = time.perf_counter()
    try:
        function = _load_fixture(ground_truth.implementation)
        observation = function(CaseContext(ground_truth.identifier, seed, ground_truth.path))
        if not isinstance(observation, BugBenchObservation):
            raise TypeError("fixture did not return BugBenchObservation")
    except Exception as error:
        observation = BugBenchObservation(False, evidence={"exception_type": type(error).__name__})
        return BugBenchCaseResult(
            ground_truth,
            observation,
            time.perf_counter() - started,
            f"{type(error).__name__}: {error}",
        )
    return BugBenchCaseResult(ground_truth, observation, time.perf_counter() - started)


def _load_fixture(target: str) -> FixtureFunction:
    if ":" not in target:
        raise BugBenchConfigError("fixture implementation must use module:qualified_name")
    module_name, path = target.split(":", 1)
    value: Any = importlib.import_module(module_name)
    for component in path.split("."):
        value = getattr(value, component)
    if not callable(value):
        raise BugBenchConfigError(f"fixture implementation is not callable: {target}")
    return cast(FixtureFunction, value)


def _required_string(value: Mapping[str, Any], key: str, path: Path) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise BugBenchConfigError(f"{key} must be a non-empty string in {path}")
    return result


def _optional_string(value: Mapping[str, Any], key: str, path: Path) -> str | None:
    result = value.get(key)
    if result is not None and (not isinstance(result, str) or not result):
        raise BugBenchConfigError(f"expected.{key} must be null or a non-empty string in {path}")
    return result


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percent(value: float | None) -> str:
    return "UNAVAILABLE" if value is None else f"{value:.2%}"


def _distribution(values: Sequence[float | int], *, unit: str) -> dict[str, Any]:
    if not values:
        return {"available": False, "samples": 0, "unit": unit, "median": None}
    return {
        "available": True,
        "samples": len(values),
        "unit": unit,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _minimization_metrics(values: list[tuple[int, int]]) -> dict[str, Any]:
    if not values:
        return {
            "available": False,
            "cases": 0,
            "mean_reduction": None,
            "fully_minimized_cases": None,
        }
    reductions = [1 - minimized / original for original, minimized in values]
    return {
        "available": True,
        "cases": len(values),
        "mean_reduction": statistics.fmean(reductions),
        "fully_minimized_cases": sum(minimized < original for original, minimized in values),
    }


def _classification(case: BugBenchCaseResult) -> str:
    if case.skipped:
        return "SKIPPED"
    if case.internal_error:
        return "INTERNAL_ERROR"
    if case.true_positive:
        return "TRUE_POSITIVE"
    if case.false_negative:
        return "FALSE_NEGATIVE"
    if case.false_positive:
        return "FALSE_POSITIVE"
    if case.true_negative:
        return "TRUE_NEGATIVE"
    return "UNKNOWN"


def json_dumps(report: BugBenchReport) -> str:
    """Encode a report without coercing unsupported values to strings."""

    return json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False)


__all__ = [
    "BugBenchCaseResult",
    "BugBenchConfigError",
    "BugBenchGroundTruth",
    "BugBenchObservation",
    "BugBenchReport",
    "CaseContext",
    "ExpectedOutcome",
    "json_dumps",
    "run_bugbench",
]
