"""Structured property-fuzzing results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nablaguard.core import NablaIssue

from .results import Comparison, OperatorCheckResult
from .specs import TensorSpec


@dataclass(slots=True)
class FuzzFailure:
    """One reproducible, optionally minimized failing trial."""

    trial: int
    seed: int
    reason: str
    original_specs: tuple[TensorSpec, ...]
    minimal_specs: tuple[TensorSpec, ...]
    minimization_steps: tuple[str, ...] = ()
    operator_result: OperatorCheckResult | None = None
    property_results: tuple[tuple[str, Comparison], ...] = ()
    issues: tuple[NablaIssue, ...] = ()
    artifact_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe failure report."""

        return {
            "trial": self.trial,
            "seed": self.seed,
            "reason": self.reason,
            "original_specs": [spec.to_dict() for spec in self.original_specs],
            "minimal_specs": [spec.to_dict() for spec in self.minimal_specs],
            "minimization_steps": list(self.minimization_steps),
            "operator_result": self.operator_result.to_dict() if self.operator_result else None,
            "property_results": [
                {"name": name, "comparison": comparison.to_dict()}
                for name, comparison in self.property_results
            ],
            "issues": [issue.to_dict() for issue in self.issues],
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
        }


@dataclass(frozen=True, slots=True)
class FuzzResult:
    """Summary of a deterministic tensor-operator fuzzing run."""

    seed: int
    requested_trials: int
    cases_run: int
    skipped_cases: int
    failures: tuple[FuzzFailure, ...]
    elapsed_seconds: float

    @property
    def passed(self) -> bool:
        """Whether no valid case failed."""

        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe run report."""

        return {
            "seed": self.seed,
            "requested_trials": self.requested_trials,
            "cases_run": self.cases_run,
            "skipped_cases": self.skipped_cases,
            "passed": self.passed,
            "elapsed_seconds": self.elapsed_seconds,
            "failures": [failure.to_dict() for failure in self.failures],
        }

    def format(self) -> str:
        """Render a concise terminal report."""

        lines = [
            "NablaGuard differentiable-operator fuzzing",
            "=" * 40,
            f"NablaGuard seed: {self.seed}",
            f"Cases run: {self.cases_run}/{self.requested_trials}",
            f"Skipped invalid cases: {self.skipped_cases}",
            f"Failures: {len(self.failures)}",
        ]
        for failure in self.failures:
            lines.extend(
                [
                    "",
                    f"TRIAL {failure.trial} (seed {failure.seed})",
                    f"Reason: {failure.reason}",
                    "Original shapes: "
                    + str([list(spec.shape) for spec in failure.original_specs]),
                    "Minimal known failing shapes: "
                    + str([list(spec.shape) for spec in failure.minimal_specs]),
                ]
            )
            if failure.artifact_path:
                lines.append(f"Artifact: {failure.artifact_path}")
        lines.extend(["", f"Result: {'PASS' if self.passed else 'FAIL'}"])
        return "\n".join(lines)

    def print(self) -> None:
        """Print the terminal report."""

        print(self.format())
