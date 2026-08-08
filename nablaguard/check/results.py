"""Structured results from differentiable-operator verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nablaguard.core import NablaIssue


@dataclass(frozen=True, slots=True)
class Comparison:
    """Numerical comparison for one output or input gradient."""

    passed: bool
    max_absolute_error: float
    max_relative_error: float
    mean_absolute_error: float
    failing_index: tuple[int, ...] | None = None
    candidate_value: float | str | None = None
    reference_value: float | str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return asdict(self)


@dataclass(slots=True)
class OperatorCheckResult:
    """Complete forward and backward verification result."""

    candidate_name: str
    reference_name: str
    seed: int
    forward: tuple[Comparison, ...]
    backward: tuple[Comparison, ...]
    jvp: tuple[Comparison, ...] = ()
    double_backward: tuple[Comparison, ...] = ()
    finite_difference: tuple[Comparison, ...] = ()
    determinism: tuple[Comparison, ...] = ()
    issues: tuple[NablaIssue, ...] = ()
    artifact_path: Path | None = None
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether every requested forward and backward comparison passed."""

        return all(
            item.passed
            for item in (
                *self.forward,
                *self.backward,
                *self.jvp,
                *self.double_backward,
                *self.finite_difference,
                *self.determinism,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""

        return {
            "candidate": self.candidate_name,
            "reference": self.reference_name,
            "seed": self.seed,
            "passed": self.passed,
            "forward": [item.to_dict() for item in self.forward],
            "backward": [item.to_dict() for item in self.backward],
            "jvp": [item.to_dict() for item in self.jvp],
            "double_backward": [item.to_dict() for item in self.double_backward],
            "finite_difference": [item.to_dict() for item in self.finite_difference],
            "determinism": [item.to_dict() for item in self.determinism],
            "issues": [issue.to_dict() for issue in self.issues],
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
            "elapsed_seconds": self.elapsed_seconds,
            "metadata": self.metadata,
        }

    def format(self) -> str:
        """Render a concise, evidence-first terminal report."""

        from nablaguard.report.console import format_operator_result

        return format_operator_result(self)

    def print(self) -> None:
        """Print the terminal report."""

        print(self.format())
