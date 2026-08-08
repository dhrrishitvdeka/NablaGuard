"""Structured multi-loss gradient geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from nablaguard.core import NablaIssue


@dataclass(frozen=True, slots=True)
class GradientComponent:
    """Magnitude of one named loss gradient."""

    name: str
    norm: float
    magnitude_share: float


@dataclass(frozen=True, slots=True)
class GradientCosine:
    """Pairwise gradient direction relationship."""

    left: str
    right: str
    cosine: float


@dataclass(frozen=True, slots=True)
class GradientReport:
    """Gradient decomposition for one parameter.

    ``cancellation`` is exactly ``1 - ||sum(g_i)|| / sum(||g_i||)``. It is in
    ``[0, 1]`` up to floating-point error, is zero for fully aligned gradients,
    and approaches one when components nearly cancel. It describes magnitude
    loss, not causality or training quality.
    """

    parameter_name: str
    components: tuple[GradientComponent, ...]
    cosine_similarities: tuple[GradientCosine, ...]
    cancellation: float
    combined_norm: float
    component_norm_sum: float
    issues: tuple[NablaIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "parameter": self.parameter_name,
            "components": [asdict(value) for value in self.components],
            "cosine_similarities": [asdict(value) for value in self.cosine_similarities],
            "cancellation": self.cancellation,
            "combined_norm": self.combined_norm,
            "component_norm_sum": self.component_norm_sum,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def format(self) -> str:
        """Render this report for a terminal."""

        from nablaguard.report.console import format_gradient_report

        return format_gradient_report(self)

    def print(self) -> None:
        """Print the terminal report."""

        print(self.format())
