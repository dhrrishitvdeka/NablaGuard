"""Observed boundary differences without heuristic causality claims."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ObservedChange:
    """One scalar change between adjacent captured boundaries."""

    label: str
    field: str
    before: Any
    after: Any
    relative_change: float | None = None


@dataclass(frozen=True, slots=True)
class BoundaryDiagnosis:
    """Evidence at N-1 and N, explicitly separated from unknown causality."""

    step: int
    trigger_batch: tuple[int, ...] | None
    observations: tuple[ObservedChange, ...]
    unknowns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe diagnosis evidence."""

        return {
            "step": self.step,
            "trigger_batch": list(self.trigger_batch) if self.trigger_batch else None,
            "observations": [asdict(value) for value in self.observations],
            "unknowns": list(self.unknowns),
        }


def diagnose_boundary(
    before: dict[str, Any], after: dict[str, Any], *, step: int
) -> BoundaryDiagnosis:
    """Compare loss and named tensor-fingerprint statistics at adjacent steps."""

    observations: list[ObservedChange] = []
    if before.get("loss") != after.get("loss"):
        observations.append(
            ObservedChange(
                "OBSERVED",
                "loss",
                before.get("loss"),
                after.get("loss"),
                _relative_change(before.get("loss"), after.get("loss")),
            )
        )
    before_fingerprints = before.get("fingerprints", {})
    after_fingerprints = after.get("fingerprints", {})
    for name in sorted(set(before_fingerprints) | set(after_fingerprints)):
        left = before_fingerprints.get(name)
        right = after_fingerprints.get(name)
        if left is None or right is None:
            observations.append(
                ObservedChange("OBSERVED", f"{name}.presence", left is not None, right is not None)
            )
            continue
        if left.get("checksum") != right.get("checksum"):
            observations.append(
                ObservedChange(
                    "OBSERVED", f"{name}.checksum", left.get("checksum"), right.get("checksum")
                )
            )
        for metric in ("minimum", "maximum", "mean", "std", "norm"):
            if left.get(metric) != right.get(metric):
                observations.append(
                    ObservedChange(
                        "OBSERVED",
                        f"{name}.{metric}",
                        left.get(metric),
                        right.get(metric),
                        _relative_change(left.get(metric), right.get(metric)),
                    )
                )
    observations.sort(
        key=lambda value: value.relative_change if value.relative_change is not None else -1,
        reverse=True,
    )
    batch = after.get("batch_indices")
    return BoundaryDiagnosis(
        step=step,
        trigger_batch=tuple(batch) if batch is not None else None,
        observations=tuple(observations),
        unknowns=(
            "The observed changes do not establish which change caused the predicate transition.",
            "Uncaptured activations, gradients, optimizer internals, data, or external state "
            "may contribute.",
        ),
    )


def _relative_change(before: Any, after: Any) -> float | None:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return None
    denominator = max(abs(float(before)), 1e-30)
    return abs(float(after) - float(before)) / denominator
