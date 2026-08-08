"""Reusable predicates for captured training metadata."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from .runner import BoundaryState


def metric_greater_than(path: str, threshold: float) -> Callable[[BoundaryState], bool]:
    """Fail when a dotted metadata metric is greater than a threshold."""

    def predicate(state: BoundaryState) -> bool:
        value = _resolve(state.metadata, path)
        return value is not None and float(value) > threshold

    return predicate


def metric_less_than(path: str, threshold: float) -> Callable[[BoundaryState], bool]:
    """Fail when a dotted metadata metric is less than a threshold."""

    def predicate(state: BoundaryState) -> bool:
        value = _resolve(state.metadata, path)
        return value is not None and float(value) < threshold

    return predicate


def metric_nonfinite(path: str) -> Callable[[BoundaryState], bool]:
    """Fail when a dotted metadata metric is NaN or infinite."""

    def predicate(state: BoundaryState) -> bool:
        value = _resolve(state.metadata, path)
        return value is not None and not math.isfinite(float(value))

    return predicate


def _resolve(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            return None
        current = current[component]
    return current
