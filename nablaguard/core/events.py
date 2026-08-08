"""Bounded metadata events captured while differentiable code executes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .issues import SourceLocation


def _event_id() -> str:
    return uuid4().hex


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class TensorEvent:
    """Small tensor metadata record; it deliberately never owns tensor storage."""

    operation: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool
    module_path: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    mean: float | None = None
    std: float | None = None
    abs_max: float | None = None
    zero_fraction: float | None = None
    nan_count: int = 0
    inf_count: int = 0
    source_location: SourceLocation | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=_event_id)
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)
