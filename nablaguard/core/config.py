"""Configuration shared across checking, tracing, and sanitizing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

Mode = Literal["light", "standard", "deep"]


@dataclass(frozen=True, slots=True)
class NablaConfig:
    """Immutable session configuration with explicit cost controls."""

    mode: Mode = "standard"
    seed: int = 81927183
    absolute_tolerance: float = 1e-7
    relative_tolerance: float = 1e-5
    max_events: int = 10_000
    extreme_value_threshold: float | None = None
    capture_inputs: bool = False
    artifact_dir: Path | None = None

    def with_overrides(self, **values: Any) -> NablaConfig:
        """Create a modified configuration without mutating an active session."""

        return replace(self, **values)
