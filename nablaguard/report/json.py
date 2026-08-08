"""Machine-readable report serialization."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol


class SerializableReport(Protocol):
    """Structural type accepted by :func:`dumps`."""

    def to_dict(self) -> Mapping[str, Any]: ...


def dumps(report: SerializableReport, *, indent: int | None = 2) -> str:
    """Serialize a structured NablaGuard report to JSON."""

    return json.dumps(report.to_dict(), indent=indent, sort_keys=True, default=str)
