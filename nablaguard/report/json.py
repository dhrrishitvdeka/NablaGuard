"""Machine-readable report serialization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from nablaguard.core.serialization import dumps_json


class SerializableReport(Protocol):
    """Structural type accepted by :func:`dumps`."""

    def to_dict(self) -> Mapping[str, Any]: ...


def dumps(report: SerializableReport, *, indent: int | None = 2) -> str:
    """Serialize a structured NablaGuard report to strict JSON."""

    return dumps_json(dict(report.to_dict()), indent=indent)
