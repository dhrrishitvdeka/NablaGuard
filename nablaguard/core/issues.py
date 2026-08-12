"""Stable, shared issue representation used by every NablaGuard subsystem."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Issue severity suitable for both console and machine-readable reports."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class SourceLocation:
    """A source position associated with an observation when one is available."""

    filename: str
    line: int
    function: str | None = None


@dataclass(frozen=True, slots=True)
class NablaIssue:
    """Evidence-backed problem emitted by any NablaGuard subsystem."""

    code: str
    category: str
    severity: Severity
    message: str
    module_path: str | None = None
    operation: str | None = None
    source_location: SourceLocation | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    reproduction: dict[str, Any] | None = None
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        from nablaguard.core.redaction import redact_string

        value = asdict(self)
        value["severity"] = self.severity.value
        location = value.get("source_location")
        if isinstance(location, dict) and isinstance(location.get("filename"), str):
            location["filename"] = redact_string(location["filename"])
        return value
