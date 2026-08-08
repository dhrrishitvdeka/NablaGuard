"""Structured issue diffs for regression and CI comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .json import SerializableReport
from .normalize import issues


@dataclass(frozen=True, slots=True)
class ReportDiff:
    """Issue identities added, removed, or retained between reports."""

    added: tuple[dict[str, Any], ...]
    removed: tuple[dict[str, Any], ...]
    unchanged: tuple[dict[str, Any], ...]

    @property
    def regressed(self) -> bool:
        """Whether the newer report introduced an issue identity."""

        return bool(self.added)

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": list(self.added),
            "removed": list(self.removed),
            "unchanged": list(self.unchanged),
            "regressed": self.regressed,
        }


def compare(before: SerializableReport, after: SerializableReport) -> ReportDiff:
    """Compare issues using stable diagnostic and location identity fields."""

    old = {_identity(issue): issue for issue in issues(before.to_dict())}
    new = {_identity(issue): issue for issue in issues(after.to_dict())}
    return ReportDiff(
        added=tuple(dict(new[key]) for key in sorted(new.keys() - old.keys())),
        removed=tuple(dict(old[key]) for key in sorted(old.keys() - new.keys())),
        unchanged=tuple(dict(new[key]) for key in sorted(new.keys() & old.keys())),
    )


def _identity(issue: Mapping[str, Any]) -> tuple[str, str, str, str, int]:
    source = issue.get("source_location")
    location = source if isinstance(source, Mapping) else {}
    return (
        str(issue.get("code", "")),
        str(issue.get("category", "")),
        str(issue.get("module_path", "")),
        str(location.get("filename", "")),
        int(location.get("line", 0) or 0),
    )
