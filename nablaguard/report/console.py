"""Dependency-free, first-class terminal rendering."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from nablaguard.core import NablaIssue

if TYPE_CHECKING:
    from nablaguard.check import Comparison, OperatorCheckResult
    from nablaguard.core import Session
    from nablaguard.trace import GradientReport


def format_operator_result(result: OperatorCheckResult) -> str:
    """Render an operator check as an evidence-first terminal report."""

    lines = [
        "NablaGuard operator verification",
        "=" * 32,
        f"Candidate: {result.candidate_name}",
        f"Reference: {result.reference_name}",
        f"NablaGuard seed: {result.seed}",
        "",
        f"Forward:  {_status(result.forward)}",
        f"Backward: {_status(result.backward) if result.backward else 'SKIPPED'}",
    ]
    optional = (
        ("JVP", result.jvp),
        ("Double backward", result.double_backward),
        ("Finite difference", result.finite_difference),
        ("Determinism", result.determinism),
    )
    lines.extend(f"{name}: {_status(values)}" for name, values in optional if values)
    for issue in result.issues:
        lines.extend(["", *_format_issue(issue)])
    if result.artifact_path is not None:
        lines.extend(["", "REPRODUCTION", f"Artifact: {result.artifact_path}"])
    lines.extend(["", f"Result: {'PASS' if result.passed else 'FAIL'}"])
    return "\n".join(lines)


def format_gradient_report(report: GradientReport) -> str:
    """Render exact multi-loss gradient geometry."""

    lines = [
        "Gradient provenance",
        "=" * 32,
        f"Parameter: {report.parameter_name}",
        "",
    ]
    for component in report.components:
        lines.append(
            f"{component.name}: norm={component.norm:.6g}, "
            f"magnitude share={component.magnitude_share:.1%}"
        )
    lines.extend(["", f"Cancellation: {report.cancellation:.1%}"])
    if report.cosine_similarities:
        lines.extend(["", "Pairwise cosine similarities:"])
        for pair in report.cosine_similarities:
            lines.append(f"  {pair.left} <-> {pair.right}: {pair.cosine:.6g}")
    for issue in report.issues:
        lines.extend(["", *_format_issue(issue)])
    return "\n".join(lines)


def format_session(session: Session) -> str:
    """Render issues and instrumentation cost for a shared session."""

    lines = [
        f"NablaGuard detected {len(session.issues)} problem(s).",
        f"Captured tensor events: {len(session.events)}",
    ]
    if session.dropped_events:
        lines.append(f"Dropped events at configured bound: {session.dropped_events}")
    for issue in session.issues:
        lines.extend(["", *_format_issue(issue)])
    return "\n".join(lines)


def _status(values: Sequence[Comparison]) -> str:
    return "PASS" if all(value.passed for value in values) else "FAIL"


def _format_issue(issue: NablaIssue) -> list[str]:
    location = None
    if issue.source_location is not None:
        location = f"{issue.source_location.filename}:{issue.source_location.line}"
    lines = [
        f"ISSUE {issue.code} {issue.category}",
        f"Severity: {issue.severity.value.upper()}",
        "OBSERVATION",
        issue.message,
    ]
    if location:
        lines.extend(["LOCATION", location])
    if issue.evidence:
        lines.append("EVIDENCE")
        for key, value in issue.evidence.items():
            lines.append(f"{key}: {value}")
    if issue.suggestion:
        lines.extend(["SUGGESTION", issue.suggestion])
    return lines
