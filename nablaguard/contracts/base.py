"""Training-specific contracts that emit ordinary NablaGuard issues."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from nablaguard.core import NablaIssue, Severity
from nablaguard.core.session import emit_issue


@dataclass(slots=True)
class ContractContext:
    """Explicit values available to a training contract."""

    loss: float | torch.Tensor | None = None
    tensor: torch.Tensor | None = None
    gradients: Mapping[str, torch.Tensor] | None = None
    parameters: Mapping[str, torch.Tensor] | None = None
    previous_parameters: Mapping[str, torch.Tensor] | None = None
    loss_history: Sequence[float] = ()
    module_path: str | None = None
    operation: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class ContractViolation(RuntimeError):
    """Raised only when a contract explicitly requests fail-fast behavior."""

    def __init__(self, issue: NablaIssue) -> None:
        super().__init__(f"{issue.code} {issue.category}: {issue.message}")
        self.issue = issue


ContractPredicate = Callable[[ContractContext], bool | torch.Tensor]
EvidenceFactory = Callable[[ContractContext], dict[str, Any]]
ArtifactCallback = Callable[[NablaIssue, ContractContext], None]


@dataclass(frozen=True, slots=True)
class Contract:
    """Named assertion with stable issue metadata."""

    name: str
    predicate: ContractPredicate
    code: str = "NG5001"
    category: str = "CONTRACT_FAILED"
    severity: Severity = Severity.HIGH
    message: str | None = None
    suggestion: str | None = None
    evidence_factory: EvidenceFactory | None = None
    raise_on_failure: bool = False
    artifact_callback: ArtifactCallback | None = None
    requires: frozenset[str] = frozenset()
    loss_history_window: int = 0

    def evaluate(self, context: ContractContext) -> NablaIssue | None:
        """Evaluate, emit, optionally persist, and optionally raise a violation."""

        result = self.predicate(context)
        if isinstance(result, torch.Tensor):
            if result.numel() != 1:
                raise TypeError(f"contract {self.name!r} returned a non-scalar Tensor")
            passed = bool(result.detach().item())
        elif isinstance(result, bool):
            passed = result
        else:
            raise TypeError(f"contract {self.name!r} must return bool or scalar Tensor")
        if passed:
            return None
        evidence = {"contract": self.name}
        if self.evidence_factory is not None:
            evidence.update(self.evidence_factory(context))
        issue = NablaIssue(
            code=self.code,
            category=self.category,
            severity=self.severity,
            message=self.message or f"Training contract {self.name!r} failed.",
            module_path=context.module_path,
            operation=context.operation,
            evidence=evidence,
            suggestion=self.suggestion,
        )
        emit_issue(issue)
        if self.artifact_callback is not None:
            self.artifact_callback(issue, context)
        if self.raise_on_failure:
            raise ContractViolation(issue)
        return issue

    __call__ = evaluate


def contract(
    name: str,
    predicate: ContractPredicate,
    *,
    code: str = "NG5001",
    category: str = "CONTRACT_FAILED",
    severity: Severity = Severity.HIGH,
    message: str | None = None,
    suggestion: str | None = None,
    raise_on_failure: bool = False,
    artifact_callback: ArtifactCallback | None = None,
) -> Contract:
    """Create a custom training contract."""

    return Contract(
        name=name,
        predicate=predicate,
        code=code,
        category=category,
        severity=severity,
        message=message,
        suggestion=suggestion,
        raise_on_failure=raise_on_failure,
        artifact_callback=artifact_callback,
    )
