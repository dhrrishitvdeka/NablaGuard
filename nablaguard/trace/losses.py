"""Exact multi-loss gradient decomposition using PyTorch autograd."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

import torch

from nablaguard.core import NablaIssue, Severity
from nablaguard.core.session import emit_issue

from .results import GradientComponent, GradientCosine, GradientReport

_LATEST_TRACE: ContextVar[LossTrace | None] = ContextVar("nablaguard_latest_trace", default=None)


@dataclass(slots=True)
class LossTrace:
    """Context manager that records per-loss gradients before normal backward.

    Parameters may be supplied explicitly for predictable cost. When omitted,
    trainable leaf tensors reachable from the loss autograd graphs are used.
    The trace retains detached gradient copies only, never their computation
    graphs.
    """

    named_losses: Mapping[str, torch.Tensor]
    parameters: Iterable[torch.Tensor] | None = None
    parameter_names: Mapping[torch.Tensor, str] | None = None
    conflict_threshold: float = -0.5
    cancellation_threshold: float = 0.5
    gradients: dict[torch.Tensor, dict[str, torch.Tensor]] = field(default_factory=dict, init=False)
    _token: Token[LossTrace | None] | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> LossTrace:
        if not self.named_losses:
            raise ValueError("named_losses must not be empty")
        selected = (
            list(self.parameters)
            if self.parameters is not None
            else _discover_leaves(self.named_losses.values())
        )
        if not selected:
            raise ValueError("no differentiable leaf parameters were found")
        self.gradients = {parameter: {} for parameter in selected}
        for name, loss in self.named_losses.items():
            if loss.numel() != 1:
                raise ValueError(f"loss {name!r} must be scalar")
            values = torch.autograd.grad(loss, selected, retain_graph=True, allow_unused=True)
            for parameter, value in zip(selected, values, strict=True):
                self.gradients[parameter][name] = (
                    torch.zeros_like(parameter) if value is None else value.detach().clone()
                )
        self._token = _LATEST_TRACE.set(self)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        # Leave the latest-trace ContextVar set so ``gradient()`` works after the
        # with-block. Nested enter replaces the value; call ``release()`` to drop
        # retained gradient copies when analysis is finished.
        self._token = None

    def release(self) -> None:
        """Drop retained gradient copies and clear the latest-trace handle if current."""

        self.gradients.clear()
        if _LATEST_TRACE.get() is self:
            _LATEST_TRACE.set(None)

    def report(self, parameter: torch.Tensor, *, name: str | None = None) -> GradientReport:
        """Calculate exact norms, cosines, and cancellation for one parameter."""

        try:
            gradients = self.gradients[parameter]
        except KeyError as error:
            raise KeyError("parameter was not part of this loss trace") from error
        norms = {
            key: float(torch.linalg.vector_norm(value).item()) for key, value in gradients.items()
        }
        norm_sum = sum(norms.values())
        components = tuple(
            GradientComponent(key, norm, norm / norm_sum if norm_sum else 0.0)
            for key, norm in norms.items()
        )
        pairs: list[GradientCosine] = []
        issues: list[NablaIssue] = []
        for left, right in itertools.combinations(gradients, 2):
            denominator = norms[left] * norms[right]
            cosine = (
                float(torch.sum(gradients[left] * gradients[right]).item()) / denominator
                if denominator
                else math.nan
            )
            pairs.append(GradientCosine(left, right, cosine))
            if not math.isnan(cosine) and cosine <= self.conflict_threshold:
                issues.append(
                    NablaIssue(
                        code="NG2003",
                        category="GRADIENT_CONFLICT",
                        severity=Severity.MEDIUM,
                        message=(
                            f"Loss gradients {left!r} and {right!r} strongly oppose each other."
                        ),
                        evidence={"left": left, "right": right, "cosine_similarity": cosine},
                        suggestion=(
                            "Inspect loss scaling and confirm that this trade-off is intended."
                        ),
                    )
                )
        combined = torch.stack(tuple(gradients.values())).sum(dim=0)
        combined_norm = float(torch.linalg.vector_norm(combined).item())
        cancellation = max(0.0, min(1.0, 1.0 - combined_norm / norm_sum)) if norm_sum else 0.0
        if cancellation >= self.cancellation_threshold:
            issues.append(
                NablaIssue(
                    code="NG2004",
                    category="GRADIENT_CANCELLATION",
                    severity=Severity.MEDIUM,
                    message="Named loss gradients substantially cancel in their combined update.",
                    evidence={
                        "definition": "1 - norm(sum(g_i)) / sum(norm(g_i))",
                        "cancellation": cancellation,
                        "combined_norm": combined_norm,
                        "component_norm_sum": norm_sum,
                    },
                    suggestion="Review the pairwise cosines before changing loss weights.",
                )
            )
        for issue in issues:
            emit_issue(issue)
        return GradientReport(
            parameter_name=(
                name
                or (
                    self.parameter_names.get(parameter)
                    if self.parameter_names is not None
                    else None
                )
                or _parameter_name(parameter)
            ),
            components=components,
            cosine_similarities=tuple(pairs),
            cancellation=cancellation,
            combined_norm=combined_norm,
            component_norm_sum=norm_sum,
            issues=tuple(issues),
        )


def losses(
    named_losses: Mapping[str, torch.Tensor],
    *,
    parameters: Iterable[torch.Tensor] | None = None,
    parameter_names: Mapping[torch.Tensor, str] | None = None,
    conflict_threshold: float = -0.5,
    cancellation_threshold: float = 0.5,
) -> LossTrace:
    """Create a multi-loss gradient trace context."""

    return LossTrace(
        named_losses,
        parameters,
        parameter_names,
        conflict_threshold,
        cancellation_threshold,
    )


def gradient(parameter: torch.Tensor, *, name: str | None = None) -> GradientReport:
    """Report a parameter from the most recent trace in this context."""

    trace = _LATEST_TRACE.get()
    if trace is None:
        raise RuntimeError("no loss trace is available; use `with ng.trace.losses(...)` first")
    return trace.report(parameter, name=name)


def _discover_leaves(losses_to_walk: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    leaves: dict[int, torch.Tensor] = {}
    seen: set[int] = set()
    stack = [loss.grad_fn for loss in losses_to_walk if loss.grad_fn is not None]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        variable = getattr(node, "variable", None)
        if isinstance(variable, torch.Tensor) and variable.requires_grad and variable.is_leaf:
            leaves[id(variable)] = variable
        stack.extend(next_node for next_node, _ in node.next_functions if next_node is not None)
    return list(leaves.values())


def _parameter_name(parameter: torch.Tensor) -> str:
    return f"tensor(shape={tuple(parameter.shape)}, dtype={parameter.dtype}, id={id(parameter)})"
