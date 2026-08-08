"""Higher-precision shadow rules for selected ATen operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import torch

ShadowRule: TypeAlias = Callable[
    [torch._ops.OpOverload, tuple[Any, ...], dict[str, Any], torch.dtype], Any
]

SENSITIVE_OPERATIONS = frozenset(
    {
        "aten::_log_softmax",
        "aten::_softmax",
        "aten::bmm",
        "aten::div",
        "aten::exp",
        "aten::linalg_vector_norm",
        "aten::log",
        "aten::logsumexp",
        "aten::matmul",
        "aten::mean",
        "aten::mm",
        "aten::norm",
        "aten::prod",
        "aten::sum",
        "aten::var",
    }
)


@dataclass(slots=True)
class ShadowRegistry:
    """Registry keyed by stable ATen schema names such as ``aten::exp``."""

    rules: dict[str, ShadowRule] = field(default_factory=dict)

    def register(self, operation: str, rule: ShadowRule) -> None:
        """Register or replace a shadow implementation."""

        self.rules[_normalize_operation(operation)] = rule

    def resolve(self, operation: torch._ops.OpOverload) -> ShadowRule | None:
        """Return an exact rule, falling back to generic casting when sensitive."""

        name = operation._schema.name
        if name in self.rules:
            return self.rules[name]
        return generic_shadow if name in SENSITIVE_OPERATIONS else None


REGISTRY = ShadowRegistry()


def shadow_rule(
    operation: str | Callable[..., Any],
) -> Callable[[ShadowRule], ShadowRule]:
    """Register a higher-precision rule for an ATen schema or familiar callable."""

    name = _normalize_operation(operation)

    def decorate(rule: ShadowRule) -> ShadowRule:
        REGISTRY.register(name, rule)
        return rule

    return decorate


def generic_shadow(
    operation: torch._ops.OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    dtype: torch.dtype,
) -> Any:
    """Re-execute an operation after recursively promoting floating operands."""

    promoted_args = _promote(args, dtype)
    promoted_kwargs = _promote(kwargs, dtype)
    return operation(*promoted_args, **promoted_kwargs)


def _promote(value: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.to(dtype=dtype)
    if isinstance(value, torch.dtype) and value.is_floating_point:
        return dtype
    if isinstance(value, tuple):
        return tuple(_promote(item, dtype) for item in value)
    if isinstance(value, list):
        return [_promote(item, dtype) for item in value]
    if isinstance(value, dict):
        return {key: _promote(item, dtype) for key, item in value.items()}
    return value


def _normalize_operation(operation: str | Callable[..., Any]) -> str:
    if isinstance(operation, str):
        if operation.startswith("aten::"):
            return operation
        return f"aten::{operation.removeprefix('aten.').split('.')[0]}"
    name = getattr(operation, "__name__", str(operation))
    aliases = {"softmax": "_softmax", "log_softmax": "_log_softmax"}
    return f"aten::{aliases.get(name, name)}"
