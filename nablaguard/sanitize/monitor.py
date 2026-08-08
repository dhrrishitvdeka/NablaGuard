"""Explicit and module-hook numerical monitoring."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any

import torch

from nablaguard.core import NablaConfig, NablaIssue, Session, Severity, TensorEvent
from nablaguard.core.config import Mode

from .statistics import TensorStatistics, compute_statistics


@dataclass(slots=True)
class Guard(Session):
    """Bounded numerical guard.

    Passing a model installs forward hooks for the duration of the context.
    ``observe`` supports tensors produced outside modules. Module filters are
    glob patterns and are evaluated before any statistics are computed.
    """

    model: torch.nn.Module | None = None
    modules: tuple[str, ...] | None = None
    exclude: tuple[str, ...] = ()
    _handles: list[Any] = field(default_factory=list, init=False, repr=False)

    def __enter__(self) -> Guard:
        # Explicit base calls avoid CPython's zero-argument super limitation
        # with dataclass(slots=True) class replacement.
        Session.__enter__(self)
        if self.model is not None:
            self._install_hooks()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        Session.__exit__(self, exc_type, exc, traceback)

    def observe(
        self, tensor: torch.Tensor, *, operation: str = "observe", module_path: str | None = None
    ) -> TensorEvent:
        """Summarize one tensor and emit evidence-backed numerical issues."""

        statistics = compute_statistics(tensor)
        event = _event(tensor, statistics, operation=operation, module_path=module_path)
        self.emit_event(event)
        if statistics.nan_count or statistics.inf_count:
            self.emit_issue(
                NablaIssue(
                    code="NG1001",
                    category="NAN_DETECTED" if statistics.nan_count else "INF_DETECTED",
                    severity=Severity.CRITICAL,
                    message="A tensor contains non-finite values.",
                    module_path=module_path,
                    operation=operation,
                    evidence={
                        "nan_count": statistics.nan_count,
                        "inf_count": statistics.inf_count,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                    },
                    suggestion="Inspect the earliest emitted non-finite event and its inputs.",
                )
            )
        threshold = self.config.extreme_value_threshold
        if (
            threshold is not None
            and statistics.abs_max is not None
            and statistics.abs_max > threshold
        ):
            self.emit_issue(
                NablaIssue(
                    code="NG1003",
                    category="OVERFLOW_RISK",
                    severity=Severity.HIGH,
                    message="A finite tensor exceeded the configured magnitude threshold.",
                    module_path=module_path,
                    operation=operation,
                    evidence={
                        "abs_max": statistics.abs_max,
                        "threshold": threshold,
                        "dtype": str(tensor.dtype),
                    },
                    suggestion=(
                        "Verify that this range is expected for the tensor dtype and next "
                        "operation."
                    ),
                )
            )
        return event

    def _install_hooks(self) -> None:
        assert self.model is not None
        for name, module in self.model.named_modules():
            path = name or "<root>"
            if not self._selected(path):
                continue
            self._handles.append(module.register_forward_hook(self._hook(path)))

    def _selected(self, name: str) -> bool:
        if any(fnmatch(name, pattern) for pattern in self.exclude):
            return False
        return self.modules is None or any(fnmatch(name, pattern) for pattern in self.modules)

    def _hook(self, name: str) -> Callable[[torch.nn.Module, Any, Any], None]:
        def hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
            del module, inputs
            for index, tensor in enumerate(_tensors(output)):
                self.observe(tensor, operation=f"forward_output[{index}]", module_path=name)

        return hook


def guard(
    model: torch.nn.Module | None = None,
    *,
    mode: Mode = "standard",
    modules: Iterable[str] | None = None,
    exclude: Iterable[str] = (),
    max_events: int = 10_000,
    extreme_value_threshold: float | None = None,
) -> Guard:
    """Create a bounded numerical monitoring context."""

    if mode not in {"light", "standard", "deep"}:
        raise ValueError("mode must be 'light', 'standard', or 'deep'")
    config = NablaConfig(
        mode=mode,
        max_events=max_events,
        extreme_value_threshold=extreme_value_threshold,
    )
    return Guard(
        config=config,
        model=model,
        modules=tuple(modules) if modules is not None else None,
        exclude=tuple(exclude),
    )


def sanitize(*tensors: torch.Tensor, extreme_value_threshold: float | None = None) -> Guard:
    """Inspect explicit tensors and return the populated guard report."""

    monitor = guard(extreme_value_threshold=extreme_value_threshold)
    with monitor:
        for index, tensor in enumerate(tensors):
            monitor.observe(tensor, operation=f"input[{index}]")
    return monitor


def _event(
    tensor: torch.Tensor,
    statistics: TensorStatistics,
    *,
    operation: str,
    module_path: str | None,
) -> TensorEvent:
    return TensorEvent(
        operation=operation,
        module_path=module_path,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        requires_grad=tensor.requires_grad,
        min_value=statistics.minimum,
        max_value=statistics.maximum,
        mean=statistics.mean,
        std=statistics.std,
        abs_max=statistics.abs_max,
        zero_fraction=statistics.zero_fraction,
        nan_count=statistics.nan_count,
        inf_count=statistics.inf_count,
    )


def _tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in value:
            result.extend(_tensors(item))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_tensors(item))
        return result
    return []
