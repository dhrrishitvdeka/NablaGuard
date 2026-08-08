"""Selective eager-mode ATen instrumentation."""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class DispatchObserver(Protocol):
    """Callback implemented by the numerical guard."""

    def _observe_dispatch(
        self,
        operation: torch._ops.OpOverload,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None: ...

    @property
    def _inside_analysis(self) -> bool: ...


class NumericalDispatchMode(TorchDispatchMode):  # type: ignore[misc]
    """Dispatch mode that delegates evidence collection after real execution."""

    def __init__(self, observer: DispatchObserver) -> None:
        super().__init__()
        self.observer = observer

    def __torch_dispatch__(
        self,
        func: torch._ops.OpOverload,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        normalized_kwargs = kwargs or {}
        output = func(*args, **normalized_kwargs)
        if not self.observer._inside_analysis:
            self.observer._observe_dispatch(func, args, normalized_kwargs, output)
        return output
