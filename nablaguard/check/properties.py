"""Reusable reference-free properties for differentiable programs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import torch

from .compare import compare_tensors
from .results import Comparison

PropertyReturn: TypeAlias = bool | torch.Tensor | tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True, slots=True)
class Property:
    """Named predicate evaluated for every generated input case."""

    name: str
    predicate: Callable[..., PropertyReturn]

    def evaluate(
        self,
        inputs: tuple[torch.Tensor, ...],
        *,
        absolute_tolerance: float,
        relative_tolerance: float,
    ) -> Comparison:
        """Evaluate the predicate and normalize its evidence."""

        outcome = self.predicate(*inputs)
        if isinstance(outcome, tuple) and len(outcome) == 2:
            return compare_tensors(
                outcome[0],
                outcome[1],
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
        if isinstance(outcome, torch.Tensor):
            if outcome.numel() != 1:
                raise TypeError(
                    f"property {self.name!r} returned a non-scalar Tensor; return bool or "
                    "(actual, expected)"
                )
            passed = bool(outcome.detach().item())
        elif isinstance(outcome, bool):
            passed = outcome
        else:
            raise TypeError(
                f"property {self.name!r} returned {type(outcome)!r}; expected bool, "
                "scalar Tensor, or (actual, expected)"
            )
        error = 0.0 if passed else 1.0
        return Comparison(passed, error, error, error, note=f"property={self.name}")


def property(
    function: Callable[..., PropertyReturn] | None = None, *, name: str | None = None
) -> Property | Callable[[Callable[..., PropertyReturn]], Property]:
    """Decorate a reusable metamorphic or invariant property."""

    def decorate(callback: Callable[..., PropertyReturn]) -> Property:
        return Property(name or callback.__name__, callback)

    return decorate(function) if function is not None else decorate


def equivalent(actual: torch.Tensor, expected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Make an explicit numerical-equivalence property result."""

    return actual, expected
