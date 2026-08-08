"""Small, deterministic failures with one known defect each."""

from __future__ import annotations

import torch


class WrongSquare(torch.autograd.Function):
    """Correct square forward with a missing factor of two in backward."""

    @staticmethod
    def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(value)  # type: ignore[attr-defined]
        return value.square()

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> torch.Tensor:
        (value,) = ctx.saved_tensors  # type: ignore[attr-defined]
        return gradient * value


def unstable_softmax(value: torch.Tensor) -> torch.Tensor:
    """Naive softmax that overflows for otherwise ordinary shifted logits."""

    exponentials = torch.exp(value)
    return exponentials / exponentials.sum(dim=-1, keepdim=True)


_STATE = [0]


def stateful_scale(value: torch.Tensor) -> torch.Tensor:
    """Deliberately stateful operator for nondeterminism regression checks."""

    _STATE[0] += 1
    return value * _STATE[0]


def boundary_bug(value: torch.Tensor) -> torch.Tensor:
    """Fail only on a shape landmark used to test fuzz discovery and shrinking."""

    return value + 1 if value.shape[-1] == 17 else value
