"""Built-in loss contracts."""

from __future__ import annotations

import math

import torch

from .base import Contract, ContractContext


def finite(*, raise_on_failure: bool = False) -> Contract:
    """Require a present scalar loss to be finite."""

    def predicate(context: ContractContext) -> bool:
        if context.loss is None:
            return False
        if isinstance(context.loss, torch.Tensor):
            return bool(torch.isfinite(context.loss).all().item())
        return math.isfinite(float(context.loss))

    return Contract(
        "loss.finite",
        predicate,
        code="NG1001",
        category="NAN_DETECTED",
        message="Loss is missing or non-finite.",
        raise_on_failure=raise_on_failure,
        evidence_factory=lambda context: {"loss": _loss_value(context.loss)},
    )


def _loss_value(value: float | torch.Tensor | None) -> float | None:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.detach().item())
    return float(value) if value is not None else None
