"""Built-in training-history contracts."""

from __future__ import annotations

import torch

from .base import Contract, ContractContext


def loss_not_exploding(
    *, max_ratio: float = 10.0, window: int = 10, raise_on_failure: bool = False
) -> Contract:
    """Require current loss not to exceed a prior-window median by ``max_ratio``."""

    if max_ratio <= 1 or window <= 0:
        raise ValueError("max_ratio must exceed one and window must be positive")

    def ratio(context: ContractContext) -> float | None:
        if context.loss is None or not context.loss_history:
            return None
        current = _loss_value(context.loss)
        history = sorted(float(value) for value in context.loss_history[-window:])
        middle = len(history) // 2
        baseline = (
            history[middle] if len(history) % 2 else (history[middle - 1] + history[middle]) / 2
        )
        if baseline == 0:
            return float("inf") if current > 0 else 1.0
        return current / abs(baseline)

    def predicate(context: ContractContext) -> bool:
        value = ratio(context)
        return value is None or value <= max_ratio

    return Contract(
        "training.loss_not_exploding",
        predicate,
        code="NG4001",
        category="TRAINING_DIVERGENCE",
        message="Current loss exceeds the configured prior-window ratio.",
        raise_on_failure=raise_on_failure,
        evidence_factory=lambda context: {
            "observed_ratio": ratio(context),
            "max_ratio": max_ratio,
            "window": window,
        },
    )


def _loss_value(value: float | torch.Tensor) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("training loss contract requires a scalar loss")
        return float(value.detach().item())
    return float(value)
