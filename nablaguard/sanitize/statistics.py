"""Bounded tensor statistics used by the numerical sanitizer."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TensorStatistics:
    """Scalar summary that owns no storage from its source tensor."""

    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None
    abs_max: float | None
    zero_fraction: float
    nan_count: int
    inf_count: int


def compute_statistics(tensor: torch.Tensor) -> TensorStatistics:
    """Compute finite-value statistics without retaining the tensor."""

    detached = tensor.detach()
    count = detached.numel()
    if count == 0:
        return TensorStatistics(None, None, None, None, None, 0.0, 0, 0)
    if not (detached.is_floating_point() or detached.is_complex()):
        values = detached.to(torch.float64)
        finite_values = values.reshape(-1)
        nan_count = 0
        inf_count = 0
    else:
        values = detached.abs() if detached.is_complex() else detached
        finite_mask = torch.isfinite(values)
        finite_values = values[finite_mask].to(torch.float64)
        nan_count = int(torch.isnan(values).sum().item())
        inf_count = int(torch.isinf(values).sum().item())
    zero_fraction = float((values == 0).sum().item()) / count
    if finite_values.numel() == 0:
        return TensorStatistics(None, None, None, None, None, zero_fraction, nan_count, inf_count)
    return TensorStatistics(
        minimum=float(finite_values.min().item()),
        maximum=float(finite_values.max().item()),
        mean=float(finite_values.mean().item()),
        std=float(finite_values.std(unbiased=False).item()),
        abs_max=float(finite_values.abs().max().item()),
        zero_fraction=zero_fraction,
        nan_count=nan_count,
        inf_count=inf_count,
    )
