"""Bounded tensor statistics used by the numerical sanitizer."""

from __future__ import annotations

import math
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
    sampled_elements: int
    total_elements: int


def compute_statistics(
    tensor: torch.Tensor, *, max_samples: int | None = None
) -> TensorStatistics:
    """Compute finite-value statistics without retaining the tensor.

    Non-finite counts always inspect the complete tensor. When ``max_samples``
    is set, descriptive statistics use a deterministic strided view bounded by
    that many elements; the returned sample counts make this explicit.
    """

    detached = tensor.detach()
    count = detached.numel()
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if count == 0:
        return TensorStatistics(None, None, None, None, None, 0.0, 0, 0, 0, 0)
    sampled = _bounded_sample(detached, max_samples) if max_samples is not None else detached
    sampled_count = sampled.numel()
    if not (detached.is_floating_point() or detached.is_complex()):
        values = sampled.to(torch.float64)
        finite_values = values.reshape(-1)
        nan_count = 0
        inf_count = 0
    else:
        full_values = detached.abs() if detached.is_complex() else detached
        values = sampled.abs() if sampled.is_complex() else sampled
        finite_mask = torch.isfinite(values)
        finite_values = values[finite_mask].to(torch.float64)
        nonfinite_counts = torch.stack(
            (torch.isnan(full_values).sum(), torch.isinf(full_values).sum())
        ).to(device="cpu")
        nan_count, inf_count = (int(value) for value in nonfinite_counts.tolist())
    zero_fraction = float((values == 0).sum().item()) / sampled_count
    if finite_values.numel() == 0:
        return TensorStatistics(
            None,
            None,
            None,
            None,
            None,
            zero_fraction,
            nan_count,
            inf_count,
            sampled_count,
            count,
        )
    return TensorStatistics(
        minimum=float(finite_values.min().item()),
        maximum=float(finite_values.max().item()),
        mean=float(finite_values.mean().item()),
        std=float(finite_values.std(unbiased=False).item()),
        abs_max=float(finite_values.abs().max().item()),
        zero_fraction=zero_fraction,
        nan_count=nan_count,
        inf_count=inf_count,
        sampled_elements=sampled_count,
        total_elements=count,
    )


def _bounded_sample(tensor: torch.Tensor, max_samples: int) -> torch.Tensor:
    sample = tensor
    while sample.numel() > max_samples and sample.ndim:
        dimension = max(range(sample.ndim), key=lambda index: sample.shape[index])
        other = sample.numel() // max(sample.shape[dimension], 1)
        target = max(1, max_samples // max(other, 1))
        step = max(2, math.ceil(sample.shape[dimension] / target))
        slices = [slice(None)] * sample.ndim
        slices[dimension] = slice(None, None, step)
        sample = sample[tuple(slices)]
    flattened = sample.reshape(-1)
    return flattened[:max_samples]
