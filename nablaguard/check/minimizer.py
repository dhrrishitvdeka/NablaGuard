"""Deterministic shrinking for failing tensor recipes."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

import torch

from .specs import Distribution, TensorSpec


@dataclass(frozen=True, slots=True)
class MinimizationResult:
    """Smallest known failing recipe found within an explicit attempt budget."""

    specs: tuple[TensorSpec, ...]
    attempts: int
    accepted_steps: tuple[str, ...]
    exhausted: bool


def minimize(
    specs: tuple[TensorSpec, ...],
    fails: Callable[[tuple[TensorSpec, ...]], bool],
    *,
    max_attempts: int = 100,
) -> MinimizationResult:
    """Greedily shrink shape, layout, distribution, and dtype.

    This is a best-effort local minimizer, not a proof of global minimality.
    Every accepted candidate is re-executed through ``fails``.
    """

    current = specs
    attempts = 0
    steps: list[str] = []
    changed = True
    while changed and attempts < max_attempts:
        changed = False
        for input_index, spec in enumerate(current):
            for description, candidate in _candidates(spec):
                if attempts >= max_attempts:
                    break
                attempts += 1
                proposed = (*current[:input_index], candidate, *current[input_index + 1 :])
                if fails(proposed):
                    current = proposed
                    steps.append(f"input[{input_index}] {description}")
                    changed = True
                    break
            if changed or attempts >= max_attempts:
                break
    return MinimizationResult(current, attempts, tuple(steps), attempts >= max_attempts)


def _candidates(spec: TensorSpec) -> Iterator[tuple[str, TensorSpec]]:
    if len(spec.shape) > 1:
        for dimension in range(len(spec.shape)):
            candidate_shape = spec.shape[:dimension] + spec.shape[dimension + 1 :]
            yield (
                f"drop dimension {dimension}: {spec.shape} -> {candidate_shape}",
                replace(spec, shape=candidate_shape, layout="contiguous"),
            )
    for dimension, size in enumerate(spec.shape):
        for smaller in _smaller_dimensions(size):
            shape = list(spec.shape)
            shape[dimension] = smaller
            candidate_shape = tuple(shape)
            yield f"shape {spec.shape} -> {candidate_shape}", replace(spec, shape=candidate_shape)
    if spec.layout != "contiguous":
        yield f"layout {spec.layout} -> contiguous", replace(spec, layout="contiguous")
    for distribution in _simpler_distributions(spec.distribution):
        yield (
            f"distribution {spec.distribution} -> {distribution}",
            replace(spec, distribution=distribution),
        )
    dtype_order = (torch.float64, torch.float32, torch.bfloat16, torch.float16)
    current_dtype_rank = (
        dtype_order.index(spec.dtype) if spec.dtype in dtype_order else len(dtype_order)
    )
    for dtype in dtype_order[:current_dtype_rank]:
        if dtype != spec.dtype:
            yield f"dtype {spec.dtype} -> {dtype}", replace(spec, dtype=dtype)


def _smaller_dimensions(size: int) -> tuple[int, ...]:
    landmarks = (0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 64, 127, 128, 129, 257)
    values = {value for value in landmarks if value < size}
    if size > 1:
        values.add(size // 2)
    return tuple(sorted(values))


def _simpler_distributions(distribution: Distribution) -> tuple[Distribution, ...]:
    order: tuple[Distribution, ...] = ("zeros", "ones", "uniform", "normal")
    current_rank = order.index(distribution) if distribution in order else len(order)
    return order[:current_rank]
