"""Opt-in JVP, higher-order, finite-difference, and determinism checks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from nablaguard.capture.rng import capture_rng_state, restore_rng_state

from .compare import compare_tensors
from .isolation import call_with_isolated_module_state, leaf_copy
from .results import Comparison


def compare_jvp(
    candidate: Callable[..., Any],
    reference: Callable[..., Any],
    inputs: Sequence[torch.Tensor],
    *,
    seed: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[Comparison, ...]:
    """Compare candidate and reference JVPs under seeded random tangents."""

    candidate_inputs = tuple(_leaf(value) for value in inputs)
    reference_inputs = tuple(_leaf(value) for value in inputs)
    tangents = tuple(_random_like(value, seed + index) for index, value in enumerate(inputs))
    state = capture_rng_state()
    try:
        restore_rng_state(state)
        _, candidate_jvp = torch.autograd.functional.jvp(
            lambda *values: call_with_isolated_module_state(candidate, values),
            candidate_inputs,
            tangents,
            create_graph=False,
            strict=False,
        )
        restore_rng_state(state)
        _, reference_jvp = torch.autograd.functional.jvp(
            lambda *values: call_with_isolated_module_state(reference, values),
            reference_inputs,
            tuple(value.clone() for value in tangents),
            create_graph=False,
            strict=False,
        )
    except (RuntimeError, TypeError, NotImplementedError) as error:
        return (_error_comparison("JVP", error),)
    finally:
        restore_rng_state(state)
    return _compare_outputs(
        candidate_jvp,
        reference_jvp,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


def compare_double_backward(
    candidate: Callable[..., Any],
    reference: Callable[..., Any],
    inputs: Sequence[torch.Tensor],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[Comparison, ...]:
    """Compare Hessian-vector products induced by all-ones first-gradient vectors."""

    state = capture_rng_state()
    try:
        restore_rng_state(state)
        candidate_values = _second_order(
            lambda *values: call_with_isolated_module_state(candidate, values), inputs
        )
        restore_rng_state(state)
        reference_values = _second_order(
            lambda *values: call_with_isolated_module_state(reference, values), inputs
        )
    except (RuntimeError, TypeError, NotImplementedError) as error:
        return (_error_comparison("double backward", error),)
    finally:
        restore_rng_state(state)
    return tuple(
        compare_tensors(
            observed,
            expected,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        for observed, expected in zip(candidate_values, reference_values, strict=True)
    )


def compare_finite_difference(
    candidate: Callable[..., Any],
    inputs: Sequence[torch.Tensor],
    *,
    epsilon: float,
    max_elements: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[Comparison, ...]:
    """Compare candidate autograd VJP with central finite differences of output sum."""

    if epsilon <= 0:
        raise ValueError("finite_difference_epsilon must be positive")
    differentiable = [
        value for value in inputs if value.is_floating_point() and value.requires_grad
    ]
    element_count = sum(value.numel() for value in differentiable)
    if element_count > max_elements:
        raise ValueError(
            "finite difference request exceeds max_finite_difference_elements: "
            f"{element_count} > {max_elements}"
        )
    analytical_inputs = tuple(_leaf(value) for value in inputs)
    try:
        outputs = _outputs(call_with_isolated_module_state(candidate, analytical_inputs))
        analytical = torch.autograd.grad(
            outputs,
            analytical_inputs,
            grad_outputs=tuple(torch.ones_like(value) for value in outputs),
            allow_unused=True,
        )
    except (RuntimeError, TypeError, NotImplementedError) as error:
        return (_error_comparison("finite-difference analytical gradient", error),)
    comparisons: list[Comparison] = []
    for input_index, (value, gradient) in enumerate(zip(inputs, analytical, strict=True)):
        # Finite differences estimate ∂phi/∂x only for inputs that participate in
        # autograd; inventing requires_grad would mis-model mixed differentiability.
        if not value.is_floating_point() or not value.requires_grad:
            continue
        numerical = torch.empty(value.shape, dtype=torch.float64, device=value.device)
        flat = numerical.reshape(-1)
        for element in range(value.numel()):
            positive = [_detached_layout_copy(item) for item in inputs]
            negative = [_detached_layout_copy(item) for item in inputs]
            _perturb(positive[input_index], element, epsilon)
            _perturb(negative[input_index], element, -epsilon)
            with torch.no_grad():
                high = _objective(call_with_isolated_module_state(candidate, positive))
                low = _objective(call_with_isolated_module_state(candidate, negative))
            flat[element] = (high - low) / (2 * epsilon)
        observed = torch.zeros_like(value) if gradient is None else gradient
        comparisons.append(
            compare_tensors(
                observed,
                numerical.to(dtype=observed.dtype, device=observed.device),
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
        )
    return tuple(comparisons)


def compare_determinism(
    candidate: Callable[..., Any],
    inputs: Sequence[torch.Tensor],
) -> tuple[Comparison, ...]:
    """Repeat a candidate under identical captured RNG state and compare exactly."""

    state = capture_rng_state()
    try:
        restore_rng_state(state)
        first = call_with_isolated_module_state(
            candidate, tuple(_leaf(value) for value in inputs)
        )
        restore_rng_state(state)
        second = call_with_isolated_module_state(
            candidate, tuple(_leaf(value) for value in inputs)
        )
    except (RuntimeError, TypeError, NotImplementedError) as error:
        return (_error_comparison("determinism", error),)
    finally:
        restore_rng_state(state)
    return _compare_outputs(
        first,
        second,
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )


def _second_order(
    function: Callable[..., Any], inputs: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    leaves = tuple(_leaf(value) for value in inputs)
    outputs = _outputs(function(*leaves))
    first = torch.autograd.grad(
        outputs,
        leaves,
        grad_outputs=tuple(torch.ones_like(value) for value in outputs),
        create_graph=True,
        allow_unused=True,
    )
    active = [value for value in first if value is not None and value.requires_grad]
    if not active:
        return tuple(torch.zeros_like(value) for value in leaves)
    scalar = sum((value.sum() for value in active), start=torch.zeros((), device=active[0].device))
    second = torch.autograd.grad(scalar, leaves, allow_unused=True)
    return tuple(
        torch.zeros_like(input_value) if gradient is None else gradient
        for input_value, gradient in zip(leaves, second, strict=True)
    )


def _random_like(value: torch.Tensor, seed: int) -> torch.Tensor:
    if not value.is_floating_point():
        return torch.zeros_like(value)
    generator = torch.Generator(device=value.device).manual_seed(seed)
    return torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)


def _perturb(value: torch.Tensor, linear_index: int, amount: float) -> None:
    coordinates: list[int] = []
    remaining = linear_index
    for size in reversed(value.shape):
        coordinates.append(remaining % size)
        remaining //= size
    value[tuple(reversed(coordinates))] += amount


def _detached_layout_copy(value: torch.Tensor) -> torch.Tensor:
    """Stride-preserving copy that is safe for in-place finite-difference probes."""

    copy = leaf_copy(value)
    copy.requires_grad_(False)
    return copy


def _leaf(value: torch.Tensor) -> torch.Tensor:
    """Clone a leaf for advanced checks without inventing requires_grad.

    Uses the same stride-preserving copy as the base operator check so JVP,
    finite-difference, double-backward, and determinism see the same layout.
    """

    return leaf_copy(value)


def _outputs(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)) and all(isinstance(item, torch.Tensor) for item in value):
        return tuple(value)
    raise TypeError("advanced checks require Tensor or Tensor-sequence outputs")


def _objective(value: Any) -> float:
    """Scalar matching an all-ones cotangent: Re(sum(output))."""

    total = 0.0
    for tensor in _outputs(value):
        detached = tensor.detach()
        if detached.is_complex():
            total += float(detached.real.to(torch.float64).sum().item())
        else:
            total += float(detached.to(torch.float64).sum().item())
    return total


def _compare_outputs(
    candidate: Any,
    reference: Any,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[Comparison, ...]:
    observed = _outputs(candidate)
    expected = _outputs(reference)
    if len(observed) != len(expected):
        return (
            Comparison(
                False,
                float("inf"),
                float("inf"),
                float("inf"),
                note=f"advanced output count mismatch: {len(observed)} != {len(expected)}",
            ),
        )
    return tuple(
        compare_tensors(
            left,
            right,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        for left, right in zip(observed, expected, strict=True)
    )


def _error_comparison(check: str, error: Exception) -> Comparison:
    return Comparison(
        False,
        float("inf"),
        float("inf"),
        float("inf"),
        note=f"{check} raised {type(error).__name__}: {error}",
    )
