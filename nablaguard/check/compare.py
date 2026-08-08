"""Numerically robust tensor comparison helpers."""

from __future__ import annotations

import torch

from .results import Comparison


def compare_tensors(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> Comparison:
    """Compare tensors and retain only scalar evidence about the worst mismatch."""

    if candidate.shape != reference.shape:
        return Comparison(
            passed=False,
            max_absolute_error=float("inf"),
            max_relative_error=float("inf"),
            mean_absolute_error=float("inf"),
            note=(
                f"shape mismatch: candidate={tuple(candidate.shape)}, "
                f"reference={tuple(reference.shape)}"
            ),
        )

    observed = candidate.detach().to(torch.float64)
    expected = reference.detach().to(torch.float64)
    absolute = (observed - expected).abs()
    denominator = expected.abs().clamp_min(max(absolute_tolerance, torch.finfo(torch.float64).tiny))
    relative = absolute / denominator
    close = torch.isclose(
        observed,
        expected,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        equal_nan=False,
    )
    finite = torch.isfinite(observed) & torch.isfinite(expected)
    passed = bool(torch.all(close & finite).item())

    if absolute.numel() == 0:
        return Comparison(
            passed=True, max_absolute_error=0.0, max_relative_error=0.0, mean_absolute_error=0.0
        )

    safe_absolute = torch.nan_to_num(absolute, nan=float("inf"), posinf=float("inf"))
    flat_index = int(torch.argmax(safe_absolute.reshape(-1)).item())
    index = _unravel_index(flat_index, tuple(absolute.shape))
    return Comparison(
        passed=passed,
        max_absolute_error=float(torch.max(safe_absolute).item()),
        max_relative_error=float(torch.max(torch.nan_to_num(relative, nan=float("inf"))).item()),
        mean_absolute_error=float(torch.mean(safe_absolute).item()),
        failing_index=None if passed else index,
        candidate_value=None if passed else float(observed[index].item()),
        reference_value=None if passed else float(expected[index].item()),
    )


def _unravel_index(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    coordinates: list[int] = []
    for dimension in reversed(shape):
        coordinates.append(index % dimension)
        index //= dimension
    return tuple(reversed(coordinates))
