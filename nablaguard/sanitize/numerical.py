"""Evidence calculations for dispatch-level numerical observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class ShadowComparison:
    """Scalar error summary between real and shadow outputs."""

    max_absolute_error: float
    max_relative_error: float
    mean_absolute_error: float
    max_ulp_difference: int | None
    finite_mismatch_count: int

    def to_dict(self) -> dict[str, float | int | None]:
        """Return JSON-safe evidence."""

        return {
            "max_absolute_error": self.max_absolute_error,
            "max_relative_error": self.max_relative_error,
            "mean_absolute_error": self.mean_absolute_error,
            "max_ulp_difference": self.max_ulp_difference,
            "finite_mismatch_count": self.finite_mismatch_count,
        }


def compare_shadow(real: torch.Tensor, shadow: torch.Tensor) -> ShadowComparison:
    """Compare real output with a high-precision shadow without retaining either."""

    observed = real.detach().to(torch.float64)
    reference = shadow.detach().to(torch.float64)
    if observed.shape != reference.shape or observed.numel() == 0:
        mismatch = 0 if observed.shape == reference.shape else 1
        error = 0.0 if mismatch == 0 else float("inf")
        return ShadowComparison(error, error, error, None, mismatch)
    observed_finite = torch.isfinite(observed)
    reference_finite = torch.isfinite(reference)
    comparable = observed_finite & reference_finite
    finite_mismatch = int((observed_finite != reference_finite).sum().item())
    if bool(comparable.any().item()):
        absolute = (observed[comparable] - reference[comparable]).abs()
        denominator = reference[comparable].abs().clamp_min(torch.finfo(torch.float64).tiny)
        relative = absolute / denominator
        max_absolute = float(absolute.max().item())
        max_relative = float(relative.max().item())
        mean_absolute = float(absolute.mean().item())
    else:
        max_absolute = 0.0
        max_relative = 0.0
        mean_absolute = 0.0
    if finite_mismatch:
        max_absolute = float("inf")
        max_relative = float("inf")
    return ShadowComparison(
        max_absolute_error=max_absolute,
        max_relative_error=max_relative,
        mean_absolute_error=mean_absolute,
        max_ulp_difference=max_ulp_difference(real.detach(), shadow.detach()),
        finite_mismatch_count=finite_mismatch,
    )


def max_ulp_difference(real: torch.Tensor, shadow: torch.Tensor) -> int | None:
    """Return maximum representable-step difference for supported real dtypes."""

    integer_dtype = {
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
        torch.float32: torch.int32,
    }.get(real.dtype)
    if integer_dtype is None or real.shape != shadow.shape or real.numel() == 0:
        return None
    expected = shadow.to(real.dtype)
    finite = torch.isfinite(real) & torch.isfinite(expected)
    if not bool(finite.any().item()):
        return None
    observed_bits = real[finite].contiguous().view(integer_dtype).to(torch.int64)
    expected_bits = expected[finite].contiguous().view(integer_dtype).to(torch.int64)
    minimum = torch.iinfo(integer_dtype).min
    observed_ordered = torch.where(observed_bits < 0, minimum - observed_bits, observed_bits)
    expected_ordered = torch.where(expected_bits < 0, minimum - expected_bits, expected_bits)
    return int((observed_ordered - expected_ordered).abs().max().item())


def exp_overflow_evidence(input_tensor: torch.Tensor) -> dict[str, float] | None:
    """Return evidence when finite exp inputs exceed the dtype overflow boundary."""

    if not input_tensor.is_floating_point() or input_tensor.numel() == 0:
        return None
    limit = math.log(torch.finfo(input_tensor.dtype).max)
    maximum = float(input_tensor.detach().max().item())
    if maximum <= limit:
        return None
    return {"input_max": maximum, "safe_exp_input_max": limit}


def reduction_cancellation(input_tensor: torch.Tensor, output: torch.Tensor) -> float | None:
    """Calculate ``1 - |sum(x)| / sum(|x|)`` for scalar sum reductions."""

    if input_tensor.numel() == 0 or output.numel() != 1:
        return None
    if not bool(torch.isfinite(input_tensor).all().item()) or not bool(
        torch.isfinite(output).all().item()
    ):
        return None
    magnitude_sum = float(input_tensor.detach().to(torch.float64).abs().sum().item())
    if magnitude_sum == 0:
        return 0.0
    result_magnitude = float(output.detach().to(torch.float64).abs().item())
    return max(0.0, min(1.0, 1.0 - result_magnitude / magnitude_sum))


def tensor_outputs(value: Any) -> tuple[torch.Tensor, ...]:
    """Flatten tensor outputs from common PyTorch container forms."""

    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(tensor for item in value for tensor in tensor_outputs(item))
    if isinstance(value, dict):
        return tuple(tensor for item in value.values() for tensor in tensor_outputs(item))
    return ()


def tensor_inputs(value: Any) -> tuple[torch.Tensor, ...]:
    """Flatten tensor operands from nested args and kwargs."""

    return tensor_outputs(value)
