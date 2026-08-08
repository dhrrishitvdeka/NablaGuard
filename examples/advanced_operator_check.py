"""Run random-VJP, JVP, second-order, finite-difference, and determinism checks."""

import torch

import nablaguard as ng

result = ng.check.operator(
    candidate=lambda x: torch.sin(x) * x,
    reference=lambda x: torch.sin(x) * x,
    inputs=[ng.tensor(shape=(8,), dtype=torch.float64, layout="strided")],
    vjp_cotangent="random",
    check_jvp=True,
    check_double_backward=True,
    check_finite_difference=True,
    check_determinism=True,
    absolute_tolerance=1e-5,
    relative_tolerance=1e-5,
)
result.print()
