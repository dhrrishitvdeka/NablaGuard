"""Forward and backward verification for PyTorch callables."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from nablaguard.core import NablaIssue, Severity
from nablaguard.core.session import emit_issue

from .artifacts import write_failure_artifact
from .compare import compare_tensors
from .results import Comparison, OperatorCheckResult
from .specs import TensorSpec


def operator(
    *,
    candidate: Callable[..., Any],
    reference: Callable[..., Any],
    inputs: Sequence[TensorSpec | torch.Tensor],
    seed: int = 81927183,
    absolute_tolerance: float = 1e-7,
    relative_tolerance: float = 1e-5,
    check_backward: bool = True,
    artifact_dir: str | Path | None = None,
) -> OperatorCheckResult:
    """Compare a candidate operator with a trusted PyTorch reference.

    Backward verification compares vector-Jacobian products under the same
    all-ones cotangent. A seed is always reported and input generation uses a
    private generator, so the experiment does not perturb global RNG state.
    """

    started = time.perf_counter()
    originals = [_materialize(value, seed + index) for index, value in enumerate(inputs)]
    candidate_inputs = [_leaf_copy(value) for value in originals]
    reference_inputs = [_leaf_copy(value) for value in originals]

    candidate_outputs = _as_tensor_tuple(candidate(*candidate_inputs))
    reference_outputs = _as_tensor_tuple(reference(*reference_inputs))
    forward = _compare_sequences(
        candidate_outputs,
        reference_outputs,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )

    backward: tuple[Comparison, ...] = ()
    if check_backward:
        backward = _compare_backward(
            candidate_outputs,
            reference_outputs,
            candidate_inputs,
            reference_inputs,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )

    issues = _issues_from_comparisons(forward, backward)
    for issue in issues:
        emit_issue(issue)

    result = OperatorCheckResult(
        candidate_name=_callable_name(candidate),
        reference_name=_callable_name(reference),
        seed=seed,
        forward=forward,
        backward=backward,
        issues=issues,
        elapsed_seconds=time.perf_counter() - started,
        metadata={
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "input_shapes": [list(value.shape) for value in originals],
            "input_dtypes": [str(value.dtype) for value in originals],
        },
    )
    if not result.passed and artifact_dir is not None:
        result.artifact_path = write_failure_artifact(
            Path(artifact_dir), metadata=result.to_dict(), inputs=originals
        )
    return result


def _materialize(value: TensorSpec | torch.Tensor, seed: int) -> torch.Tensor:
    if isinstance(value, TensorSpec):
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed)
        return value.generate(generator).detach()
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    raise TypeError(f"inputs must contain TensorSpec or Tensor, got {type(value)!r}")


def _leaf_copy(value: torch.Tensor) -> torch.Tensor:
    copy = value.detach().clone()
    if copy.is_floating_point() or copy.is_complex():
        copy.requires_grad_(True)
    return copy


def _as_tensor_tuple(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)) and all(isinstance(item, torch.Tensor) for item in value):
        return tuple(value)
    raise TypeError("candidate and reference must return a Tensor or a sequence of Tensors")


def _compare_sequences(
    candidate: tuple[torch.Tensor, ...],
    reference: tuple[torch.Tensor, ...],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[Comparison, ...]:
    if len(candidate) != len(reference):
        return (
            Comparison(
                False,
                float("inf"),
                float("inf"),
                float("inf"),
                note=(
                    f"output count mismatch: candidate={len(candidate)}, reference={len(reference)}"
                ),
            ),
        )
    return tuple(
        compare_tensors(
            observed,
            expected,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        for observed, expected in zip(candidate, reference, strict=True)
    )


def _compare_backward(
    candidate_outputs: tuple[torch.Tensor, ...],
    reference_outputs: tuple[torch.Tensor, ...],
    candidate_inputs: list[torch.Tensor],
    reference_inputs: list[torch.Tensor],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[Comparison, ...]:
    if len(candidate_outputs) != len(reference_outputs):
        return ()
    candidate_cotangents = tuple(torch.ones_like(output) for output in candidate_outputs)
    reference_cotangents = tuple(torch.ones_like(output) for output in reference_outputs)
    candidate_gradients = torch.autograd.grad(
        candidate_outputs,
        candidate_inputs,
        grad_outputs=candidate_cotangents,
        allow_unused=True,
    )
    reference_gradients = torch.autograd.grad(
        reference_outputs,
        reference_inputs,
        grad_outputs=reference_cotangents,
        allow_unused=True,
    )
    comparisons: list[Comparison] = []
    for observed, expected in zip(candidate_gradients, reference_gradients, strict=True):
        if observed is None or expected is None:
            comparisons.append(
                Comparison(
                    passed=observed is None and expected is None,
                    max_absolute_error=0.0 if observed is expected else float("inf"),
                    max_relative_error=0.0 if observed is expected else float("inf"),
                    mean_absolute_error=0.0 if observed is expected else float("inf"),
                    note=(
                        f"gradient presence mismatch: candidate={observed is not None}, "
                        f"reference={expected is not None}"
                    ),
                )
            )
        else:
            comparisons.append(
                compare_tensors(
                    observed,
                    expected,
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
            )
    return tuple(comparisons)


def _issues_from_comparisons(
    forward: tuple[Comparison, ...], backward: tuple[Comparison, ...]
) -> tuple[NablaIssue, ...]:
    issues: list[NablaIssue] = []
    if any(not comparison.passed for comparison in forward):
        worst = max(forward, key=lambda item: item.max_absolute_error)
        issues.append(
            NablaIssue(
                code="NG3001",
                category="FORWARD_MISMATCH",
                severity=Severity.HIGH,
                message="Candidate forward output differs from the reference.",
                evidence=worst.to_dict(),
                suggestion="Inspect the reported element and reproduce with the recorded seed.",
            )
        )
    if any(not comparison.passed for comparison in backward):
        worst = max(backward, key=lambda item: item.max_absolute_error)
        issues.append(
            NablaIssue(
                code="NG3002",
                category="BACKWARD_MISMATCH",
                severity=Severity.CRITICAL,
                message="Candidate VJP differs from reference autograd.",
                evidence=worst.to_dict(),
                suggestion="Check the custom backward formula and saved forward values.",
            )
        )
    return tuple(issues)


def _callable_name(value: Callable[..., Any]) -> str:
    owner = getattr(value, "__self__", None)
    if isinstance(owner, type):
        method_name = getattr(value, "__name__", type(value).__name__)
        return f"{owner.__module__}.{owner.__qualname__}.{method_name}"
    module = getattr(value, "__module__", None)
    name = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
    return f"{module}.{name}" if module else str(name)
