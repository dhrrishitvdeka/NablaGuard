"""Forward and backward verification for PyTorch callables."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import torch

from nablaguard.artifact import ArtifactPolicy
from nablaguard.capture.rng import capture_rng_state, restore_rng_state
from nablaguard.core import NablaIssue, Severity
from nablaguard.core.session import emit_issue

from .advanced import (
    compare_determinism,
    compare_double_backward,
    compare_finite_difference,
    compare_jvp,
)
from .artifacts import write_failure_artifact
from .compare import compare_tensors
from .isolation import isolated_callable, leaf_copy
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
    vjp_cotangent: Literal["ones", "random"] = "ones",
    check_jvp: bool = False,
    check_double_backward: bool = False,
    check_finite_difference: bool = False,
    check_determinism: bool = False,
    finite_difference_epsilon: float = 1e-6,
    max_finite_difference_elements: int = 128,
    artifact_dir: str | Path | None = None,
    artifact_raw_tensors: bool = False,
    artifact_max_size: str | int = "500MB",
    artifact_max_tensors: int = 16,
    _emit_issues: bool = True,
) -> OperatorCheckResult:
    """Compare a candidate operator with a trusted PyTorch reference.

    Backward verification compares vector-Jacobian products under the same
    all-ones cotangent. A seed is always reported and input generation uses a
    private generator, so the experiment does not perturb global RNG state.
    """

    rng_state = capture_rng_state()
    try:
        return _operator_impl(
            candidate=candidate,
            reference=reference,
            inputs=inputs,
            seed=seed,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            check_backward=check_backward,
            vjp_cotangent=vjp_cotangent,
            check_jvp=check_jvp,
            check_double_backward=check_double_backward,
            check_finite_difference=check_finite_difference,
            check_determinism=check_determinism,
            finite_difference_epsilon=finite_difference_epsilon,
            max_finite_difference_elements=max_finite_difference_elements,
            artifact_dir=artifact_dir,
            artifact_policy=ArtifactPolicy.create(
                raw_tensors=artifact_raw_tensors,
                max_size=artifact_max_size,
                max_stored_tensors=artifact_max_tensors,
            ),
            _emit_issues=_emit_issues,
        )
    finally:
        restore_rng_state(rng_state)


def _operator_impl(
    *,
    candidate: Callable[..., Any],
    reference: Callable[..., Any],
    inputs: Sequence[TensorSpec | torch.Tensor],
    seed: int,
    absolute_tolerance: float,
    relative_tolerance: float,
    check_backward: bool,
    vjp_cotangent: Literal["ones", "random"],
    check_jvp: bool,
    check_double_backward: bool,
    check_finite_difference: bool,
    check_determinism: bool,
    finite_difference_epsilon: float,
    max_finite_difference_elements: int,
    artifact_dir: str | Path | None,
    artifact_policy: ArtifactPolicy,
    _emit_issues: bool,
) -> OperatorCheckResult:
    started = time.perf_counter()
    originals = [_materialize(value, seed + index) for index, value in enumerate(inputs)]
    candidate_inputs = [leaf_copy(value) for value in originals]
    reference_inputs = [leaf_copy(value) for value in originals]
    candidate_execution = isolated_callable(candidate)
    reference_execution = isolated_callable(reference)

    paired_rng_state = capture_rng_state()
    restore_rng_state(paired_rng_state)
    candidate_outputs = _as_tensor_tuple(candidate_execution(*candidate_inputs))
    restore_rng_state(paired_rng_state)
    reference_outputs = _as_tensor_tuple(reference_execution(*reference_inputs))
    restore_rng_state(paired_rng_state)
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
            cotangent=vjp_cotangent,
            seed=seed,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )

    jvp = (
        compare_jvp(
            candidate_execution,
            reference_execution,
            originals,
            seed=seed,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        if check_jvp
        else ()
    )
    double_backward = (
        compare_double_backward(
            candidate_execution,
            reference_execution,
            originals,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        if check_double_backward
        else ()
    )
    finite_difference = (
        compare_finite_difference(
            candidate_execution,
            originals,
            epsilon=finite_difference_epsilon,
            max_elements=max_finite_difference_elements,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        if check_finite_difference
        else ()
    )
    determinism = compare_determinism(candidate_execution, originals) if check_determinism else ()

    issues = _issues_from_comparisons(
        forward, backward, jvp, double_backward, finite_difference, determinism
    )
    if _emit_issues:
        for issue in issues:
            emit_issue(issue)

    result = OperatorCheckResult(
        candidate_name=_callable_name(candidate),
        reference_name=_callable_name(reference),
        seed=seed,
        forward=forward,
        backward=backward,
        jvp=jvp,
        double_backward=double_backward,
        finite_difference=finite_difference,
        determinism=determinism,
        issues=issues,
        elapsed_seconds=time.perf_counter() - started,
        metadata={
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "input_shapes": [list(value.shape) for value in originals],
            "input_dtypes": [str(value.dtype) for value in originals],
            "input_strides": [list(value.stride()) for value in originals],
            "input_requires_grad": [value.requires_grad for value in originals],
            "vjp_cotangent": vjp_cotangent,
        },
    )
    if not result.passed and artifact_dir is not None:
        try:
            result.artifact_path = write_failure_artifact(
                Path(artifact_dir),
                metadata=result.to_dict(),
                inputs=originals,
                policy=artifact_policy,
            )
        except Exception as error:  # noqa: BLE001 — never drop a completed failure
            result.artifact_error = f"{type(error).__name__}: {error}"
            result.metadata["artifact_error"] = result.artifact_error
    return result


def _materialize(value: TensorSpec | torch.Tensor, seed: int) -> torch.Tensor:
    if isinstance(value, TensorSpec):
        generator = torch.Generator(device=value.device)
        generator.manual_seed(seed)
        generated = value.generate(generator)
        return generated.detach().requires_grad_(generated.requires_grad)
    if isinstance(value, torch.Tensor):
        return leaf_copy(value)
    raise TypeError(f"inputs must contain TensorSpec or Tensor, got {type(value)!r}")


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
    cotangent: Literal["ones", "random"],
    seed: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[Comparison, ...]:
    if len(candidate_outputs) != len(reference_outputs):
        return ()
    if cotangent == "ones":
        candidate_cotangents = tuple(torch.ones_like(output) for output in candidate_outputs)
    elif cotangent == "random":
        candidate_cotangents = tuple(
            _seeded_cotangent(output, seed + index)
            for index, output in enumerate(candidate_outputs)
        )
    else:
        raise ValueError("vjp_cotangent must be 'ones' or 'random'")
    reference_cotangents = tuple(value.clone() for value in candidate_cotangents)
    candidate_gradients = _safe_gradients(
        candidate_outputs, candidate_inputs, candidate_cotangents
    )
    reference_gradients = _safe_gradients(
        reference_outputs, reference_inputs, reference_cotangents
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


def _safe_gradients(
    outputs: tuple[torch.Tensor, ...],
    inputs: list[torch.Tensor],
    cotangents: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor | None, ...]:
    active_inputs = [(index, value) for index, value in enumerate(inputs) if value.requires_grad]
    active_outputs = [
        (output, cotangent)
        for output, cotangent in zip(outputs, cotangents, strict=True)
        if output.requires_grad
    ]
    gradients: list[torch.Tensor | None] = [None] * len(inputs)
    if not active_inputs or not active_outputs:
        return tuple(gradients)
    calculated = torch.autograd.grad(
        tuple(output for output, _ in active_outputs),
        tuple(value for _, value in active_inputs),
        grad_outputs=tuple(cotangent for _, cotangent in active_outputs),
        allow_unused=True,
    )
    for (index, _), gradient in zip(active_inputs, calculated, strict=True):
        gradients[index] = gradient
    return tuple(gradients)


def _seeded_cotangent(output: torch.Tensor, seed: int) -> torch.Tensor:
    if not output.is_floating_point():
        return torch.ones_like(output)
    generator = torch.Generator(device=output.device).manual_seed(seed)
    return torch.randn(
        output.shape,
        dtype=output.dtype,
        device=output.device,
        generator=generator,
    )


def _issues_from_comparisons(
    forward: tuple[Comparison, ...],
    backward: tuple[Comparison, ...],
    jvp: tuple[Comparison, ...] = (),
    double_backward: tuple[Comparison, ...] = (),
    finite_difference: tuple[Comparison, ...] = (),
    determinism: tuple[Comparison, ...] = (),
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
        missing_gradient = any(
            comparison.note == "gradient presence mismatch: candidate=False, reference=True"
            for comparison in backward
        )
        issues.append(
            NablaIssue(
                code="NG3002",
                category="MISSING_GRADIENT" if missing_gradient else "BACKWARD_MISMATCH",
                severity=Severity.CRITICAL,
                message=(
                    "Candidate did not produce a gradient required by the reference."
                    if missing_gradient
                    else "Candidate VJP differs from reference autograd."
                ),
                evidence=worst.to_dict(),
                suggestion="Check the custom backward formula and saved forward values.",
            )
        )
    if any(not comparison.passed for comparison in jvp):
        worst = max(jvp, key=lambda item: item.max_absolute_error)
        issues.append(
            NablaIssue(
                code="NG3003",
                category="JVP_MISMATCH",
                severity=Severity.HIGH,
                message="Candidate JVP differs from the reference under a seeded tangent.",
                evidence=worst.to_dict(),
            )
        )
    if any(not comparison.passed for comparison in double_backward):
        worst = max(double_backward, key=lambda item: item.max_absolute_error)
        issues.append(
            NablaIssue(
                code="NG3002",
                category="DOUBLE_BACKWARD_MISMATCH",
                severity=Severity.HIGH,
                message="Candidate second-order VJP differs from the reference or is unsupported.",
                evidence=worst.to_dict(),
            )
        )
    if any(not comparison.passed for comparison in finite_difference):
        worst = max(finite_difference, key=lambda item: item.max_absolute_error)
        issues.append(
            NablaIssue(
                code="NG3002",
                category="FINITE_DIFFERENCE_MISMATCH",
                severity=Severity.HIGH,
                message="Candidate autograd VJP differs from central finite differences.",
                evidence=worst.to_dict(),
            )
        )
    if any(not comparison.passed for comparison in determinism):
        worst = max(determinism, key=lambda item: item.max_absolute_error)
        issues.append(
            NablaIssue(
                code="NG3004",
                category="NONDETERMINISTIC_OPERATOR",
                severity=Severity.HIGH,
                message=(
                    "Candidate outputs changed across identical-input, identical-RNG executions."
                ),
                evidence=worst.to_dict(),
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
