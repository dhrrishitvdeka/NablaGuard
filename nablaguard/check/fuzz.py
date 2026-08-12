"""Deterministic tensor-aware fuzzing for differentiable operators."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from nablaguard.artifact import ArtifactPolicy
from nablaguard.core import NablaIssue, Severity
from nablaguard.core.session import emit_issue

from .artifacts import write_failure_artifact
from .fuzz_results import FuzzFailure, FuzzResult
from .isolation import leaf_copy
from .minimizer import minimize as minimize_specs
from .properties import Property
from .results import Comparison, OperatorCheckResult
from .runner import operator
from .specs import TensorSpec, TensorStrategy


@dataclass(slots=True)
class _Evaluation:
    valid: bool
    failed: bool
    reason: str
    inputs: tuple[torch.Tensor, ...]
    operator_result: OperatorCheckResult | None = None
    property_results: tuple[tuple[str, Comparison], ...] = ()
    issues: tuple[NablaIssue, ...] = ()


def fuzz(
    *,
    candidate: Callable[..., Any],
    inputs: Sequence[TensorSpec | TensorStrategy],
    reference: Callable[..., Any] | None = None,
    properties: Sequence[Property] = (),
    trials: int = 100,
    seed: int = 81927183,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-4,
    check_backward: bool = True,
    minimize: bool = True,
    max_minimization_attempts: int = 100,
    max_failures: int = 1,
    artifact_dir: str | Path | None = None,
    artifact_raw_tensors: bool = False,
    artifact_max_size: str | int = "500MB",
    artifact_max_tensors: int = 16,
) -> FuzzResult:
    """Explore shape, dtype, layout, and value-distribution combinations.

    The run is deterministic under ``seed``. Cases unsupported by the reference
    are counted as skipped rather than blamed on the candidate. Failure
    minimization re-executes every proposed recipe and reports only a minimal
    *known* failure within the explicit attempt budget.
    """

    if reference is None and not properties:
        raise ValueError("fuzz requires a reference, at least one property, or both")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if max_failures <= 0:
        raise ValueError("max_failures must be positive")
    started = time.perf_counter()
    chooser = random.Random(seed)
    failures: list[FuzzFailure] = []
    cases_run = 0
    skipped = 0

    for trial in range(trials):
        specs = tuple(_sample(recipe, chooser) for recipe in inputs)
        trial_seed = chooser.randrange(0, 2**63)
        evaluation = _evaluate(
            candidate,
            reference,
            tuple(properties),
            specs,
            seed=trial_seed,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            check_backward=check_backward,
        )
        if not evaluation.valid:
            skipped += 1
            continue
        cases_run += 1
        if not evaluation.failed:
            continue

        minimized = specs
        steps: tuple[str, ...] = ()
        if minimize:

            def still_fails(
                candidate_specs: tuple[TensorSpec, ...], frozen_seed: int = trial_seed
            ) -> bool:
                outcome = _evaluate(
                    candidate,
                    reference,
                    tuple(properties),
                    candidate_specs,
                    seed=frozen_seed,
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                    check_backward=check_backward,
                )
                return outcome.valid and outcome.failed

            minimized_result = minimize_specs(
                specs, still_fails, max_attempts=max_minimization_attempts
            )
            minimized = minimized_result.specs
            steps = minimized_result.accepted_steps

        failure = FuzzFailure(
            trial=trial,
            seed=trial_seed,
            reason=evaluation.reason,
            original_specs=specs,
            minimal_specs=minimized,
            minimization_steps=steps,
            operator_result=evaluation.operator_result,
            property_results=evaluation.property_results,
            issues=evaluation.issues,
        )
        if artifact_dir is not None:
            try:
                minimal_inputs = _generate_inputs(minimized, trial_seed)
                failure.artifact_path = write_failure_artifact(
                    Path(artifact_dir),
                    metadata=failure.to_dict(),
                    inputs=evaluation.inputs,
                    minimized_inputs=minimal_inputs,
                    policy=ArtifactPolicy.create(
                        raw_tensors=artifact_raw_tensors,
                        max_size=artifact_max_size,
                        max_stored_tensors=artifact_max_tensors,
                    ),
                )
            except Exception as error:  # noqa: BLE001 — never drop a found failure
                failure.artifact_error = f"{type(error).__name__}: {error}"
        failures.append(failure)
        for issue in evaluation.issues:
            emit_issue(issue)
        if len(failures) >= max_failures:
            break

    return FuzzResult(
        seed=seed,
        requested_trials=trials,
        cases_run=cases_run,
        skipped_cases=skipped,
        failures=tuple(failures),
        elapsed_seconds=time.perf_counter() - started,
    )


def _sample(recipe: TensorSpec | TensorStrategy, chooser: random.Random) -> TensorSpec:
    return recipe.sample(chooser) if isinstance(recipe, TensorStrategy) else recipe


def _evaluate(
    candidate: Callable[..., Any],
    reference: Callable[..., Any] | None,
    properties: tuple[Property, ...],
    specs: tuple[TensorSpec, ...],
    *,
    seed: int,
    absolute_tolerance: float,
    relative_tolerance: float,
    check_backward: bool,
) -> _Evaluation:
    generated = _generate_inputs(specs, seed)
    operator_result: OperatorCheckResult | None = None
    issues: list[NablaIssue] = []
    reasons: list[str] = []

    if reference is not None:
        try:
            operator_result = operator(
                candidate=candidate,
                reference=reference,
                inputs=generated,
                seed=seed,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                check_backward=check_backward,
                _emit_issues=False,
            )
        except Exception as candidate_error:
            try:
                reference(*(leaf_copy(value) for value in generated))
            except Exception:
                return _Evaluation(False, False, "reference rejected generated case", generated)
            issue = NablaIssue(
                code="NG3006",
                category="OPERATOR_EXCEPTION",
                severity=Severity.HIGH,
                message="Candidate raised for an input accepted by the reference.",
                evidence={
                    "exception_type": type(candidate_error).__name__,
                    "exception": str(candidate_error),
                },
                reproduction={"seed": seed},
            )
            issues.append(issue)
            reasons.append(f"candidate exception: {type(candidate_error).__name__}")
        else:
            if not operator_result.passed:
                issues.extend(operator_result.issues)
                reasons.append("operator comparison failed")
    else:
        try:
            candidate(*(leaf_copy(value) for value in generated))
        except Exception as error:
            issue = NablaIssue(
                code="NG3006",
                category="OPERATOR_EXCEPTION",
                severity=Severity.HIGH,
                message="Candidate raised while evaluating a property case.",
                evidence={"exception_type": type(error).__name__, "exception": str(error)},
                reproduction={"seed": seed},
            )
            issues.append(issue)
            reasons.append(f"candidate exception: {type(error).__name__}")

    property_results: list[tuple[str, Comparison]] = []
    for invariant in properties:
        try:
            comparison = invariant.evaluate(
                tuple(leaf_copy(value) for value in generated),
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
        except Exception as error:
            comparison = Comparison(
                False,
                float("inf"),
                float("inf"),
                float("inf"),
                note=f"property raised {type(error).__name__}: {error}",
            )
        property_results.append((invariant.name, comparison))
        if not comparison.passed:
            issues.append(
                NablaIssue(
                    code="NG3005",
                    category="PROPERTY_VIOLATION",
                    severity=Severity.HIGH,
                    message=f"Differentiable property {invariant.name!r} failed.",
                    evidence=comparison.to_dict(),
                    reproduction={"seed": seed},
                )
            )
            reasons.append(f"property {invariant.name!r} failed")

    return _Evaluation(
        valid=True,
        failed=bool(issues),
        reason="; ".join(reasons),
        inputs=generated,
        operator_result=operator_result,
        property_results=tuple(property_results),
        issues=tuple(issues),
    )


def _generate_inputs(specs: tuple[TensorSpec, ...], seed: int) -> tuple[torch.Tensor, ...]:
    values: list[torch.Tensor] = []
    for index, spec in enumerate(specs):
        generator = torch.Generator(device=spec.device)
        generator.manual_seed(seed + index)
        # Keep spec.requires_grad. detach() would drop it and silently disable
        # every default backward / VJP comparison in fuzz trials.
        values.append(spec.generate(generator))
    return tuple(values)
