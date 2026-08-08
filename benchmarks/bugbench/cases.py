"""Deliberately broken ML programs used by the versioned BugBench corpus."""

from __future__ import annotations

import statistics
import tempfile
import time
from collections.abc import Callable
from typing import Any

import torch

import nablaguard as ng
from nablaguard.benchmark import BugBenchObservation, CaseContext
from nablaguard.contracts import ContractContext


class _UnaryModel(torch.nn.Module):
    def __init__(self, name: str, function: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.add_module(name, _FunctionModule(function))
        self._name = name

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.get_submodule(self._name)(value)


class _FunctionModule(torch.nn.Module):
    def __init__(self, function: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.function = function

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.function(value)


class _WrongSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(value)
        return value.square()

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> torch.Tensor:
        (value,) = ctx.saved_tensors
        return gradient * value


class _MissingGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor) -> torch.Tensor:
        del ctx
        return value.square()

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> None:
        del ctx, gradient
        return None


class _WrongBroadcastGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        del ctx
        return value + bias

    @staticmethod
    def backward(
        ctx: Any, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del ctx
        return gradient, gradient[0]


def fp16_overflow(context: CaseContext) -> BugBenchObservation:
    del context
    model = _UnaryModel("unstable_exp", torch.exp)
    with ng.guard(model, mode="standard", capture_source=False) as monitor:
        model(torch.tensor([12.0], dtype=torch.float16))
    return _issue_observation(monitor.issues, "OVERFLOW_RISK", stage="forward")


def unstable_softmax(context: CaseContext) -> BugBenchObservation:
    del context

    def naive_softmax(value: torch.Tensor) -> torch.Tensor:
        numerator = torch.exp(value)
        return numerator / numerator.sum(dim=-1, keepdim=True)

    model = _UnaryModel("softmax", naive_softmax)
    with ng.guard(model, mode="standard", capture_source=False) as monitor:
        model(torch.tensor([[100.0, 99.0]], dtype=torch.float32))
    return _issue_observation(monitor.issues, "OVERFLOW_RISK", stage="forward")


def catastrophic_cancellation(context: CaseContext) -> BugBenchObservation:
    del context
    model = _UnaryModel("reduction", lambda value: value.sum())
    with ng.guard(model, mode="standard", capture_source=False) as monitor:
        model(torch.tensor([1.0e16, 1.0, -1.0e16], dtype=torch.float64))
    return _issue_observation(monitor.issues, "NUMERICAL_CANCELLATION", stage="forward")


def stable_numerics_control(context: CaseContext) -> BugBenchObservation:
    del context
    value = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    def workload() -> None:
        torch.logsumexp(value, dim=0)

    def guarded() -> None:
        with ng.guard(mode="standard", capture_source=False):
            workload()

    baseline = _median_runtime(workload)
    instrumented = _median_runtime(guarded)
    with ng.guard(mode="standard", capture_source=False) as monitor:
        workload()
    return BugBenchObservation(
        bool(monitor.issues),
        category=monitor.issues[0].category if monitor.issues else None,
        module=monitor.issues[0].module_path if monitor.issues else None,
        stage="forward" if monitor.issues else None,
        evidence={"issue_count": len(monitor.issues)},
        baseline_seconds=baseline,
        instrumented_seconds=instrumented,
    )


def incorrect_custom_backward(context: CaseContext) -> BugBenchObservation:
    result = ng.check.operator(
        candidate=_WrongSquare.apply,
        reference=lambda value: value.square(),
        inputs=[ng.tensor(shape=(16,), dtype=torch.float64)],
        seed=context.seed,
    )
    return _operator_observation(result, "BACKWARD_MISMATCH", stage="backward")


def missing_gradient(context: CaseContext) -> BugBenchObservation:
    result = ng.check.operator(
        candidate=_MissingGradient.apply,
        reference=lambda value: value.square(),
        inputs=[ng.tensor(shape=(8,), dtype=torch.float64)],
        seed=context.seed,
    )
    return _operator_observation(result, "MISSING_GRADIENT", stage="backward")


def detached_computation(context: CaseContext) -> BugBenchObservation:
    def detached_square(value: torch.Tensor) -> torch.Tensor:
        return value.detach().square()

    try:
        result = ng.check.operator(
            candidate=detached_square,
            reference=lambda value: value.square(),
            inputs=[ng.tensor(shape=(8,), dtype=torch.float64)],
            seed=context.seed,
        )
    except RuntimeError as error:
        return BugBenchObservation(
            False,
            evidence={
                "observed": "operator checker raised instead of returning a correctness issue",
                "exception_type": type(error).__name__,
            },
        )
    return _operator_observation(result, "MISSING_GRADIENT", stage="backward")


def incorrect_broadcast_gradient(context: CaseContext) -> BugBenchObservation:
    result = ng.check.operator(
        candidate=_WrongBroadcastGradient.apply,
        reference=lambda value, bias: value + bias,
        inputs=[
            ng.tensor(shape=(4, 3), dtype=torch.float64),
            ng.tensor(shape=(3,), dtype=torch.float64),
        ],
        seed=context.seed,
    )
    return _operator_observation(result, "BACKWARD_MISMATCH", stage="backward")


def correct_autograd_control(context: CaseContext) -> BugBenchObservation:
    result = ng.check.operator(
        candidate=lambda value: value.square(),
        reference=lambda value: value.square(),
        inputs=[ng.tensor(shape=(8,), dtype=torch.float64)],
        seed=context.seed,
    )
    issue = result.issues[0] if result.issues else None
    return BugBenchObservation(
        not result.passed,
        category=issue.category if issue else None,
        stage="backward" if issue else None,
        evidence={"operator_passed": result.passed},
    )


def gradient_explosion(context: CaseContext) -> BugBenchObservation:
    del context
    contract = ng.contracts.gradient.norm(max=100.0)
    issue = contract.evaluate(
        ContractContext(
            gradients={"linear.weight": torch.tensor([1000.0])},
            module_path="linear.weight",
            operation="backward",
        )
    )
    return BugBenchObservation(
        issue is not None,
        category=issue.category if issue else None,
        module=issue.module_path if issue else None,
        stage=issue.operation if issue else None,
        evidence=issue.evidence if issue else {},
    )


def gradient_cancellation(context: CaseContext) -> BugBenchObservation:
    del context
    parameter = torch.tensor([1.0], requires_grad=True)
    positive = parameter.sum()
    negative = -0.99 * parameter.sum()
    with ng.trace.losses(
        {"positive": positive, "negative": negative},
        parameters=[parameter],
        cancellation_threshold=0.9,
    ) as trace:
        report = trace.report(parameter, name="classifier.weight")
    issue = next(
        (value for value in report.issues if value.category == "GRADIENT_CANCELLATION"), None
    )
    return BugBenchObservation(
        issue is not None,
        category=issue.category if issue else None,
        module=report.parameter_name if issue else None,
        stage="backward" if issue else None,
        evidence=issue.evidence if issue else {},
    )


def aligned_gradients_control(context: CaseContext) -> BugBenchObservation:
    del context
    parameter = torch.tensor([1.0], requires_grad=True)
    first = parameter.sum()
    second = 2.0 * parameter.sum()
    with ng.trace.losses(
        {"first": first, "second": second}, parameters=[parameter]
    ) as trace:
        report = trace.report(parameter, name="classifier.weight")
    issue = report.issues[0] if report.issues else None
    return BugBenchObservation(
        issue is not None,
        category=issue.category if issue else None,
        module=report.parameter_name if issue else None,
        stage="backward" if issue else None,
        evidence={"cancellation": report.cancellation},
    )


def stride_sensitive_kernel(context: CaseContext) -> BugBenchObservation:
    base = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    value = base.transpose(0, 1)

    def stride_kernel(input_value: torch.Tensor) -> torch.Tensor:
        contiguous_stride = (input_value.shape[1], 1)
        return torch.as_strided(input_value, input_value.shape, contiguous_stride)

    result = ng.check.operator(
        candidate=stride_kernel,
        reference=lambda input_value: input_value,
        inputs=[value],
        seed=context.seed,
        check_backward=False,
    )
    return _operator_observation(result, "FORWARD_MISMATCH", stage="forward")


def wrong_boundary_mask(context: CaseContext) -> BugBenchObservation:
    def masked_copy_kernel(value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] >= 17:
            output = value.clone()
            output[..., -1] = 0
            return output
        return value.clone()

    strategy = ng.tensor(
        shape=ng.shapes(ranks=(1,), dimensions=(32,)),
        dtype=[torch.float64],
        distribution=["ones"],
        layout=["contiguous"],
    )
    result = ng.check.fuzz(
        candidate=masked_copy_kernel,
        reference=lambda value: value,
        inputs=[strategy],
        trials=12,
        seed=context.seed,
    )
    if not result.failures:
        return BugBenchObservation(False, evidence={"cases_run": result.cases_run})
    failure = result.failures[0]
    issue = next(
        (value for value in failure.issues if value.category == "FORWARD_MISMATCH"), None
    )
    operator_result = failure.operator_result
    return BugBenchObservation(
        issue is not None,
        category=issue.category if issue else None,
        module=operator_result.candidate_name if issue and operator_result else None,
        stage="forward" if issue else None,
        evidence={"seed": failure.seed, "minimization_steps": list(failure.minimization_steps)},
        original_failure_size=failure.original_specs[0].shape[-1],
        minimized_failure_size=failure.minimal_specs[0].shape[-1],
    )


def contiguous_kernel_control(context: CaseContext) -> BugBenchObservation:
    result = ng.check.operator(
        candidate=lambda value: value.clone(),
        reference=lambda value: value,
        inputs=[ng.tensor(shape=(17,), dtype=torch.float64)],
        seed=context.seed,
    )
    issue = result.issues[0] if result.issues else None
    return BugBenchObservation(
        not result.passed,
        category=issue.category if issue else None,
        stage="forward" if issue else None,
        evidence={"operator_passed": result.passed},
    )


def bf16_precision_degradation(context: CaseContext) -> BugBenchObservation:
    del context
    model = _UnaryModel("reduction", lambda value: (value[0] + value[1]) + value[2])
    values = torch.tensor([10000.0, 1.0, -10000.0], dtype=torch.float64)
    result = ng.precision.audit(
        model,
        values,
        candidate_dtypes=(torch.bfloat16,),
        reference_dtype=torch.float64,
        max_relative_error=1.0e-6,
        absolute_tolerance=1.0e-8,
    )
    issue = next(
        (
            value
            for value in result.issues
            if value.category == "PRECISION_BUDGET_EXCEEDED"
            and value.module_path == "reduction"
        ),
        None,
    )
    return BugBenchObservation(
        issue is not None,
        category=issue.category if issue else None,
        module=issue.module_path if issue else None,
        stage="forward" if issue else None,
        evidence=issue.evidence if issue else {},
    )


def nondeterministic_operator(context: CaseContext) -> BugBenchObservation:
    state = {"calls": 0}

    def stateful(value: torch.Tensor) -> torch.Tensor:
        state["calls"] += 1
        return value + state["calls"]

    result = ng.check.operator(
        candidate=stateful,
        reference=lambda value: value + 1,
        inputs=[ng.tensor(shape=(4,), dtype=torch.float64)],
        seed=context.seed,
        check_backward=False,
        check_determinism=True,
    )
    return _operator_observation(result, "NONDETERMINISTIC_OPERATOR", stage="forward")


def missing_rng_restoration(context: CaseContext) -> BugBenchObservation:
    def corrupting_random_op(value: torch.Tensor) -> torch.Tensor:
        torch.seed()
        return value + torch.rand_like(value)

    result = ng.check.operator(
        candidate=corrupting_random_op,
        reference=lambda value: value,
        inputs=[ng.tensor(shape=(4,), dtype=torch.float64)],
        seed=context.seed,
        check_backward=False,
        check_determinism=True,
    )
    return _operator_observation(result, "NONDETERMINISTIC_OPERATOR", stage="forward")


def distributed_rank_divergence(context: CaseContext) -> BugBenchObservation:
    del context
    return BugBenchObservation(
        False,
        skip_reason=(
            "NablaGuard has no distributed detector yet; this ground truth remains an explicit "
            "coverage gap and is not simulated as supported."
        ),
    )


def dataloader_state_mismatch(context: CaseContext) -> BugBenchObservation:
    del context

    class StatefulDataset(torch.utils.data.IterableDataset[torch.Tensor]):
        def __init__(self) -> None:
            self.offset = 0

        def __iter__(self):
            self.offset += 1
            yield torch.tensor(self.offset)

    dataset = StatefulDataset()
    captured = int(next(iter(dataset)).item())
    replayed = int(next(iter(dataset)).item())
    with tempfile.TemporaryDirectory() as directory:
        model = torch.nn.Identity()
        with ng.capture(model, root=directory, run_id="data-state") as recorder:
            recorder.record_step(
                step=1,
                data_state={"dataset_id": "StatefulDataset", "row": captured},
                batch_indices=[captured],
            )
        replay_result = ng.replay(
            recorder.run_path,
            model=torch.nn.Identity(),
            step_fn=lambda step, metadata: ng.ReplayObservation(
                tensors={},
                data_state={"dataset_id": "StatefulDataset", "row": replayed},
                batch_indices=(replayed,),
            ),
            to_step=1,
        )
    issue = next(
        (
            value
            for value in replay_result.issues
            if value.category == "DATALOADER_STATE_MISMATCH"
        ),
        None,
    )
    return BugBenchObservation(
        issue is not None,
        category=issue.category if issue else None,
        module=issue.module_path if issue else None,
        stage=issue.operation if issue else None,
        evidence={
            "captured_first_row": captured,
            "replayed_first_row": replayed,
            "observed_mismatch": captured != replayed,
            "replay_status": replay_result.steps[0].status,
        },
    )


def _operator_observation(result: Any, category: str, *, stage: str) -> BugBenchObservation:
    issue = next((value for value in result.issues if value.category == category), None)
    return BugBenchObservation(
        issue is not None,
        category=issue.category if issue else None,
        module=result.candidate_name if issue else None,
        stage=stage if issue else None,
        evidence=issue.evidence if issue else {"available_categories": [
            value.category for value in result.issues
        ]},
    )


def _issue_observation(
    issues: tuple[Any, ...], category: str, *, stage: str
) -> BugBenchObservation:
    issue = next((value for value in issues if value.category == category), None)
    return BugBenchObservation(
        issue is not None,
        category=issue.category if issue else None,
        module=issue.module_path if issue else None,
        stage=stage if issue else None,
        evidence=issue.evidence if issue else {"available_categories": [
            value.category for value in issues
        ]},
    )


def _median_runtime(callback: Callable[[], None], repeats: int = 5) -> float:
    callback()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        callback()
        samples.append(time.perf_counter() - started)
    return max(statistics.median(samples), 1.0e-12)
