"""Explicit, module-hook, and selective dispatch-level numerical monitoring."""

from __future__ import annotations

import inspect
import weakref
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import torch

from nablaguard.contracts import Contract, ContractContext
from nablaguard.core import (
    NablaConfig,
    NablaIssue,
    Session,
    Severity,
    SourceLocation,
    TensorEvent,
)
from nablaguard.core.config import Mode

from .dispatch import NumericalDispatchMode
from .numerical import (
    compare_shadow,
    exp_overflow_evidence,
    reduction_cancellation,
    tensor_inputs,
    tensor_outputs,
)
from .shadow import REGISTRY
from .statistics import TensorStatistics, compute_statistics


@dataclass(slots=True)
class Guard(Session):
    """Bounded numerical guard with explicit cost controls.

    Standard and deep modes inspect a curated set of sensitive eager ATen
    operations. Deep mode enables higher-precision shadow execution by default.
    Light mode uses explicit observations and module boundaries only.
    """

    model: torch.nn.Module | None = None
    modules: tuple[str, ...] | None = None
    exclude: tuple[str, ...] = ()
    dispatch: bool = True
    shadow: bool = False
    shadow_dtype: torch.dtype = torch.float64
    max_relative_error: float = 1e-3
    max_absolute_error: float = 1e-6
    cancellation_threshold: float = 0.99
    operations: tuple[str, ...] | None = None
    capture_source: bool = True
    contracts: tuple[Contract, ...] = ()
    light_sample_elements: int = 1024
    _handles: list[Any] = field(default_factory=list, init=False, repr=False)
    _dispatch_mode: NumericalDispatchMode | None = field(default=None, init=False, repr=False)
    _analysis_depth: int = field(default=0, init=False, repr=False)
    _module_stack: list[str] = field(default_factory=list, init=False, repr=False)
    _tensor_producers: dict[int, str] = field(default_factory=dict, init=False, repr=False)
    _producer_refs: list[weakref.ref[torch.Tensor]] = field(
        default_factory=list, init=False, repr=False
    )

    @property
    def _inside_analysis(self) -> bool:
        return self._analysis_depth > 0

    def __enter__(self) -> Guard:
        # Explicit base calls avoid CPython's zero-argument super limitation
        # with dataclass(slots=True) class replacement.
        Session.__enter__(self)
        if self.model is not None:
            self._install_hooks()
        if self.dispatch and self.config.mode != "light":
            self._dispatch_mode = NumericalDispatchMode(self)
            self._dispatch_mode.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._dispatch_mode is not None:
            self._dispatch_mode.__exit__(exc_type, exc, traceback)
            self._dispatch_mode = None
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._module_stack.clear()
        self._tensor_producers.clear()
        self._producer_refs.clear()
        Session.__exit__(self, exc_type, exc, traceback)

    def observe(
        self, tensor: torch.Tensor, *, operation: str = "observe", module_path: str | None = None
    ) -> TensorEvent:
        """Summarize one tensor and emit evidence-backed numerical issues."""

        return self._record_tensor(tensor, operation=operation, module_path=module_path)

    def format(self) -> str:
        """Render issues and bounded-event accounting for the terminal."""

        from nablaguard.report.console import format_session

        return format_session(self)

    def print(self) -> None:
        """Print this guard's report."""

        print(self.format())

    def _record_tensor(
        self,
        tensor: torch.Tensor,
        *,
        operation: str,
        module_path: str | None,
        source_location: SourceLocation | None = None,
        tags: dict[str, Any] | None = None,
    ) -> TensorEvent:
        self._analysis_depth += 1
        try:
            statistics = compute_statistics(
                tensor,
                max_samples=(
                    self.light_sample_elements if self.config.mode == "light" else None
                ),
            )
        finally:
            self._analysis_depth -= 1
        event = _event(
            tensor,
            statistics,
            operation=operation,
            module_path=module_path,
            source_location=source_location,
            tags={
                **(tags or {}),
                "statistics_sampled": statistics.sampled_elements < statistics.total_elements,
                "statistics_sampled_elements": statistics.sampled_elements,
                "statistics_total_elements": statistics.total_elements,
            },
        )
        self.emit_event(event)
        self._remember_producer(tensor, event.event_id)
        context = ContractContext(
            tensor=tensor,
            module_path=module_path,
            operation=operation,
        )
        for assertion in self.contracts:
            assertion.evaluate(context)
        if statistics.nan_count or statistics.inf_count:
            self.emit_issue(
                NablaIssue(
                    code="NG1001",
                    category="NAN_DETECTED" if statistics.nan_count else "INF_DETECTED",
                    severity=Severity.CRITICAL,
                    message="A tensor contains non-finite values.",
                    module_path=module_path,
                    operation=operation,
                    source_location=source_location,
                    evidence={
                        "nan_count": statistics.nan_count,
                        "inf_count": statistics.inf_count,
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                    },
                    suggestion="Inspect the earliest emitted non-finite event and its inputs.",
                )
            )
        threshold = self.config.extreme_value_threshold
        if (
            threshold is not None
            and statistics.abs_max is not None
            and statistics.abs_max > threshold
        ):
            self.emit_issue(
                NablaIssue(
                    code="NG1003",
                    category="OVERFLOW_RISK",
                    severity=Severity.HIGH,
                    message="A finite tensor exceeded the configured magnitude threshold.",
                    module_path=module_path,
                    operation=operation,
                    source_location=source_location,
                    evidence={
                        "abs_max": statistics.abs_max,
                        "threshold": threshold,
                        "dtype": str(tensor.dtype),
                    },
                    suggestion=(
                        "Verify that this range is expected for the tensor dtype and next "
                        "operation."
                    ),
                )
            )
        return event

    def _observe_dispatch(
        self,
        operation: torch._ops.OpOverload,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        rule = REGISTRY.resolve(operation)
        operation_name = str(operation)
        if rule is None or not self._operation_selected(operation_name, operation._schema.name):
            return
        module_path = self._module_stack[-1] if self._module_stack else None
        if module_path is not None and not self._selected(module_path):
            return
        inputs = tensor_inputs((args, kwargs))
        outputs = tensor_outputs(output)
        if not outputs:
            return
        source = _source_location() if self.capture_source else None
        upstream = [
            producer for tensor in inputs if (producer := self._tensor_producers.get(id(tensor)))
        ]

        if operation._schema.name == "aten::exp" and inputs:
            evidence = exp_overflow_evidence(inputs[0])
            if evidence is not None:
                self.emit_issue(
                    NablaIssue(
                        code="NG1003",
                        category="OVERFLOW_RISK",
                        severity=Severity.HIGH,
                        message="Exponential input exceeds the finite range of its dtype.",
                        module_path=module_path,
                        operation=operation_name,
                        source_location=source,
                        evidence={**evidence, "dtype": str(inputs[0].dtype)},
                        suggestion=(
                            "Subtract a stable offset or execute this operation in higher "
                            "precision."
                        ),
                    )
                )

        for index, tensor in enumerate(outputs):
            self._record_tensor(
                tensor,
                operation=operation_name,
                module_path=module_path,
                source_location=source,
                tags={"output_index": index, "upstream_event_ids": upstream},
            )

        if operation._schema.name == "aten::sum" and inputs and outputs:
            cancellation = reduction_cancellation(inputs[0], outputs[0])
            if cancellation is not None and cancellation >= self.cancellation_threshold:
                self.emit_issue(
                    NablaIssue(
                        code="NG1002",
                        category="NUMERICAL_CANCELLATION",
                        severity=Severity.MEDIUM,
                        message=(
                            "A scalar sum discarded most component magnitude through cancellation."
                        ),
                        module_path=module_path,
                        operation=operation_name,
                        source_location=source,
                        evidence={
                            "definition": "1 - abs(sum(x)) / sum(abs(x))",
                            "cancellation": cancellation,
                            "threshold": self.cancellation_threshold,
                        },
                    )
                )

        if not self.shadow:
            return
        self._analysis_depth += 1
        try:
            shadow_output = rule(operation, args, kwargs, self.shadow_dtype)
        except (RuntimeError, TypeError, NotImplementedError) as error:
            self.emit_issue(
                NablaIssue(
                    code="NG1005",
                    category="SHADOW_UNSUPPORTED",
                    severity=Severity.LOW,
                    message="Higher-precision shadow execution could not evaluate this operation.",
                    module_path=module_path,
                    operation=operation_name,
                    source_location=source,
                    evidence={
                        "exception_type": type(error).__name__,
                        "exception": str(error),
                        "shadow_dtype": str(self.shadow_dtype),
                        "status": "UNKNOWN",
                    },
                    suggestion=(
                        "Treat this region as unverified for shadow precision; "
                        "register a custom shadow rule or exclude the operation."
                    ),
                )
            )
            return
        finally:
            self._analysis_depth -= 1
        shadow_outputs = tensor_outputs(shadow_output)
        for real_tensor, shadow_tensor in zip(outputs, shadow_outputs, strict=False):
            if not real_tensor.is_floating_point() or not shadow_tensor.is_floating_point():
                continue
            comparison = compare_shadow(real_tensor, shadow_tensor)
            unstable = comparison.finite_mismatch_count > 0 or (
                comparison.max_absolute_error > self.max_absolute_error
                and comparison.max_relative_error > self.max_relative_error
            )
            if unstable:
                self.emit_issue(
                    NablaIssue(
                        code="NG1002",
                        category="NUMERICAL_INSTABILITY",
                        severity=Severity.HIGH,
                        message=(
                            "Real output differs materially from higher-precision shadow execution."
                        ),
                        module_path=module_path,
                        operation=operation_name,
                        source_location=source,
                        evidence={
                            **comparison.to_dict(),
                            "real_dtype": str(real_tensor.dtype),
                            "shadow_dtype": str(self.shadow_dtype),
                            "absolute_tolerance": self.max_absolute_error,
                            "relative_tolerance": self.max_relative_error,
                        },
                        suggestion=(
                            "Confirm the error budget and consider promoting only this operation."
                        ),
                    )
                )

        if operation._schema.name == "aten::exp" and inputs and outputs and shadow_outputs:
            underflow_count = int(
                (
                    (outputs[0].detach() == 0)
                    & (shadow_outputs[0].detach().to(outputs[0].device) != 0)
                )
                .sum()
                .item()
            )
            if underflow_count:
                self.emit_issue(
                    NablaIssue(
                        code="NG1004",
                        category="UNDERFLOW_RISK",
                        severity=Severity.MEDIUM,
                        message=(
                            "Real exponential output underflowed where the shadow remained nonzero."
                        ),
                        module_path=module_path,
                        operation=operation_name,
                        source_location=source,
                        evidence={"underflow_count": underflow_count},
                    )
                )

    def _remember_producer(self, tensor: torch.Tensor, event_id: str) -> None:
        """Associate a live tensor with an event id; drop the map entry on GC."""

        key = id(tensor)
        self._tensor_producers[key] = event_id

        def _forget(
            _ref: weakref.ref[torch.Tensor],
            *,
            producer_key: int = key,
            producers: dict[int, str] = self._tensor_producers,
        ) -> None:
            producers.pop(producer_key, None)

        try:
            self._producer_refs.append(weakref.ref(tensor, _forget))
        except TypeError:
            # Some tensor subclasses reject weak references; keep a best-effort id map.
            pass

    def _operation_selected(self, overload_name: str, schema_name: str) -> bool:
        if self.operations is None:
            return True
        return any(
            fnmatch(overload_name, pattern) or fnmatch(schema_name, pattern)
            for pattern in self.operations
        )

    def _install_hooks(self) -> None:
        assert self.model is not None
        for name, module in self.model.named_modules():
            path = name or "<root>"
            if self.config.mode == "light" and self.modules is None and name:
                continue
            if not self._selected(path):
                continue
            self._handles.append(module.register_forward_pre_hook(self._pre_hook(path)))
            self._handles.append(module.register_forward_hook(self._hook(path), always_call=True))

    def _selected(self, name: str) -> bool:
        if any(fnmatch(name, pattern) for pattern in self.exclude):
            return False
        return self.modules is None or any(fnmatch(name, pattern) for pattern in self.modules)

    def _pre_hook(self, name: str) -> Callable[[torch.nn.Module, Any], None]:
        def hook(module: torch.nn.Module, inputs: Any) -> None:
            del module, inputs
            self._module_stack.append(name)

        return hook

    def _hook(self, name: str) -> Callable[[torch.nn.Module, Any, Any], None]:
        def hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
            del module, inputs
            for index, tensor in enumerate(_tensors(output)):
                self.observe(tensor, operation=f"forward_output[{index}]", module_path=name)
            if self._module_stack and self._module_stack[-1] == name:
                self._module_stack.pop()

        return hook


def guard(
    model: torch.nn.Module | None = None,
    *,
    mode: Mode = "standard",
    modules: Iterable[str] | None = None,
    exclude: Iterable[str] = (),
    operations: Iterable[str] | None = None,
    dispatch: bool = True,
    shadow: bool | None = None,
    shadow_dtype: torch.dtype = torch.float64,
    max_relative_error: float = 1e-3,
    max_absolute_error: float = 1e-6,
    cancellation_threshold: float = 0.99,
    capture_source: bool = True,
    max_events: int = 10_000,
    max_issues: int = 10_000,
    extreme_value_threshold: float | None = None,
    contracts: Iterable[Contract] = (),
    light_sample_elements: int = 1024,
) -> Guard:
    """Create a bounded numerical monitoring context."""

    if mode not in {"light", "standard", "deep"}:
        raise ValueError("mode must be 'light', 'standard', or 'deep'")
    if max_relative_error < 0 or max_absolute_error < 0:
        raise ValueError("shadow error tolerances must be non-negative")
    if not 0 <= cancellation_threshold <= 1:
        raise ValueError("cancellation_threshold must be in [0, 1]")
    if light_sample_elements <= 0:
        raise ValueError("light_sample_elements must be positive")
    if max_events <= 0 or max_issues <= 0:
        raise ValueError("max_events and max_issues must be positive")
    config = NablaConfig(
        mode=mode,
        max_events=max_events,
        max_issues=max_issues,
        extreme_value_threshold=extreme_value_threshold,
    )
    return Guard(
        config=config,
        model=model,
        modules=tuple(modules) if modules is not None else None,
        exclude=tuple(exclude),
        dispatch=dispatch,
        shadow=mode == "deep" if shadow is None else shadow,
        shadow_dtype=shadow_dtype,
        max_relative_error=max_relative_error,
        max_absolute_error=max_absolute_error,
        cancellation_threshold=cancellation_threshold,
        operations=tuple(operations) if operations is not None else None,
        capture_source=capture_source,
        contracts=tuple(contracts),
        light_sample_elements=light_sample_elements,
    )


def sanitize(*tensors: torch.Tensor, extreme_value_threshold: float | None = None) -> Guard:
    """Inspect explicit tensors and return the populated guard report."""

    monitor = guard(dispatch=False, extreme_value_threshold=extreme_value_threshold)
    with monitor:
        for index, tensor in enumerate(tensors):
            monitor.observe(tensor, operation=f"input[{index}]")
    return monitor


def _event(
    tensor: torch.Tensor,
    statistics: TensorStatistics,
    *,
    operation: str,
    module_path: str | None,
    source_location: SourceLocation | None = None,
    tags: dict[str, Any] | None = None,
) -> TensorEvent:
    return TensorEvent(
        operation=operation,
        module_path=module_path,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
        requires_grad=tensor.requires_grad,
        min_value=statistics.minimum,
        max_value=statistics.maximum,
        mean=statistics.mean,
        std=statistics.std,
        abs_max=statistics.abs_max,
        zero_fraction=statistics.zero_fraction,
        nan_count=statistics.nan_count,
        inf_count=statistics.inf_count,
        source_location=source_location,
        tags=tags or {},
    )


def _tensors(value: Any) -> list[torch.Tensor]:
    return list(tensor_outputs(value))


def _source_location() -> SourceLocation | None:
    frame = inspect.currentframe()
    if frame is None:
        return None
    frame = frame.f_back
    package_root = Path(__file__).resolve().parents[1]
    while frame is not None:
        filename = Path(frame.f_code.co_filename).resolve()
        in_torch_package = "site-packages" in filename.parts and "torch" in filename.parts
        if package_root not in filename.parents and not in_torch_package:
            return SourceLocation(str(filename), frame.f_lineno, frame.f_code.co_name)
        frame = frame.f_back
    return None
