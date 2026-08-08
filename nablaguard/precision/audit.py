"""Experimental dtype auditing against a higher-precision model reference."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from typing import Any

import torch

from nablaguard.check.compare import compare_tensors
from nablaguard.core import NablaIssue, Severity


@dataclass(frozen=True, slots=True)
class DtypeMeasurement:
    """Observed module error for one candidate dtype."""

    dtype: str
    supported: bool
    passed: bool
    max_absolute_error: float | None
    max_relative_error: float | None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class PrecisionEntry:
    """Experimentally selected dtype for one observed module boundary."""

    module_path: str
    recommended_dtype: str
    measurements: tuple[DtypeMeasurement, ...]


@dataclass(frozen=True, slots=True)
class PrecisionAuditResult:
    """Bounded module-boundary precision experiment."""

    entries: tuple[PrecisionEntry, ...]
    reference_dtype: str
    max_relative_error: float
    absolute_tolerance: float
    skipped_modules: tuple[str, ...]
    issues: tuple[NablaIssue, ...]
    elapsed_seconds: float
    captured_elements: int
    max_capture_elements: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report."""

        return {
            "entries": [
                {
                    "module_path": entry.module_path,
                    "recommended_dtype": entry.recommended_dtype,
                    "measurements": [asdict(value) for value in entry.measurements],
                }
                for entry in self.entries
            ],
            "reference_dtype": self.reference_dtype,
            "max_relative_error": self.max_relative_error,
            "absolute_tolerance": self.absolute_tolerance,
            "skipped_modules": list(self.skipped_modules),
            "issues": [issue.to_dict() for issue in self.issues],
            "elapsed_seconds": self.elapsed_seconds,
            "captured_elements": self.captured_elements,
            "max_capture_elements": self.max_capture_elements,
        }

    def format(self) -> str:
        """Render a compact terminal precision table."""

        lines = [
            "NablaGuard precision audit",
            "=" * 32,
            f"Reference dtype: {self.reference_dtype}",
            f"Relative error budget: {self.max_relative_error:g}",
            "",
            "Module                              Recommended dtype",
            "-" * 58,
        ]
        for entry in self.entries:
            lines.append(f"{entry.module_path:<36}{entry.recommended_dtype}")
        if self.skipped_modules:
            lines.extend(["", f"Skipped by capture bound: {', '.join(self.skipped_modules)}"])
        lines.extend(
            [
                "",
                f"Captured elements: {self.captured_elements}/{self.max_capture_elements}",
                f"Measured audit time: {self.elapsed_seconds:.6f}s",
            ]
        )
        return "\n".join(lines)

    def print(self) -> None:
        """Print the terminal report."""

        print(self.format())


def audit(
    model: torch.nn.Module,
    inputs: torch.Tensor | Sequence[Any] | dict[str, Any],
    *,
    candidate_dtypes: Sequence[torch.dtype] = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ),
    reference_dtype: torch.dtype = torch.float64,
    max_relative_error: float = 1e-4,
    absolute_tolerance: float = 1e-7,
    modules: Iterable[str] | None = None,
    max_capture_elements: int = 1_000_000,
) -> PrecisionAuditResult:
    """Compare whole-model candidate dtypes against a high-precision reference.

    Recommendations are the first user-ordered candidate dtype whose observed
    module output stays within both tolerances. The experiment runs deep-copied
    models and never rewrites the original.
    """

    if max_relative_error < 0 or absolute_tolerance < 0:
        raise ValueError("error tolerances must be non-negative")
    if max_capture_elements <= 0:
        raise ValueError("max_capture_elements must be positive")
    started = time.perf_counter()
    patterns = tuple(modules) if modules is not None else None
    reference_model = copy.deepcopy(model).to(dtype=reference_dtype)
    reference_model.eval()
    reference_capture = _capture(
        reference_model,
        _cast_tree(inputs, reference_dtype),
        patterns=patterns,
        max_elements=max_capture_elements,
    )
    measurements: dict[str, list[DtypeMeasurement]] = {
        name: [] for name in reference_capture.outputs
    }
    issues: list[NablaIssue] = []

    ordered_dtypes = tuple(dict.fromkeys((*candidate_dtypes, reference_dtype)))
    for dtype in ordered_dtypes:
        if dtype == reference_dtype:
            for name in measurements:
                measurements[name].append(DtypeMeasurement(str(dtype), True, True, 0.0, 0.0))
            continue
        try:
            candidate_model = copy.deepcopy(model).to(dtype=dtype)
            candidate_model.eval()
            candidate_capture = _capture(
                candidate_model,
                _cast_tree(inputs, dtype),
                patterns=patterns,
                max_elements=max_capture_elements,
            )
        except (RuntimeError, TypeError, NotImplementedError) as error:
            for name in measurements:
                measurements[name].append(
                    DtypeMeasurement(str(dtype), False, False, None, None, str(error))
                )
            continue
        for name, reference_outputs in reference_capture.outputs.items():
            candidate_outputs = candidate_capture.outputs.get(name)
            if candidate_outputs is None or len(candidate_outputs) != len(reference_outputs):
                measurements[name].append(
                    DtypeMeasurement(
                        str(dtype), False, False, None, None, "module output was not captured"
                    )
                )
                continue
            comparisons = tuple(
                compare_tensors(
                    candidate,
                    reference,
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=max_relative_error,
                )
                for candidate, reference in zip(candidate_outputs, reference_outputs, strict=True)
            )
            measurements[name].append(
                DtypeMeasurement(
                    dtype=str(dtype),
                    supported=True,
                    passed=all(value.passed for value in comparisons),
                    max_absolute_error=max(value.max_absolute_error for value in comparisons),
                    max_relative_error=max(value.max_relative_error for value in comparisons),
                )
            )

    entries: list[PrecisionEntry] = []
    for name, values in measurements.items():
        selected = next((value.dtype for value in values if value.supported and value.passed), None)
        recommended = selected or str(reference_dtype)
        entries.append(PrecisionEntry(name, recommended, tuple(values)))
        if selected == str(reference_dtype) and any(value.supported for value in values[:-1]):
            issue = NablaIssue(
                code="NG1002",
                category="PRECISION_BUDGET_EXCEEDED",
                severity=Severity.MEDIUM,
                message="Lower-precision candidates exceeded the module output error budget.",
                module_path=name,
                evidence={
                    "recommended_dtype": recommended,
                    "max_relative_error": max_relative_error,
                    "absolute_tolerance": absolute_tolerance,
                },
            )
            issues.append(issue)
    skipped = tuple(sorted(reference_capture.skipped))
    return PrecisionAuditResult(
        entries=tuple(entries),
        reference_dtype=str(reference_dtype),
        max_relative_error=max_relative_error,
        absolute_tolerance=absolute_tolerance,
        skipped_modules=skipped,
        issues=tuple(issues),
        elapsed_seconds=time.perf_counter() - started,
        captured_elements=reference_capture.elements,
        max_capture_elements=max_capture_elements,
    )


@dataclass(slots=True)
class _Capture:
    outputs: dict[str, tuple[torch.Tensor, ...]]
    skipped: set[str]
    elements: int


def _capture(
    model: torch.nn.Module,
    inputs: torch.Tensor | Sequence[Any] | dict[str, Any],
    *,
    patterns: tuple[str, ...] | None,
    max_elements: int,
) -> _Capture:
    outputs: dict[str, tuple[torch.Tensor, ...]] = {}
    skipped: set[str] = set()
    elements = 0
    handles: list[Any] = []

    def make_hook(name: str) -> Callable[[torch.nn.Module, Any, Any], None]:
        def hook(module: torch.nn.Module, args: Any, output: Any) -> None:
            nonlocal elements
            del module, args
            tensors = _tensor_outputs(output)
            count = sum(value.numel() for value in tensors)
            if not tensors:
                return
            if elements + count > max_elements:
                skipped.add(name)
                return
            outputs[name] = tuple(value.detach().to(device="cpu") for value in tensors)
            elements += count

        return hook

    for module_name, module in model.named_modules():
        name = module_name or "<root>"
        if patterns is None or any(fnmatch(name, pattern) for pattern in patterns):
            handles.append(module.register_forward_hook(make_hook(name)))
    try:
        with torch.no_grad():
            _invoke(model, inputs)
    finally:
        for handle in handles:
            handle.remove()
    return _Capture(outputs, skipped, elements)


def _invoke(model: torch.nn.Module, inputs: Any) -> Any:
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


def _cast_tree(value: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.detach().to(dtype=dtype)
    if isinstance(value, tuple):
        return tuple(_cast_tree(item, dtype) for item in value)
    if isinstance(value, list):
        return [_cast_tree(item, dtype) for item in value]
    if isinstance(value, dict):
        return {key: _cast_tree(item, dtype) for key, item in value.items()}
    return value


def _tensor_outputs(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, torch.Tensor):
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(tensor for item in value for tensor in _tensor_outputs(item))
    if isinstance(value, dict):
        return tuple(tensor for item in value.values() for tensor in _tensor_outputs(item))
    return ()
