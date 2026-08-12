"""Bounded per-sample gradient analysis for selected model parameters."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from typing import Any

import torch

from nablaguard.capture.rng import capture_rng_state, restore_rng_state
from nablaguard.core import NablaIssue, Severity
from nablaguard.core.session import emit_issue


@dataclass(frozen=True, slots=True)
class SampleGradient:
    """Magnitude and direction evidence for one batch sample."""

    index: int
    norm: float
    magnitude_share: float
    cosine_to_batch: float


@dataclass(frozen=True, slots=True)
class SamplePair:
    """Pair of samples with nearly duplicate gradient direction."""

    left: int
    right: int
    cosine: float


@dataclass(frozen=True, slots=True)
class BatchGradientReport:
    """Per-sample geometry for selected parameters."""

    samples: tuple[SampleGradient, ...]
    dominant_samples: tuple[SampleGradient, ...]
    conflicting_samples: tuple[SampleGradient, ...]
    duplicate_pairs: tuple[SamplePair, ...]
    cancellation: float
    selected_parameters: tuple[str, ...]
    parameter_elements: int
    gradient_elements: int
    microbatch_size: int
    warnings: tuple[str, ...]
    issues: tuple[NablaIssue, ...]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report."""

        return {
            "samples": [asdict(value) for value in self.samples],
            "dominant_samples": [asdict(value) for value in self.dominant_samples],
            "conflicting_samples": [asdict(value) for value in self.conflicting_samples],
            "duplicate_pairs": [asdict(value) for value in self.duplicate_pairs],
            "cancellation": self.cancellation,
            "selected_parameters": list(self.selected_parameters),
            "parameter_elements": self.parameter_elements,
            "gradient_elements": self.gradient_elements,
            "microbatch_size": self.microbatch_size,
            "warnings": list(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
            "elapsed_seconds": self.elapsed_seconds,
        }

    def format(self) -> str:
        """Render a terminal batch-gradient report."""

        lines = [
            "Batch Gradient Report",
            "=" * 32,
            f"Selected parameters: {', '.join(self.selected_parameters)}",
            f"Gradient elements retained: {self.gradient_elements}",
            f"Cancellation: {self.cancellation:.1%}",
            "",
            "Dominant samples:",
        ]
        for sample in self.dominant_samples:
            lines.append(
                f"#{sample.index:<6}{sample.magnitude_share:>7.1%}  norm={sample.norm:.6g}"
            )
        lines.extend(["", "Conflicting samples:"])
        if self.conflicting_samples:
            for sample in self.conflicting_samples:
                lines.append(f"#{sample.index:<6}cosine={sample.cosine_to_batch:.6g}")
        else:
            lines.append("none")
        for warning in self.warnings:
            lines.extend(["", f"PERFORMANCE WARNING: {warning}"])
        return "\n".join(lines)

    def print(self) -> None:
        """Print the terminal report."""

        print(self.format())


def samples(
    model: torch.nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    batch: Any,
    *,
    parameters: Mapping[str, torch.Tensor] | Iterable[torch.Tensor] | None = None,
    layers: Iterable[str] | None = None,
    sample_indices: Sequence[int] | None = None,
    microbatch_size: int = 1,
    top_k: int = 5,
    conflict_threshold: float = 0.0,
    duplicate_threshold: float = 0.999,
    cancellation_threshold: float = 0.5,
    max_gradient_elements: int = 5_000_000,
    max_pairwise_samples: int = 128,
) -> BatchGradientReport:
    """Compute exact selected-parameter gradients for individual samples.

    ``batch`` may be an input tensor, ``(inputs, targets)``, or a mapping with
    ``inputs`` and optional ``targets``. A non-scalar loss with leading batch
    dimension enables shared microbatch forwards. Scalar reduced losses fall
    back to one forward per sample and report that cost.
    """

    started = time.perf_counter()
    if microbatch_size <= 0 or top_k <= 0:
        raise ValueError("microbatch_size and top_k must be positive")
    model_inputs, targets = _split_batch(batch)
    batch_size = _batch_size(model_inputs)
    indices = tuple(range(batch_size)) if sample_indices is None else tuple(sample_indices)
    if not indices or any(index < 0 or index >= batch_size for index in indices):
        raise ValueError("sample_indices must select valid batch rows")
    named_parameters = _select_parameters(model, parameters, layers)
    if not named_parameters:
        raise ValueError("no trainable parameters matched the selection")
    parameter_elements = sum(value.numel() for value in named_parameters.values())
    gradient_elements = parameter_elements * len(indices)
    if gradient_elements > max_gradient_elements:
        raise MemoryError(
            "per-sample gradient request exceeds max_gradient_elements: "
            f"{gradient_elements} > {max_gradient_elements}; select fewer samples or layers"
        )

    rng_state = capture_rng_state()
    buffer_state = {name: value.detach().clone() for name, value in model.named_buffers()}
    existing_gradients = {
        name: None if value.grad is None else value.grad.detach().clone()
        for name, value in named_parameters.items()
    }
    warnings = [
        "Per-sample analysis performs one VJP per selected sample and retains one "
        "selected-parameter gradient vector per sample."
    ]
    vectors: list[torch.Tensor] = []
    try:
        for start in range(0, len(indices), microbatch_size):
            chunk_indices = indices[start : start + microbatch_size]
            chunk_inputs = _slice_tree(model_inputs, chunk_indices)
            chunk_targets = _slice_tree(targets, chunk_indices) if targets is not None else None
            output = _invoke_model(model, chunk_inputs)
            losses = _invoke_loss(loss_fn, output, chunk_targets)
            if losses.numel() == 1 and len(chunk_indices) > 1:
                warnings.append(
                    "loss_fn returned a reduced scalar for a multi-sample microbatch; "
                    "falling back to individual forwards for exact per-sample gradients."
                )
                for index in chunk_indices:
                    vectors.append(
                        _single_sample_vector(
                            model,
                            loss_fn,
                            model_inputs,
                            targets,
                            index,
                            tuple(named_parameters.values()),
                        )
                    )
                continue
            per_sample_losses = _per_sample_losses(losses, len(chunk_indices))
            for offset, loss in enumerate(per_sample_losses):
                vectors.append(
                    _gradient_vector(
                        loss,
                        tuple(named_parameters.values()),
                        retain_graph=offset < len(per_sample_losses) - 1,
                    )
                )
    finally:
        restore_rng_state(rng_state)
        model_buffers = dict(model.named_buffers())
        with torch.no_grad():
            for name, value in buffer_state.items():
                model_buffers[name].copy_(value)
        for name, parameter in named_parameters.items():
            parameter.grad = existing_gradients[name]

    matrix = torch.stack(vectors)
    norms = torch.linalg.vector_norm(matrix, dim=1)
    combined = matrix.sum(dim=0)
    combined_norm = torch.linalg.vector_norm(combined)
    norm_sum = norms.sum()
    cancellation = (
        float((1 - combined_norm / norm_sum).clamp(0, 1).item())
        if float(norm_sum.item()) > 0
        else 0.0
    )
    cosines = _cosines_to_batch(matrix, norms, combined, combined_norm)
    sample_results = tuple(
        SampleGradient(
            index=index,
            norm=float(norm.item()),
            magnitude_share=float((norm / norm_sum).item()) if float(norm_sum.item()) else 0.0,
            cosine_to_batch=float(cosine.item()),
        )
        for index, norm, cosine in zip(indices, norms, cosines, strict=True)
    )
    dominant = tuple(
        sorted(sample_results, key=lambda value: value.magnitude_share, reverse=True)[:top_k]
    )
    conflicting = tuple(
        sorted(
            (
                value
                for value in sample_results
                if not math.isnan(value.cosine_to_batch)
                and value.cosine_to_batch < conflict_threshold
            ),
            key=lambda value: value.cosine_to_batch,
        )
    )
    duplicate_pairs = _duplicate_pairs(
        matrix,
        norms,
        indices,
        threshold=duplicate_threshold,
        max_samples=max_pairwise_samples,
    )
    issues: list[NablaIssue] = []
    if conflicting:
        issues.append(
            NablaIssue(
                code="NG2003",
                category="GRADIENT_CONFLICT",
                severity=Severity.MEDIUM,
                message="One or more sample gradients oppose the selected batch gradient.",
                evidence={
                    "samples": [value.index for value in conflicting],
                    "cosines": [value.cosine_to_batch for value in conflicting],
                },
            )
        )
    if cancellation >= cancellation_threshold:
        issues.append(
            NablaIssue(
                code="NG2004",
                category="GRADIENT_CANCELLATION",
                severity=Severity.MEDIUM,
                message="Selected per-sample gradients substantially cancel.",
                evidence={
                    "definition": "1 - norm(sum(g_i)) / sum(norm(g_i))",
                    "cancellation": cancellation,
                },
            )
        )
    for issue in issues:
        emit_issue(issue)
    if len(indices) > max_pairwise_samples:
        warnings.append(
            f"Duplicate-direction search was limited to the first {max_pairwise_samples} samples."
        )
    return BatchGradientReport(
        samples=sample_results,
        dominant_samples=dominant,
        conflicting_samples=conflicting,
        duplicate_pairs=duplicate_pairs,
        cancellation=cancellation,
        selected_parameters=tuple(named_parameters),
        parameter_elements=parameter_elements,
        gradient_elements=gradient_elements,
        microbatch_size=microbatch_size,
        warnings=tuple(dict.fromkeys(warnings)),
        issues=tuple(issues),
        elapsed_seconds=time.perf_counter() - started,
    )


def _select_parameters(
    model: torch.nn.Module,
    parameters: Mapping[str, torch.Tensor] | Iterable[torch.Tensor] | None,
    layers: Iterable[str] | None,
) -> dict[str, torch.Tensor]:
    if isinstance(parameters, Mapping):
        selected = dict(parameters)
    elif parameters is not None:
        selected = {f"parameter[{index}]": value for index, value in enumerate(parameters)}
    else:
        patterns = tuple(layers) if layers is not None else None
        selected = {
            name: value
            for name, value in model.named_parameters()
            if value.requires_grad
            and (patterns is None or any(fnmatch(name, pattern) for pattern in patterns))
        }
    return {name: value for name, value in selected.items() if value.requires_grad}


def _split_batch(batch: Any) -> tuple[Any, Any | None]:
    if isinstance(batch, dict) and "inputs" in batch:
        return batch["inputs"], batch.get("targets")
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch[0], batch[1]
    return batch, None


def _batch_size(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            raise ValueError("batch input tensors must have a leading sample dimension")
        return int(value.shape[0])
    if isinstance(value, (tuple, list)):
        sizes = {_batch_size(item) for item in value}
    elif isinstance(value, dict):
        sizes = {_batch_size(item) for item in value.values()}
    else:
        raise TypeError("batch inputs must be tensors or nested tensor containers")
    if len(sizes) != 1:
        raise ValueError("all batch input tensors must share the leading dimension")
    return sizes.pop()


def _slice_tree(value: Any, indices: Sequence[int]) -> Any:
    if isinstance(value, torch.Tensor):
        index = torch.tensor(indices, device=value.device)
        return value.index_select(0, index)
    if isinstance(value, tuple):
        return tuple(_slice_tree(item, indices) for item in value)
    if isinstance(value, list):
        return [_slice_tree(item, indices) for item in value]
    if isinstance(value, dict):
        return {key: _slice_tree(item, indices) for key, item in value.items()}
    return value


def _invoke_model(model: torch.nn.Module, inputs: Any) -> Any:
    if isinstance(inputs, dict):
        return model(**inputs)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


def _invoke_loss(loss_fn: Callable[..., torch.Tensor], output: Any, targets: Any) -> torch.Tensor:
    if targets is None:
        return loss_fn(output)
    if isinstance(targets, dict):
        return loss_fn(output, **targets)
    if isinstance(targets, (tuple, list)):
        return loss_fn(output, *targets)
    return loss_fn(output, targets)


def _per_sample_losses(losses: torch.Tensor, count: int) -> tuple[torch.Tensor, ...]:
    if losses.numel() == 1 and count == 1:
        return (losses.reshape(()),)
    if losses.ndim == 0 or losses.shape[0] != count:
        raise ValueError("loss_fn must return a scalar or a tensor with leading batch dimension")
    reduced = losses.reshape(count, -1).mean(dim=1)
    return tuple(reduced.unbind())


def _single_sample_vector(
    model: torch.nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    inputs: Any,
    targets: Any,
    index: int,
    parameters: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    selected_inputs = _slice_tree(inputs, (index,))
    selected_targets = _slice_tree(targets, (index,)) if targets is not None else None
    output = _invoke_model(model, selected_inputs)
    loss = _invoke_loss(loss_fn, output, selected_targets)
    if loss.numel() != 1:
        loss = loss.mean()
    return _gradient_vector(loss, parameters, retain_graph=False)


def _gradient_vector(
    loss: torch.Tensor, parameters: tuple[torch.Tensor, ...], *, retain_graph: bool
) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat(
        tuple(
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
            for parameter, gradient in zip(parameters, gradients, strict=True)
        )
    )


def _cosines_to_batch(
    matrix: torch.Tensor,
    norms: torch.Tensor,
    combined: torch.Tensor,
    combined_norm: torch.Tensor,
) -> torch.Tensor:
    denominator = norms * combined_norm
    values = (matrix.conj() @ combined).real
    nan = torch.full_like(values, float("nan"))
    return torch.where(denominator > 0, values / denominator, nan)


def _duplicate_pairs(
    matrix: torch.Tensor,
    norms: torch.Tensor,
    indices: tuple[int, ...],
    *,
    threshold: float,
    max_samples: int,
) -> tuple[SamplePair, ...]:
    count = min(len(indices), max_samples)
    if count < 2:
        return ()
    selected = matrix[:count]
    selected_norms = norms[:count]
    denominator = selected_norms[:, None] * selected_norms[None, :]
    cosines = (selected.conj() @ selected.mT).real
    cosines = torch.where(denominator > 0, cosines / denominator, torch.nan)
    pairs: list[SamplePair] = []
    for left in range(count):
        for right in range(left + 1, count):
            cosine = float(cosines[left, right].item())
            if not math.isnan(cosine) and cosine >= threshold:
                pairs.append(SamplePair(indices[left], indices[right], cosine))
    return tuple(sorted(pairs, key=lambda value: value.cosine, reverse=True))
