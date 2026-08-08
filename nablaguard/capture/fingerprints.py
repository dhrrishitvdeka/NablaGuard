"""Bounded tensor fingerprints for divergence localization."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class TensorFingerprint:
    """Scalar statistics plus a deterministic bounded-content hash."""

    shape: tuple[int, ...]
    dtype: str
    device: str
    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None
    norm: float | None
    finite: bool
    checksum: str
    sampled_elements: int
    total_elements: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation."""

        value = asdict(self)
        value["shape"] = list(self.shape)
        return value


def fingerprint(tensor: torch.Tensor, *, max_samples: int = 4096) -> TensorFingerprint:
    """Fingerprint a tensor with at most ``max_samples`` content elements."""

    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    detached = tensor.detach()
    total = detached.numel()
    if total > max_samples:
        linear = torch.linspace(0, total - 1, steps=max_samples, device=detached.device).long()
        coordinates: list[torch.Tensor] = []
        remaining = linear
        for size in reversed(detached.shape):
            coordinates.append(remaining.remainder(size))
            remaining = torch.div(remaining, size, rounding_mode="floor")
        sampled = detached[tuple(reversed(coordinates))]
    else:
        sampled = detached.reshape(-1)
    cpu_sample = sampled.contiguous().to(device="cpu")
    raw = cpu_sample.view(torch.uint8).numpy().tobytes()
    header = f"{tuple(detached.shape)}|{detached.dtype}|{total}|{len(raw)}".encode()
    checksum = hashlib.sha256(header + raw).hexdigest()

    values = detached.abs() if detached.is_complex() else detached
    if total == 0 or not (values.is_floating_point() or values.is_complex()):
        numeric = values.to(torch.float64) if total else None
        finite = True
    else:
        finite_mask = torch.isfinite(values)
        finite = bool(finite_mask.all().item())
        numeric = values[finite_mask].to(torch.float64)
    if numeric is None or numeric.numel() == 0:
        minimum = maximum = mean = std = norm = None
    else:
        minimum = float(numeric.min().item())
        maximum = float(numeric.max().item())
        mean = float(numeric.mean().item())
        std = float(numeric.std(unbiased=False).item())
        norm = float(torch.linalg.vector_norm(numeric).item())
        for value in (minimum, maximum, mean, std, norm):
            if not math.isfinite(value):
                finite = False
    return TensorFingerprint(
        shape=tuple(detached.shape),
        dtype=str(detached.dtype),
        device=str(detached.device),
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        std=std,
        norm=norm,
        finite=finite,
        checksum=checksum,
        sampled_elements=sampled.numel(),
        total_elements=total,
    )


def fingerprint_mapping(
    tensors: dict[str, torch.Tensor], *, max_samples: int = 4096
) -> dict[str, dict[str, Any]]:
    """Fingerprint named tensors for JSON step metadata."""

    return {
        name: fingerprint(value, max_samples=max_samples).to_dict()
        for name, value in tensors.items()
    }
