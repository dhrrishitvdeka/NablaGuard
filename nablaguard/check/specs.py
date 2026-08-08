"""Input specifications for reproducible differentiable-operator checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Distribution = Literal["normal", "uniform", "zeros", "ones"]


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Recipe for one generated operator input."""

    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float64
    device: str | torch.device = "cpu"
    distribution: Distribution = "normal"
    low: float = -1.0
    high: float = 1.0
    requires_grad: bool = True

    def generate(self, generator: torch.Generator) -> torch.Tensor:
        """Generate a tensor from this recipe without changing global RNG state."""

        kwargs = {"dtype": self.dtype, "device": self.device}
        if self.distribution == "normal":
            value = torch.randn(self.shape, generator=generator, **kwargs)
        elif self.distribution == "uniform":
            value = torch.empty(self.shape, **kwargs)
            value.uniform_(self.low, self.high, generator=generator)
        elif self.distribution == "zeros":
            value = torch.zeros(self.shape, **kwargs)
        else:
            value = torch.ones(self.shape, **kwargs)
        if self.requires_grad and (value.is_floating_point() or value.is_complex()):
            value.requires_grad_(True)
        return value


def tensor(
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
    distribution: Distribution = "normal",
    low: float = -1.0,
    high: float = 1.0,
    requires_grad: bool = True,
) -> TensorSpec:
    """Describe an input for :func:`nablaguard.check.operator`."""

    return TensorSpec(shape, dtype, device, distribution, low, high, requires_grad)
