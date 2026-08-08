"""Deterministic tensor recipes and strategies for operator checks."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypeVar, cast

import torch

Distribution = Literal[
    "normal",
    "uniform",
    "zeros",
    "ones",
    "tiny",
    "huge",
    "mixed_magnitude",
    "positive",
    "negative",
    "near_zero",
    "nearly_identical",
    "high_dynamic_range",
    "repeated",
]
Layout = Literal["contiguous", "transposed", "strided", "sliced", "broadcasted"]

EDGE_DIMENSIONS = (1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 127, 128, 129, 257)
DEFAULT_DTYPES = (torch.float64, torch.float32, torch.bfloat16, torch.float16)
DEFAULT_DISTRIBUTIONS: tuple[Distribution, ...] = (
    "normal",
    "zeros",
    "ones",
    "uniform",
    "tiny",
    "huge",
    "mixed_magnitude",
    "nearly_identical",
)
DEFAULT_LAYOUTS: tuple[Layout, ...] = (
    "contiguous",
    "transposed",
    "strided",
    "sliced",
    "broadcasted",
)


@dataclass(frozen=True, slots=True)
class ShapeStrategy:
    """Finite, bounded shape search space."""

    explicit: tuple[tuple[int, ...], ...] = ()
    ranks: tuple[int, ...] = (1, 2, 3)
    dimensions: tuple[int, ...] = EDGE_DIMENSIONS
    max_elements: int = 65_536
    allow_zero: bool = False

    def sample(self, chooser: random.Random) -> tuple[int, ...]:
        """Sample a valid shape without depending on global RNG state."""

        if self.explicit:
            return chooser.choice(self.explicit)
        dimensions = self.dimensions + ((0,) if self.allow_zero else ())
        for _ in range(100):
            rank = chooser.choice(self.ranks)
            shape = tuple(chooser.choice(dimensions) for _ in range(rank))
            if math.prod(shape) <= self.max_elements:
                return shape
        smallest = min((value for value in dimensions if value > 0), default=1)
        return (smallest,) * min(self.ranks)


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
    layout: Layout = "contiguous"

    def generate(self, generator: torch.Generator) -> torch.Tensor:
        """Generate a tensor without changing global RNG state."""

        value = _generate_values(self, generator)
        value = _apply_layout(value, self, generator)
        if self.requires_grad and (value.is_floating_point() or value.is_complex()):
            value.requires_grad_(True)
        return value

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe recipe description."""

        return {
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "device": str(self.device),
            "distribution": self.distribution,
            "low": self.low,
            "high": self.high,
            "requires_grad": self.requires_grad,
            "layout": self.layout,
        }


@dataclass(frozen=True, slots=True)
class TensorStrategy:
    """Search space that resolves to a concrete :class:`TensorSpec` per trial."""

    shape: ShapeStrategy
    dtypes: tuple[torch.dtype, ...] = DEFAULT_DTYPES
    device: str | torch.device = "cpu"
    distributions: tuple[Distribution, ...] = DEFAULT_DISTRIBUTIONS
    layouts: tuple[Layout, ...] = DEFAULT_LAYOUTS
    low: float = -1.0
    high: float = 1.0
    requires_grad: bool = True

    def sample(self, chooser: random.Random) -> TensorSpec:
        """Resolve one reproducible concrete tensor recipe."""

        shape = self.shape.sample(chooser)
        valid_layouts = tuple(
            layout for layout in self.layouts if layout != "transposed" or len(shape) >= 2
        )
        return TensorSpec(
            shape=shape,
            dtype=chooser.choice(self.dtypes),
            device=self.device,
            distribution=chooser.choice(self.distributions),
            low=self.low,
            high=self.high,
            requires_grad=self.requires_grad,
            layout=chooser.choice(valid_layouts or ("contiguous",)),
        )


InputRecipe: TypeAlias = TensorSpec | TensorStrategy | torch.Tensor
T = TypeVar("T")


def shapes(
    *explicit: tuple[int, ...],
    ranks: Sequence[int] = (1, 2, 3),
    dimensions: Sequence[int] = EDGE_DIMENSIONS,
    max_elements: int = 65_536,
    allow_zero: bool = False,
) -> ShapeStrategy:
    """Describe a bounded shape search space for :func:`check.fuzz`."""

    if not ranks or any(rank < 0 for rank in ranks):
        raise ValueError("ranks must contain non-negative integers")
    if not dimensions or any(dimension <= 0 for dimension in dimensions):
        raise ValueError("dimensions must contain positive integers")
    if max_elements < 0:
        raise ValueError("max_elements must be non-negative")
    normalized = tuple(tuple(int(value) for value in shape) for shape in explicit)
    if any(any(value < 0 for value in shape) for shape in normalized):
        raise ValueError("explicit shapes cannot contain negative dimensions")
    return ShapeStrategy(normalized, tuple(ranks), tuple(dimensions), max_elements, allow_zero)


def tensor(
    *,
    shape: tuple[int, ...] | ShapeStrategy,
    dtype: torch.dtype | Sequence[torch.dtype] = torch.float64,
    device: str | torch.device = "cpu",
    distribution: Distribution | Sequence[Distribution] = "normal",
    layout: Layout | Sequence[Layout] = "contiguous",
    low: float = -1.0,
    high: float = 1.0,
    requires_grad: bool = True,
) -> TensorSpec | TensorStrategy:
    """Describe a concrete input or a fuzzing search space."""

    if isinstance(shape, ShapeStrategy) or not isinstance(dtype, torch.dtype):
        shape_strategy = shape if isinstance(shape, ShapeStrategy) else shapes(shape)
        return TensorStrategy(
            shape=shape_strategy,
            dtypes=_tuple_option(dtype),
            device=device,
            distributions=_tuple_option(distribution),
            layouts=_tuple_option(layout),
            low=low,
            high=high,
            requires_grad=requires_grad,
        )
    if not isinstance(distribution, str) or not isinstance(layout, str):
        return TensorStrategy(
            shape=shapes(shape),
            dtypes=(dtype,),
            device=device,
            distributions=_tuple_option(distribution),
            layouts=_tuple_option(layout),
            low=low,
            high=high,
            requires_grad=requires_grad,
        )
    return TensorSpec(shape, dtype, device, distribution, low, high, requires_grad, layout)


def _tuple_option(value: T | Sequence[T]) -> tuple[T, ...]:
    if isinstance(value, (str, torch.dtype)):
        return (cast(T, value),)
    return tuple(cast(Sequence[T], value))


def _generate_values(spec: TensorSpec, generator: torch.Generator) -> torch.Tensor:
    kwargs = {"dtype": spec.dtype, "device": spec.device}
    shape = _base_shape(spec)
    if spec.distribution == "zeros":
        return torch.zeros(shape, **kwargs)
    if spec.distribution == "ones":
        return torch.ones(shape, **kwargs)
    if spec.distribution == "uniform":
        return torch.empty(shape, **kwargs).uniform_(spec.low, spec.high, generator=generator)
    if spec.distribution == "positive":
        return torch.rand(shape, generator=generator, **kwargs).abs()
    if spec.distribution == "negative":
        return -torch.rand(shape, generator=generator, **kwargs).abs()
    standard = torch.randn(shape, generator=generator, **kwargs)
    if spec.distribution in {"tiny", "near_zero"}:
        return standard * _tiny_scale(spec.dtype)
    if spec.distribution == "huge":
        return standard.sign() * _huge_scale(spec.dtype)
    if spec.distribution in {"mixed_magnitude", "high_dynamic_range"}:
        exponent_limit = max(1, int(math.log10(math.sqrt(torch.finfo(spec.dtype).max))) - 1)
        exponents = torch.randint(
            -exponent_limit,
            exponent_limit + 1,
            shape,
            generator=generator,
            device=spec.device,
        )
        return standard * torch.pow(torch.tensor(10.0, **kwargs), exponents)
    if spec.distribution == "nearly_identical":
        return torch.ones(shape, **kwargs) + standard * _epsilon_scale(spec.dtype)
    if spec.distribution == "repeated":
        return torch.round(standard * 2) / 2
    return standard


def _base_shape(spec: TensorSpec) -> tuple[int, ...]:
    if spec.layout == "transposed" and len(spec.shape) >= 2:
        return (*spec.shape[:-2], spec.shape[-1], spec.shape[-2])
    if spec.layout in {"strided", "sliced"} and spec.shape:
        return (*spec.shape[:-1], spec.shape[-1] * 2 + 1)
    if spec.layout == "broadcasted" and spec.shape:
        dimension = next((index for index, size in enumerate(spec.shape) if size > 1), 0)
        return tuple(1 if index == dimension else size for index, size in enumerate(spec.shape))
    return spec.shape


def _apply_layout(
    value: torch.Tensor, spec: TensorSpec, generator: torch.Generator
) -> torch.Tensor:
    del generator
    if spec.layout == "transposed" and value.ndim >= 2:
        return value.transpose(-1, -2)
    if spec.layout == "strided" and value.ndim:
        return value[..., ::2][..., : spec.shape[-1]]
    if spec.layout == "sliced" and value.ndim:
        return value[..., 1::2][..., : spec.shape[-1]]
    if spec.layout == "broadcasted" and value.ndim:
        return value.expand(spec.shape)
    return value.contiguous()


def _tiny_scale(dtype: torch.dtype) -> float:
    return float(torch.finfo(dtype).tiny * 8) if dtype.is_floating_point else 1e-12


def _huge_scale(dtype: torch.dtype) -> float:
    return float(math.sqrt(torch.finfo(dtype).max) / 4) if dtype.is_floating_point else 1e6


def _epsilon_scale(dtype: torch.dtype) -> float:
    return float(torch.finfo(dtype).eps * 4) if dtype.is_floating_point else 1e-6
