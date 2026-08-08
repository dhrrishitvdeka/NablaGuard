"""Differentiable operator verification."""

from .fuzz import fuzz
from .fuzz_results import FuzzFailure, FuzzResult
from .minimizer import MinimizationResult, minimize
from .properties import Property, equivalent, property
from .results import Comparison, OperatorCheckResult
from .runner import operator
from .specs import ShapeStrategy, TensorSpec, TensorStrategy, shapes

__all__ = [
    "Comparison",
    "FuzzFailure",
    "FuzzResult",
    "MinimizationResult",
    "OperatorCheckResult",
    "Property",
    "ShapeStrategy",
    "TensorSpec",
    "TensorStrategy",
    "equivalent",
    "fuzz",
    "minimize",
    "operator",
    "property",
    "shapes",
]
