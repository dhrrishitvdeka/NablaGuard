"""Differentiable operator verification."""

from .results import Comparison, OperatorCheckResult
from .runner import operator
from .specs import TensorSpec

__all__ = ["Comparison", "OperatorCheckResult", "TensorSpec", "operator"]
