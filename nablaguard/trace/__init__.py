"""Gradient provenance and multi-loss geometry."""

from .losses import LossTrace, gradient, losses
from .results import GradientComponent, GradientCosine, GradientReport

__all__ = [
    "GradientComponent",
    "GradientCosine",
    "GradientReport",
    "LossTrace",
    "gradient",
    "losses",
]
