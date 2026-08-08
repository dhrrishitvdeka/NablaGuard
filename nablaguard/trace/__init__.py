"""Gradient provenance and multi-loss geometry."""

from .losses import LossTrace, gradient, losses
from .results import GradientComponent, GradientCosine, GradientReport
from .samples import BatchGradientReport, SampleGradient, SamplePair, samples

__all__ = [
    "BatchGradientReport",
    "GradientComponent",
    "GradientCosine",
    "GradientReport",
    "LossTrace",
    "SampleGradient",
    "SamplePair",
    "gradient",
    "losses",
    "samples",
]
