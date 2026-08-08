"""NablaGuard public API: verify and explain differentiable computations."""

from . import check, report, trace
from .check.properties import Property, equivalent, property
from .check.specs import ShapeStrategy, TensorSpec, TensorStrategy, shapes, tensor
from .core import NablaConfig, NablaIssue, Session, Severity, TensorEvent
from .sanitize import Guard, guard, sanitize

__version__ = "0.2.0"

__all__ = [
    "Guard",
    "NablaConfig",
    "NablaIssue",
    "Property",
    "Session",
    "ShapeStrategy",
    "Severity",
    "TensorEvent",
    "TensorSpec",
    "TensorStrategy",
    "check",
    "equivalent",
    "guard",
    "report",
    "property",
    "sanitize",
    "shapes",
    "tensor",
    "trace",
]
