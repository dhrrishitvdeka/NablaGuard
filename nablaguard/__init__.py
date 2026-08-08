"""NablaGuard public API: verify and explain differentiable computations."""

from . import check, precision, report, trace
from .check.properties import Property, equivalent, property
from .check.specs import ShapeStrategy, TensorSpec, TensorStrategy, shapes, tensor
from .core import NablaConfig, NablaIssue, Session, Severity, TensorEvent
from .sanitize import Guard, guard, sanitize, shadow_rule

__version__ = "0.3.0"

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
    "precision",
    "sanitize",
    "shadow_rule",
    "shapes",
    "tensor",
    "trace",
]
