"""NablaGuard public API: verify and explain differentiable computations."""

from . import check, report, trace
from .check.specs import TensorSpec, tensor
from .core import NablaConfig, NablaIssue, Session, Severity, TensorEvent
from .sanitize import Guard, guard, sanitize

__version__ = "0.1.0"

__all__ = [
    "Guard",
    "NablaConfig",
    "NablaIssue",
    "Session",
    "Severity",
    "TensorEvent",
    "TensorSpec",
    "check",
    "guard",
    "report",
    "sanitize",
    "tensor",
    "trace",
]
