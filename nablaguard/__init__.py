"""NablaGuard public API: verify and explain differentiable computations."""

from . import check, precision, report, trace
from .bisect.runner import BisectResult, BoundaryState, bisect
from .capture.recorder import Recorder, capture
from .check.properties import Property, equivalent, property
from .check.specs import ShapeStrategy, TensorSpec, TensorStrategy, shapes, tensor
from .core import NablaConfig, NablaIssue, Session, Severity, TensorEvent
from .replay.runner import ReplayResult, replay
from .sanitize import Guard, guard, sanitize, shadow_rule

__version__ = "0.7.0"

__all__ = [
    "Guard",
    "BisectResult",
    "BoundaryState",
    "NablaConfig",
    "NablaIssue",
    "Property",
    "Recorder",
    "ReplayResult",
    "Session",
    "ShapeStrategy",
    "Severity",
    "TensorEvent",
    "TensorSpec",
    "TensorStrategy",
    "check",
    "capture",
    "bisect",
    "equivalent",
    "guard",
    "report",
    "replay",
    "property",
    "precision",
    "sanitize",
    "shadow_rule",
    "shapes",
    "tensor",
    "trace",
]
