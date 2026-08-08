"""NablaGuard public API: verify and explain differentiable computations."""

from . import check, contracts, precision, report, trace
from .bisect.runner import BisectResult, BoundaryState, bisect
from .capture.recorder import Recorder, capture
from .check.properties import Property, equivalent, property
from .check.specs import ShapeStrategy, TensorSpec, TensorStrategy, shapes, tensor
from .contracts.base import Contract, ContractContext, ContractViolation, contract
from .core import NablaConfig, NablaIssue, Session, Severity, TensorEvent
from .replay.runner import ReplayObservation, ReplayResult, replay
from .sanitize import Guard, guard, sanitize, shadow_rule

__version__ = "1.0.0"

__all__ = [
    "Guard",
    "BisectResult",
    "BoundaryState",
    "Contract",
    "ContractContext",
    "ContractViolation",
    "NablaConfig",
    "NablaIssue",
    "Property",
    "Recorder",
    "ReplayObservation",
    "ReplayResult",
    "Session",
    "ShapeStrategy",
    "Severity",
    "TensorEvent",
    "TensorSpec",
    "TensorStrategy",
    "check",
    "capture",
    "contract",
    "contracts",
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
