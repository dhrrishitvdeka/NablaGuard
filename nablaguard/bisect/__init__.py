"""Training failure boundary search and diagnosis."""

from .diagnosis import BoundaryDiagnosis, ObservedChange, diagnose_boundary
from .predicates import metric_greater_than, metric_less_than, metric_nonfinite
from .runner import BisectProbe, BisectResult, BoundaryState, bisect
from .search import SearchProbe, SearchResult, first_bad

__all__ = [
    "BisectProbe",
    "BisectResult",
    "BoundaryDiagnosis",
    "BoundaryState",
    "ObservedChange",
    "SearchProbe",
    "SearchResult",
    "bisect",
    "diagnose_boundary",
    "first_bad",
    "metric_greater_than",
    "metric_less_than",
    "metric_nonfinite",
]
