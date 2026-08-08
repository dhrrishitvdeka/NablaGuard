"""Human- and machine-readable reporting."""

from .console import format_gradient_report, format_operator_result, format_session
from .diff import ReportDiff, compare
from .html import render as html
from .json import dumps
from .junit import dumps as junit

__all__ = [
    "ReportDiff",
    "compare",
    "dumps",
    "format_gradient_report",
    "format_operator_result",
    "format_session",
    "html",
    "junit",
]
