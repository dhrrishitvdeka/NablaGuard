"""Shared secret and path redaction for JSON metadata surfaces."""

from __future__ import annotations

import getpass
import math
import os
import re
import socket
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SECRET_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|authorization|credential|password|secret|token)([_-]|$)", re.I
)


def redact_value(value: Any, *, key: str | None = None) -> Any:
    """Recursively redact secret-like keys and sensitive path substrings."""

    if key is not None and _SECRET_KEY.search(key):
        return "<REDACTED>"
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_value(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_value(item) for item in value]
    if isinstance(value, str):
        return redact_string(value)
    return normalize_json(value)


def redact_string(value: str) -> str:
    """Redact home prefixes and whole-token user/host names in strings."""

    sanitized = value
    home = str(Path.home())
    if home:
        variants = {home, home.replace("\\", "/")}
        for variant in sorted(variants, key=len, reverse=True):
            if not variant:
                continue
            if (
                sanitized == variant
                or sanitized.startswith(variant + "/")
                or sanitized.startswith(variant + os.sep)
            ):
                sanitized = "<HOME>" + sanitized[len(variant) :]
            sanitized = sanitized.replace(variant + "/", "<HOME>/")
            sanitized = sanitized.replace(variant + os.sep, "<HOME>" + os.sep)
    for sensitive, replacement in (
        (getpass.getuser(), "<USER>"),
        (socket.gethostname(), "<HOST>"),
    ):
        if not sensitive or len(sensitive) < 2:
            continue
        sanitized = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(sensitive)}(?![A-Za-z0-9_])",
            replacement,
            sanitized,
        )
    return sanitized


def normalize_json(value: Any) -> Any:
    """Convert finite JSON-friendly scalars; reject unsupported types."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, Mapping):
        return {str(key): normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json(item) for item in value]
    raise TypeError(f"metadata contains unsupported value type: {type(value).__name__}")
