"""Transparent environment metadata for reproducibility reports."""

from __future__ import annotations

import os
import platform
import sys
from typing import Any

import torch


def environment_metadata() -> dict[str, Any]:
    """Describe runtime factors known to affect replay behavior."""

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cudnn_version": torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "pid": os.getpid(),
    }


def determinism_limitations(environment: dict[str, Any]) -> list[str]:
    """Return explicit reasons bitwise replay is not guaranteed."""

    limitations = [
        "External data sources and user callbacks are not captured automatically.",
        "Custom operators may own randomness or mutable state outside state_dict().",
    ]
    if not environment["deterministic_algorithms"]:
        limitations.append("PyTorch deterministic algorithms were not enabled.")
    if environment["cuda_available"]:
        limitations.append(
            "CUDA kernels, library versions, and launch ordering may prevent bitwise replay."
        )
    return limitations
