"""Full training-state checkpoint serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from nablaguard.core.serialization import atomic_torch_save

from .rng import capture_rng_state


def save_checkpoint(
    path: Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any = None,
    scaler: Any = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """Save trusted local state at the boundary after ``step``."""

    payload = {
        "format_version": 1,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": capture_rng_state(),
        "extra_state": extra_state or {},
    }
    atomic_torch_save(path, payload)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a **trusted local** NablaGuard checkpoint.

    Checkpoints are pickle-based (optimizer/scheduler state cannot use
    ``weights_only=True``). Only load paths produced by this process or another
    trusted NablaGuard capture. Never load checkpoints from untrusted sources.
    Failure-artifact inspection uses the separate NGF JSON path and never loads
    tensor pickles.
    """

    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or value.get("format_version") != 1:
        raise ValueError(f"unsupported or corrupt NablaGuard checkpoint: {path}")
    return value
