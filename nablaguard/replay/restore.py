"""Restore captured training-state boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from nablaguard.capture.checkpoint import load_checkpoint
from nablaguard.capture.rng import restore_rng_state


def checkpoint_steps(run_path: Path) -> tuple[int, ...]:
    """List valid full-checkpoint boundary steps."""

    return tuple(
        sorted(
            int(path.stem.removeprefix("step-"))
            for path in (run_path / "checkpoints").glob("step-*.pt")
        )
    )


def nearest_checkpoint(run_path: Path, step: int) -> tuple[int, Path]:
    """Find the latest full checkpoint not after ``step``."""

    candidates = [value for value in checkpoint_steps(run_path) if value <= step]
    if not candidates:
        raise FileNotFoundError(f"no checkpoint at or before step {step} in {run_path}")
    selected = max(candidates)
    return selected, run_path / "checkpoints" / f"step-{selected:08d}.pt"


def restore_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
) -> dict[str, Any]:
    """Restore model, optimizer, optional helpers, and RNG from a trusted checkpoint."""

    checkpoint = load_checkpoint(path)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and checkpoint["optimizer"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint["scheduler"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint["scaler"] is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint["rng"])
    return checkpoint
