"""Layered training-state capture with bounded per-step metadata."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

from nablaguard.contracts import Contract, ContractContext
from nablaguard.core import NablaIssue
from nablaguard.core.serialization import atomic_torch_save, atomic_write_json

from .checkpoint import save_checkpoint
from .environment import determinism_limitations, environment_metadata
from .fingerprints import fingerprint_mapping
from .rng import capture_rng_state, rng_digest


@dataclass(slots=True)
class Recorder:
    """Capture full checkpoints plus replay-oriented lightweight step evidence."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer | None = None
    scheduler: Any = None
    scaler: Any = None
    root: Path = Path(".nabla/runs")
    run_id: str | None = None
    checkpoint_every: int = 1000
    metadata_every: int = 1
    fingerprint_samples: int = 4096
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    extra_state: dict[str, Any] = field(default_factory=dict)
    contracts: tuple[Contract, ...] = ()
    run_path: Path = field(init=False)
    current_step: int = field(default=0, init=False)
    contract_issues: list[NablaIssue] = field(default_factory=list, init=False)
    _entered: bool = field(default=False, init=False, repr=False)
    _previous_parameters: dict[str, torch.Tensor] = field(
        default_factory=dict, init=False, repr=False
    )
    _loss_history: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.checkpoint_every <= 0 or self.metadata_every <= 0:
            raise ValueError("checkpoint_every and metadata_every must be positive")
        if self.fingerprint_samples <= 0:
            raise ValueError("fingerprint_samples must be positive")
        identifier = self.run_id or _run_id()
        self.run_id = identifier
        self.run_path = Path(self.root) / identifier

    def __enter__(self) -> Recorder:
        if self._entered:
            raise RuntimeError("capture recorder is already active")
        self._entered = True
        self.run_path.mkdir(parents=True, exist_ok=True)
        environment = environment_metadata()
        manifest = {
            "format_version": 1,
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_every": self.checkpoint_every,
            "metadata_every": self.metadata_every,
            "fingerprint_samples": self.fingerprint_samples,
            "hyperparameters": self.hyperparameters,
            "environment": environment,
            "determinism_guaranteed": False,
            "determinism_limitations": determinism_limitations(environment),
        }
        atomic_write_json(self.run_path / "manifest.json", manifest)
        self._previous_parameters = _parameter_snapshot(self.model)
        self._save_full_checkpoint(0)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._entered = False

    def record_step(
        self,
        *,
        step: int | None = None,
        loss: float | torch.Tensor | None = None,
        batch_indices: list[int] | tuple[int, ...] | None = None,
        tensors: dict[str, torch.Tensor] | None = None,
        gradients: dict[str, torch.Tensor] | None = None,
        data_state: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path | None:
        """Record the state boundary after a completed training step."""

        if not self._entered:
            raise RuntimeError("record_step requires an active capture context")
        selected_step = self.current_step + 1 if step is None else step
        if selected_step <= self.current_step:
            raise ValueError("steps must be recorded in strictly increasing order")
        self.current_step = selected_step
        scalar_loss = _scalar_loss(loss)
        parameters = dict(self.model.named_parameters())
        context = ContractContext(
            loss=scalar_loss,
            gradients=gradients,
            parameters=parameters,
            previous_parameters=self._previous_parameters,
            loss_history=tuple(self._loss_history),
            extras={"step": selected_step, **(extra or {})},
        )
        step_contract_issues = [
            issue
            for assertion in self.contracts
            if (issue := assertion.evaluate(context)) is not None
        ]
        self.contract_issues.extend(step_contract_issues)
        metadata_path: Path | None = None
        if selected_step % self.metadata_every == 0:
            rng = capture_rng_state()
            metadata = {
                "format_version": 1,
                "step": selected_step,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "loss": scalar_loss,
                "batch_indices": list(batch_indices) if batch_indices is not None else None,
                "fingerprints": fingerprint_mapping(
                    tensors or {}, max_samples=self.fingerprint_samples
                ),
                "rng_digest": rng_digest(rng),
                "data_state": data_state or {},
                "extra": extra or {},
                "contract_issues": [issue.to_dict() for issue in step_contract_issues],
            }
            metadata_path = self.run_path / "steps" / f"step-{selected_step:08d}.json"
            atomic_write_json(metadata_path, metadata)
            atomic_torch_save(self.run_path / "rng" / f"step-{selected_step:08d}.pt", rng)
        if selected_step % self.checkpoint_every == 0:
            self._save_full_checkpoint(selected_step)
        self._previous_parameters = _parameter_snapshot(self.model)
        if scalar_loss is not None:
            self._loss_history.append(scalar_loss)
        return metadata_path

    def _save_full_checkpoint(self, step: int) -> None:
        save_checkpoint(
            self.run_path / "checkpoints" / f"step-{step:08d}.pt",
            step=step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            extra_state=self.extra_state,
        )


def capture(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    scheduler: Any = None,
    scaler: Any = None,
    root: str | Path = ".nabla/runs",
    run_id: str | None = None,
    checkpoint_every: int = 1000,
    metadata_every: int = 1,
    fingerprint_samples: int = 4096,
    hyperparameters: dict[str, Any] | None = None,
    extra_state: dict[str, Any] | None = None,
    contracts: Iterable[Contract] = (),
) -> Recorder:
    """Create a layered training-state recorder."""

    return Recorder(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        root=Path(root),
        run_id=run_id,
        checkpoint_every=checkpoint_every,
        metadata_every=metadata_every,
        fingerprint_samples=fingerprint_samples,
        hyperparameters=hyperparameters or {},
        extra_state=extra_state or {},
        contracts=tuple(contracts),
    )


def _scalar_loss(loss: float | torch.Tensor | None) -> float | None:
    if loss is None:
        return None
    if isinstance(loss, torch.Tensor):
        if loss.numel() != 1:
            raise ValueError("captured loss must be scalar")
        return float(loss.detach().item())
    return float(loss)


def _run_id() -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"run-{stamp}-{uuid4().hex[:8]}"


def _parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
