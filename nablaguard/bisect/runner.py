"""Captured-run metadata and checkpoint-aware training bisection."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from nablaguard.core import NablaIssue, Severity
from nablaguard.core.session import emit_issue
from nablaguard.replay import ReplayResult, replay

from .diagnosis import BoundaryDiagnosis, diagnose_boundary
from .search import first_bad

BoundaryPredicate = Callable[["BoundaryState"], bool]
ModelFactory = Callable[[], torch.nn.Module]
OptimizerFactory = Callable[[torch.nn.Module], torch.optim.Optimizer | None]
ReplayStepFunction = Callable[
    [torch.nn.Module, torch.optim.Optimizer | None, int, dict[str, Any]],
    Mapping[str, torch.Tensor] | None,
]


@dataclass(frozen=True, slots=True)
class BoundaryState:
    """State visible to a training-failure predicate."""

    step: int
    metadata: dict[str, Any]
    tensors: Mapping[str, torch.Tensor] | None = None
    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    replay_result: ReplayResult | None = None

    @property
    def loss(self) -> float | None:
        """Convenient access to captured scalar loss."""

        value = self.metadata.get("loss")
        return float(value) if value is not None else None


@dataclass(frozen=True, slots=True)
class BisectProbe:
    """One binary-search decision with restoration cost evidence."""

    step: int
    outcome: str
    checkpoint_step: int | None = None
    replayed_steps: int = 0


@dataclass(frozen=True, slots=True)
class BisectResult:
    """First captured false-to-true predicate boundary."""

    run_path: Path
    known_good: int
    known_bad: int
    first_bad_step: int
    probes: tuple[BisectProbe, ...]
    diagnosis: BoundaryDiagnosis
    issues: tuple[NablaIssue, ...]
    checkpoint_aware: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report."""

        return {
            "run_path": str(self.run_path),
            "known_good": self.known_good,
            "known_bad": self.known_bad,
            "first_bad_step": self.first_bad_step,
            "probes": [asdict(value) for value in self.probes],
            "diagnosis": self.diagnosis.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
            "checkpoint_aware": self.checkpoint_aware,
            "elapsed_seconds": self.elapsed_seconds,
        }

    def format(self) -> str:
        """Render a git-bisect-like terminal report."""

        lines = [
            "Training Failure Bisect",
            "=" * 32,
            f"Known good: step {self.known_good}",
            f"Known bad: step {self.known_bad}",
            "",
        ]
        for probe in self.probes:
            cost = (
                f" (checkpoint {probe.checkpoint_step}, replayed {probe.replayed_steps})"
                if probe.checkpoint_step is not None
                else ""
            )
            lines.append(f"{probe.step:<12}{probe.outcome}{cost}")
        lines.extend(
            [
                "",
                f"FIRST BAD STEP: {self.first_bad_step}",
                f"Trigger batch: {self.diagnosis.trigger_batch}",
                "",
                "Boundary observations:",
            ]
        )
        for observation in self.diagnosis.observations[:10]:
            lines.append(
                f"{observation.label} {observation.field}: "
                f"{observation.before!r} -> {observation.after!r}"
            )
        lines.extend(["", "Causality: UNKNOWN", *self.diagnosis.unknowns])
        return "\n".join(lines)

    def print(self) -> None:
        """Print the terminal report."""

        print(self.format())


def bisect(
    run: str | Path,
    predicate: BoundaryPredicate,
    *,
    known_good: int = 0,
    known_bad: int | None = None,
    model_factory: ModelFactory | None = None,
    optimizer_factory: OptimizerFactory | None = None,
    step_fn: ReplayStepFunction | None = None,
) -> BisectResult:
    """Locate the first bad captured step under a monotonic predicate.

    With ``model_factory`` and ``step_fn``, each probe restores the nearest full
    checkpoint into fresh objects and replays to the midpoint. Without them,
    predicates operate on captured metadata only.
    """

    started = time.perf_counter()
    run_path = Path(run)
    available = _metadata_steps(run_path)
    if not available:
        raise ValueError("run has no captured step metadata")
    selected_bad = max(available) if known_bad is None else known_bad
    if selected_bad not in available:
        raise ValueError(f"known_bad step {selected_bad} is not captured")
    checkpoint_aware = model_factory is not None or step_fn is not None
    if checkpoint_aware and (model_factory is None or step_fn is None):
        raise ValueError("checkpoint-aware bisection requires model_factory and step_fn")

    states: dict[int, BoundaryState] = {}
    costs: dict[int, tuple[int, int]] = {}

    def state_at(step: int) -> BoundaryState:
        if step in states:
            return states[step]
        metadata = _load_metadata(run_path, step)
        if model_factory is None or step_fn is None:
            state = BoundaryState(step, metadata)
            states[step] = state
            return state
        model = model_factory()
        optimizer = optimizer_factory(model) if optimizer_factory is not None else None
        last_tensors: Mapping[str, torch.Tensor] | None = None

        def invoke(
            current_step: int, current_metadata: dict[str, Any]
        ) -> Mapping[str, torch.Tensor] | None:
            nonlocal last_tensors
            last_tensors = step_fn(model, optimizer, current_step, current_metadata)
            return last_tensors

        replay_result = replay(
            run_path,
            model=model,
            optimizer=optimizer,
            step_fn=invoke,
            from_step=step,
            to_step=step,
        )
        if not replay_result.passed:
            raise RuntimeError(f"cannot evaluate predicate at step {step}: replay diverged first")
        costs[step] = (replay_result.checkpoint_step, len(replay_result.steps))
        state = BoundaryState(step, metadata, last_tensors, model, optimizer, replay_result)
        states[step] = state
        return state

    search = first_bad(known_good, selected_bad, lambda step: predicate(state_at(step)))
    probes = tuple(
        BisectProbe(
            value.step,
            value.outcome,
            costs.get(value.step, (None, 0))[0],
            costs.get(value.step, (0, 0))[1],
        )
        for value in search.probes
    )
    before = _load_metadata(run_path, search.first_bad_step - 1)
    after = _load_metadata(run_path, search.first_bad_step)
    diagnosis = diagnose_boundary(before, after, step=search.first_bad_step)
    issue = NablaIssue(
        code="NG4003",
        category="FAILURE_BOUNDARY_FOUND",
        severity=Severity.HIGH,
        message="A monotonic predicate transitioned from good to bad between adjacent steps.",
        evidence={
            "known_good": known_good,
            "known_bad": selected_bad,
            "first_bad_step": search.first_bad_step,
            "checkpoint_aware": checkpoint_aware,
            "conclusion_label": "OBSERVED",
            "causality": "UNKNOWN",
        },
        reproduction={"run_path": str(run_path)},
    )
    emit_issue(issue)
    return BisectResult(
        run_path=run_path,
        known_good=known_good,
        known_bad=selected_bad,
        first_bad_step=search.first_bad_step,
        probes=probes,
        diagnosis=diagnosis,
        issues=(issue,),
        checkpoint_aware=checkpoint_aware,
        elapsed_seconds=time.perf_counter() - started,
    )


def _metadata_steps(run_path: Path) -> set[int]:
    return {
        int(path.stem.removeprefix("step-")) for path in (run_path / "steps").glob("step-*.json")
    }


def _load_metadata(run_path: Path, step: int) -> dict[str, Any]:
    if step == 0:
        return {"step": 0, "loss": None, "fingerprints": {}, "batch_indices": None}
    path = run_path / "steps" / f"step-{step:08d}.json"
    if not path.is_file():
        return {"step": step, "loss": None, "fingerprints": {}, "batch_indices": None}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid captured metadata: {path}")
    return value
