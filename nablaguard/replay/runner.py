"""Checkpoint-aware deterministic replay and fingerprint validation."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch

from nablaguard.capture.environment import environment_metadata
from nablaguard.capture.rng import rng_digest
from nablaguard.core import NablaIssue, Severity
from nablaguard.core.session import emit_issue

from .restore import nearest_checkpoint, restore_checkpoint
from .validator import FingerprintMismatch, validate_fingerprints

ReplayStatus = Literal["MATCH", "DIVERGENCE", "UNVERIFIED", "ERROR"]


@dataclass(frozen=True, slots=True)
class ReplayObservation:
    """Replay outputs plus optional data and batch identity evidence."""

    tensors: Mapping[str, torch.Tensor] | None = None
    data_state: Mapping[str, Any] | None = None
    batch_indices: tuple[int, ...] | None = None


StepFunction = Callable[
    [int, dict[str, Any]], Mapping[str, torch.Tensor] | ReplayObservation | None
]


@dataclass(frozen=True, slots=True)
class ReplayStepResult:
    """Validation result for one re-executed captured step."""

    step: int
    status: ReplayStatus
    warmup: bool
    mismatches: tuple[FingerprintMismatch, ...] = ()
    rng_matches: bool | None = None
    data_state_matches: bool | None = None
    batch_identity_matches: bool | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Evidence-backed replay report."""

    run_path: Path
    checkpoint_step: int
    from_step: int
    to_step: int
    steps: tuple[ReplayStepResult, ...]
    issues: tuple[NablaIssue, ...]
    environment_mismatches: tuple[str, ...]
    elapsed_seconds: float

    @property
    def first_divergence(self) -> ReplayStepResult | None:
        """Return the first divergent or error step."""

        return next(
            (value for value in self.steps if value.status in {"DIVERGENCE", "ERROR"}), None
        )

    @property
    def passed(self) -> bool:
        """Whether every requested step produced verified matching evidence.

        An empty step list is a pass only when the restored checkpoint already
        is the requested boundary (nothing left to re-execute).
        """

        if not self.steps:
            return self.checkpoint_step == self.to_step and self.from_step <= self.to_step
        return all(value.status == "MATCH" for value in self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report."""

        return {
            "run_path": str(self.run_path),
            "checkpoint_step": self.checkpoint_step,
            "from_step": self.from_step,
            "to_step": self.to_step,
            "passed": self.passed,
            "first_divergence": self.first_divergence.step if self.first_divergence else None,
            "steps": [asdict(value) for value in self.steps],
            "issues": [issue.to_dict() for issue in self.issues],
            "environment_mismatches": list(self.environment_mismatches),
            "elapsed_seconds": self.elapsed_seconds,
        }

    def format(self) -> str:
        """Render a terminal replay report."""

        lines = [
            "Replay verification",
            "=" * 32,
            f"Run: {self.run_path}",
            f"Restored checkpoint: step {self.checkpoint_step}",
        ]
        for step in self.steps:
            suffix = " (warmup)" if step.warmup else ""
            lines.append(f"Step {step.step}: {step.status}{suffix}")
            for mismatch in step.mismatches:
                lines.append(
                    f"  {mismatch.name}: expected {mismatch.expected_checksum}, "
                    f"observed {mismatch.observed_checksum}"
                )
        if self.first_divergence is not None:
            lines.extend(["", f"First divergence: step {self.first_divergence.step}"])
        if self.environment_mismatches:
            lines.extend(["", "Environment differences:", *self.environment_mismatches])
        lines.extend(["", f"Result: {'PASS' if self.passed else 'FAIL'}"])
        return "\n".join(lines)

    def print(self) -> None:
        """Print the terminal report."""

        print(self.format())


def replay(
    run: str | Path,
    *,
    model: torch.nn.Module,
    step_fn: StepFunction,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    from_step: int = 0,
    to_step: int | None = None,
    stop_on_divergence: bool = True,
) -> ReplayResult:
    """Restore a captured boundary, re-execute steps, and compare fingerprints."""

    started = time.perf_counter()
    run_path = Path(run)
    manifest_path = run_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"not a NablaGuard run: {run_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("metadata_every") != 1:
        raise ValueError("replay requires metadata_every=1 so no training steps are omitted")
    available_steps = _metadata_steps(run_path)
    selected_to = max(available_steps) if to_step is None and available_steps else to_step
    if selected_to is None:
        raise ValueError("run has no captured step metadata")
    if from_step < 0 or selected_to < from_step:
        raise ValueError("replay requires 0 <= from_step <= to_step")
    checkpoint_step, checkpoint_path = nearest_checkpoint(run_path, from_step)
    restore_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    expected_sequence = list(range(checkpoint_step + 1, selected_to + 1))
    missing = [step for step in expected_sequence if step not in available_steps]
    if missing:
        raise ValueError(f"captured metadata has replay-breaking gaps: {missing[:5]}")

    step_results: list[ReplayStepResult] = []
    issues: list[NablaIssue] = []
    sample_count = int(manifest.get("fingerprint_samples", 4096))
    for step in expected_sequence:
        metadata = _load_step(run_path, step)
        try:
            raw_observed = step_fn(step, metadata)
            observed = _normalize_observation(raw_observed)
        except Exception as error:
            issue = NablaIssue(
                code="NG4002",
                category="REPLAY_EXECUTION_ERROR",
                severity=Severity.HIGH,
                message="Replay callback raised before the captured boundary was reproduced.",
                evidence={
                    "step": step,
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                },
            )
            issues.append(issue)
            step_results.append(
                ReplayStepResult(step, "ERROR", step <= from_step, error=str(error))
            )
            break
        current_rng_digest = rng_digest()
        rng_matches = current_rng_digest == metadata.get("rng_digest")
        data_state_matches = _optional_state_match(
            metadata.get("data_state", {}), observed.data_state
        )
        batch_identity_matches = _optional_batch_match(
            metadata.get("batch_indices"), observed.batch_indices
        )
        if observed.tensors is None:
            status: ReplayStatus = "UNVERIFIED"
            mismatches: tuple[FingerprintMismatch, ...] = ()
        else:
            mismatches = validate_fingerprints(
                metadata.get("fingerprints", {}),
                dict(observed.tensors),
                max_samples=sample_count,
            )
            status = "MATCH" if not mismatches and rng_matches else "DIVERGENCE"
        if not rng_matches and observed.tensors is None:
            status = "DIVERGENCE"
        if data_state_matches is False or batch_identity_matches is False:
            status = "DIVERGENCE"
        result = ReplayStepResult(
            step,
            status,
            step <= from_step,
            mismatches,
            rng_matches,
            data_state_matches,
            batch_identity_matches,
        )
        step_results.append(result)
        if status == "DIVERGENCE":
            first = mismatches[0] if mismatches else None
            data_mismatch = data_state_matches is False or batch_identity_matches is False
            captured_data_state = metadata.get("data_state", {})
            dataset_id = (
                captured_data_state.get("dataset_id")
                if isinstance(captured_data_state, dict)
                else None
            )
            issues.append(
                NablaIssue(
                    code="NG4002",
                    category=(
                        "DATALOADER_STATE_MISMATCH"
                        if data_mismatch
                        else "UNREPRODUCIBLE_STATE"
                    ),
                    severity=Severity.HIGH,
                    message=(
                        "Replay data or batch identity differs from captured state."
                        if data_mismatch
                        else "Replay diverged from captured state."
                    ),
                    module_path=str(dataset_id) if dataset_id is not None else None,
                    operation="replay",
                    evidence={
                        "step": step,
                        "first_mismatch": (
                            first.name
                            if first
                            else "data_state"
                            if data_state_matches is False
                            else "batch_identity"
                            if batch_identity_matches is False
                            else "rng"
                        ),
                        "rng_matches": rng_matches,
                        "data_state_matches": data_state_matches,
                        "batch_identity_matches": batch_identity_matches,
                    },
                )
            )
            if stop_on_divergence:
                break
        elif status == "UNVERIFIED":
            issues.append(
                NablaIssue(
                    code="NG4002",
                    category="REPLAY_UNVERIFIED",
                    severity=Severity.HIGH,
                    message="Replay step returned no tensor evidence to verify.",
                    evidence={
                        "step": step,
                        "rng_matches": rng_matches,
                        "data_state_matches": data_state_matches,
                        "batch_identity_matches": batch_identity_matches,
                    },
                )
            )
            if stop_on_divergence:
                break

    environment_mismatches = _environment_mismatches(manifest.get("environment", {}))
    for issue in issues:
        emit_issue(issue)
    return ReplayResult(
        run_path=run_path,
        checkpoint_step=checkpoint_step,
        from_step=from_step,
        to_step=selected_to,
        steps=tuple(step_results),
        issues=tuple(issues),
        environment_mismatches=environment_mismatches,
        elapsed_seconds=time.perf_counter() - started,
    )


def _normalize_observation(
    value: Mapping[str, torch.Tensor] | ReplayObservation | None,
) -> ReplayObservation:
    if value is None:
        return ReplayObservation()
    if isinstance(value, ReplayObservation):
        return value
    return ReplayObservation(tensors=value)


def _optional_state_match(
    expected: Any, observed: Mapping[str, Any] | None
) -> bool | None:
    if observed is None:
        return None
    try:
        expected_json = json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False)
        observed_json = json.dumps(
            dict(observed), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise TypeError("replay data_state must contain strict JSON values") from error
    return observed_json == expected_json


def _optional_batch_match(expected: Any, observed: tuple[int, ...] | None) -> bool | None:
    if observed is None:
        return None
    return bool(list(observed) == expected)


def _metadata_steps(run_path: Path) -> set[int]:
    return {
        int(path.stem.removeprefix("step-")) for path in (run_path / "steps").glob("step-*.json")
    }


def _load_step(run_path: Path, step: int) -> dict[str, Any]:
    path = run_path / "steps" / f"step-{step:08d}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid step metadata: {path}")
    return cast(dict[str, Any], value)


def _environment_mismatches(captured: dict[str, Any]) -> tuple[str, ...]:
    current = environment_metadata()
    keys = ("python", "torch", "cuda_version", "cuda_available", "platform")
    return tuple(
        f"{key}: captured={captured.get(key)!r}, current={current.get(key)!r}"
        for key in keys
        if captured.get(key) != current.get(key)
    )
