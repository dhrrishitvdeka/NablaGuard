import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

import nablaguard as ng
from nablaguard.capture import (
    capture_rng_state,
    fingerprint,
    restore_rng_state,
)


def test_rng_capture_restores_python_numpy_and_torch() -> None:
    random.seed(3)
    np.random.seed(4)
    torch.manual_seed(5)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), torch.rand(3))

    restore_rng_state(state)
    observed = (random.random(), float(np.random.rand()), torch.rand(3))

    assert observed[0] == expected[0]
    assert observed[1] == expected[1]
    assert torch.equal(observed[2], expected[2])


def test_fingerprint_is_bounded_and_detects_content_change() -> None:
    value = torch.arange(10_000, dtype=torch.float32).reshape(100, 100).t()
    first = fingerprint(value, max_samples=32)
    changed = value.clone()
    changed[0, 0] += 1
    second = fingerprint(changed, max_samples=32)

    assert first.sampled_elements == 32
    assert first.total_elements == 10_000
    assert first.checksum_scope == "sampled"
    assert first.statistics_scope == "full"
    assert first.checksum != second.checksum
    assert first.shape == (100, 100)
    full = fingerprint(torch.arange(4, dtype=torch.float32), max_samples=32)
    assert full.checksum_scope == "full"


def _capture_training_run(root: Path) -> Path:
    torch.manual_seed(123)
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    with ng.capture(
        model,
        optimizer,
        root=root,
        run_id="fixture",
        checkpoint_every=2,
        metadata_every=1,
        hyperparameters={"learning_rate": 0.05},
    ) as recorder:
        for step in range(1, 5):
            x = torch.randn(4, 1, dtype=torch.float64)
            loss = model(x).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            recorder.record_step(
                step=step,
                loss=loss,
                batch_indices=[step],
                tensors={"weight": model.weight},
            )
    return recorder.run_path


def _replay_objects():
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    return model, optimizer


def test_recorder_writes_layered_state_and_limitations(tmp_path: Path) -> None:
    run_path = _capture_training_run(tmp_path)

    assert sorted(path.name for path in (run_path / "checkpoints").glob("*.pt")) == [
        "step-00000000.pt",
        "step-00000002.pt",
        "step-00000004.pt",
    ]
    assert len(list((run_path / "steps").glob("*.json"))) == 4
    assert len(list((run_path / "rng").glob("*.pt"))) == 4
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["determinism_guaranteed"] is False
    assert manifest["determinism_limitations"]
    assert manifest["hyperparameters"]["learning_rate"] == 0.05


def test_replay_matches_exact_fingerprints_and_rng(tmp_path: Path) -> None:
    run_path = _capture_training_run(tmp_path)
    model, optimizer = _replay_objects()

    def step_fn(step: int, metadata: dict):
        assert metadata["batch_indices"] == [step]
        x = torch.randn(4, 1, dtype=torch.float64)
        loss = model(x).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"weight": model.weight}

    result = ng.replay(
        run_path,
        model=model,
        optimizer=optimizer,
        step_fn=step_fn,
        from_step=0,
        to_step=4,
    )

    assert result.passed
    assert [step.status for step in result.steps] == ["MATCH"] * 4
    assert all(step.rng_matches for step in result.steps)
    assert "Step 4: MATCH" in result.format()


def test_replay_stops_at_first_tensor_divergence(tmp_path: Path) -> None:
    run_path = _capture_training_run(tmp_path)
    model, optimizer = _replay_objects()

    def step_fn(step: int, metadata: dict):
        del metadata
        x = torch.randn(4, 1, dtype=torch.float64)
        loss = model(x).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 2:
            with torch.no_grad():
                model.weight.add_(1)
        return {"weight": model.weight}

    result = ng.replay(
        run_path,
        model=model,
        optimizer=optimizer,
        step_fn=step_fn,
        to_step=4,
    )

    assert not result.passed
    assert result.first_divergence is not None
    assert result.first_divergence.step == 2
    assert result.first_divergence.mismatches[0].name == "weight"
    assert result.issues[0].code == "NG4002"


def test_replay_uses_nearest_checkpoint_for_requested_boundary(tmp_path: Path) -> None:
    run_path = _capture_training_run(tmp_path)
    model, optimizer = _replay_objects()

    def step_fn(step: int, metadata: dict):
        del step, metadata
        x = torch.randn(4, 1, dtype=torch.float64)
        loss = model(x).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"weight": model.weight}

    result = ng.replay(
        run_path,
        model=model,
        optimizer=optimizer,
        step_fn=step_fn,
        from_step=2,
        to_step=4,
    )

    assert result.checkpoint_step == 2
    assert [value.step for value in result.steps] == [3, 4]
    assert result.passed


def test_recorder_rejects_non_monotonic_steps(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    with ng.capture(model, root=tmp_path, run_id="order") as recorder:
        recorder.record_step(step=2)
        with pytest.raises(ValueError, match="strictly increasing"):
            recorder.record_step(step=1)


def test_capture_rejects_run_id_path_escape(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match="relative identifier"):
        ng.capture(model, root=tmp_path, run_id="../escape")
    with pytest.raises(ValueError, match="relative identifier"):
        ng.capture(model, root=tmp_path, run_id=str(tmp_path / "abs"))
    with pytest.raises(ValueError, match="relative identifier"):
        ng.capture(model, root=tmp_path, run_id="bad/slash")


def test_capture_redacts_secret_metadata_keys(tmp_path: Path) -> None:
    model = torch.nn.Identity()
    with ng.capture(
        model,
        root=tmp_path,
        run_id="redact-meta",
        hyperparameters={"learning_rate": 0.1, "api_key": "super-secret"},
    ) as recorder:
        recorder.record_step(
            step=1,
            extra={"password": "also-secret", "note": "ok"},
            data_state={"token": "tok", "row": 3},
        )
    manifest = json.loads((recorder.run_path / "manifest.json").read_text(encoding="utf-8"))
    step = json.loads(
        (recorder.run_path / "steps" / "step-00000001.json").read_text(encoding="utf-8")
    )
    assert manifest["hyperparameters"]["api_key"] == "<REDACTED>"
    assert manifest["hyperparameters"]["learning_rate"] == 0.1
    assert step["extra"]["password"] == "<REDACTED>"
    assert step["extra"]["note"] == "ok"
    assert step["data_state"]["token"] == "<REDACTED>"
    assert step["data_state"]["row"] == 3


def test_capture_does_not_clone_parameters_without_change_contract(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 4)

    with ng.capture(model, root=tmp_path, run_id="no-parameter-copies") as recorder:
        assert recorder._previous_parameters == {}
        recorder.record_step(step=1)
        assert recorder._previous_parameters == {}


def test_capture_bounds_loss_history_to_contract_window(tmp_path: Path) -> None:
    model = torch.nn.Identity()
    contract = ng.contracts.training.loss_not_exploding(window=3)

    with ng.capture(
        model,
        root=tmp_path,
        run_id="bounded-loss-history",
        contracts=[contract],
    ) as recorder:
        for step in range(1, 8):
            recorder.record_step(step=step, loss=float(step))

    assert recorder._loss_history == [5.0, 6.0, 7.0]


def test_replay_without_tensor_evidence_is_not_a_pass(tmp_path: Path) -> None:
    model = torch.nn.Identity()
    with ng.capture(model, root=tmp_path, run_id="unverified") as recorder:
        recorder.record_step(step=1)

    result = ng.replay(
        recorder.run_path,
        model=torch.nn.Identity(),
        step_fn=lambda step, metadata: None,
        to_step=1,
    )

    assert not result.passed
    assert result.steps[0].status == "UNVERIFIED"
    assert result.issues[0].category == "REPLAY_UNVERIFIED"


def test_replay_detects_data_state_and_batch_identity_mismatch(tmp_path: Path) -> None:
    model = torch.nn.Identity()
    with ng.capture(model, root=tmp_path, run_id="data-state") as recorder:
        recorder.record_step(
            step=1,
            data_state={"dataset_id": "train", "row": 17},
            batch_indices=[17],
        )

    result = ng.replay(
        recorder.run_path,
        model=torch.nn.Identity(),
        step_fn=lambda step, metadata: ng.ReplayObservation(
            tensors={},
            data_state={"dataset_id": "train", "row": 18},
            batch_indices=(18,),
        ),
        to_step=1,
    )

    assert not result.passed
    assert result.steps[0].data_state_matches is False
    assert result.steps[0].batch_identity_matches is False
    assert result.issues[0].category == "DATALOADER_STATE_MISMATCH"
