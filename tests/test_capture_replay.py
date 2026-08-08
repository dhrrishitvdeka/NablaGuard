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
    assert first.checksum != second.checksum
    assert first.shape == (100, 100)


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
