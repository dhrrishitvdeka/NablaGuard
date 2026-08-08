from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch

import nablaguard as ng


def test_transformer_training_backward_under_guard() -> None:
    torch.manual_seed(13)
    layer = torch.nn.TransformerEncoderLayer(
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        dropout=0.0,
        batch_first=True,
        dtype=torch.float64,
    )
    value = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)

    with ng.guard(layer, mode="light", capture_source=False) as monitor:
        loss = layer(value).square().mean()
        loss.backward()

    assert torch.isfinite(loss)
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert monitor.events
    assert monitor.issues == []


def test_cnn_optimizer_step_capture_with_contracts(tmp_path) -> None:
    torch.manual_seed(17)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 2),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(4, 3, 8, 8)
    targets = torch.tensor([0, 1, 0, 1])

    with ng.capture(
        model,
        optimizer,
        root=tmp_path,
        run_id="cnn-workload",
        checkpoint_every=10,
        contracts=[
            ng.contracts.loss.finite(),
            ng.contracts.gradient.norm(max=100),
            ng.contracts.parameter.change(min_relative=1e-12),
        ],
    ) as recorder:
        loss = torch.nn.functional.cross_entropy(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        metadata_path = recorder.record_step(loss=loss)

    assert metadata_path is not None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["contract_issues"] == []
    assert recorder.contract_issues == []


@pytest.mark.parametrize("mode", ["light", "standard", "deep"])
def test_observation_modes_do_not_change_training_semantics(mode: str) -> None:
    baseline = _deterministic_training(None)
    observed = _deterministic_training(mode)

    assert observed["losses"] == baseline["losses"]
    assert torch.equal(observed["rng"], baseline["rng"])
    for name, expected in baseline["parameters"].items():
        assert torch.equal(observed["parameters"][name], expected), name
    for name, expected in baseline["gradients"].items():
        assert torch.equal(observed["gradients"][name], expected), name
    _assert_nested_equal(observed["optimizer"], baseline["optimizer"])


def _deterministic_training(mode: str | None) -> dict[str, object]:
    torch.manual_seed(9127)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.Tanh(),
        torch.nn.Dropout(p=0.25),
        torch.nn.Linear(8, 2),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    losses: list[float] = []
    monitor = ng.guard(model, mode=mode, capture_source=False) if mode is not None else None
    if monitor is not None:
        monitor.__enter__()
    try:
        for _ in range(3):
            inputs = torch.randn(6, 4)
            targets = torch.randn(6, 2)
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
            loss.backward()
            losses.append(float(loss.detach()))
            optimizer.step()
    finally:
        if monitor is not None:
            monitor.__exit__(None, None, None)
    return {
        "losses": losses,
        "parameters": {
            name: value.detach().clone() for name, value in model.named_parameters()
        },
        "gradients": {
            name: value.grad.detach().clone()
            for name, value in model.named_parameters()
            if value.grad is not None
        },
        "optimizer": deepcopy(optimizer.state_dict()),
        "rng": torch.get_rng_state().clone(),
    }


def _assert_nested_equal(observed: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(observed, torch.Tensor)
        assert torch.equal(observed, expected)
        return
    if isinstance(expected, dict):
        assert isinstance(observed, dict)
        assert observed.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(observed[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(observed, type(expected))
        assert len(observed) == len(expected)
        for left, right in zip(observed, expected, strict=True):
            _assert_nested_equal(left, right)
        return
    assert observed == expected
