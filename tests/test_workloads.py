from __future__ import annotations

import json

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
