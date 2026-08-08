from __future__ import annotations

import torch

import nablaguard as ng
from tests.fixtures.broken_models import WrongSquare, boundary_bug, unstable_softmax


def test_wrong_backward_fixture_is_detected() -> None:
    result = ng.check.operator(
        candidate=WrongSquare.apply,
        reference=lambda value: value.square(),
        inputs=[ng.tensor(shape=(8,), dtype=torch.float64)],
        seed=7,
    )

    assert not result.passed
    assert any(issue.code == "NG3002" for issue in result.issues)


def test_unstable_softmax_fixture_is_detected() -> None:
    with ng.guard(mode="standard", capture_source=False) as monitor:
        unstable_softmax(torch.tensor([[100.0, 99.0]], dtype=torch.float32))

    assert any(issue.category == "OVERFLOW_RISK" for issue in monitor.issues)


def test_boundary_fixture_is_discovered_and_shrunk() -> None:
    strategy = ng.tensor(
        shape=ng.shapes(ranks=(1,), dimensions=(16, 17)),
        dtype=[torch.float64],
        distribution=["normal"],
        layout=["contiguous"],
    )
    result = ng.check.fuzz(
        candidate=boundary_bug,
        reference=lambda value: value,
        inputs=[strategy],
        trials=20,
        seed=4,
    )

    assert result.failures
    assert result.failures[0].minimal_specs[0].shape == (17,)
