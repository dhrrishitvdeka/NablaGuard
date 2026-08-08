import math

import pytest
import torch

import nablaguard as ng


def test_exact_opposition_has_complete_cancellation() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    left = parameter.sum()
    right = -parameter.sum()

    with ng.trace.losses(
        {"left": left, "right": right},
        parameters=[parameter],
        cancellation_threshold=0.5,
    ) as trace:
        (left + right).backward()

    report = trace.report(parameter, name="weight")
    assert report.cancellation == pytest.approx(1.0)
    assert report.cosine_similarities[0].cosine == pytest.approx(-1.0)
    assert {issue.code for issue in report.issues} == {"NG2003", "NG2004"}
    assert parameter.grad is not None
    assert torch.equal(parameter.grad, torch.zeros_like(parameter))


def test_aligned_gradients_have_no_cancellation() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    small = parameter.sum()
    large = 3.0 * parameter.sum()

    with ng.trace.losses({"small": small, "large": large}, parameters=[parameter]):
        (small + large).backward()

    report = ng.trace.gradient(parameter, name="linear.weight")
    assert report.cancellation == pytest.approx(0.0)
    assert report.cosine_similarities[0].cosine == pytest.approx(1.0)
    assert [component.magnitude_share for component in report.components] == pytest.approx(
        [0.25, 0.75]
    )


def test_parameter_name_mapping_is_used() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    loss = parameter.square().sum()

    with ng.trace.losses(
        {"loss": loss}, parameters=[parameter], parameter_names={parameter: "layer.weight"}
    ) as trace:
        loss.backward()

    assert trace.report(parameter).parameter_name == "layer.weight"


def test_zero_gradient_cosine_is_explicitly_undefined() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    zero = (parameter * 0).sum()
    active = parameter.sum()

    with ng.trace.losses({"zero": zero, "active": active}, parameters=[parameter]) as trace:
        (zero + active).backward()

    report = trace.report(parameter)
    assert math.isnan(report.cosine_similarities[0].cosine)
    assert report.cancellation == pytest.approx(0.0)


def test_trace_can_discover_reachable_leaf_parameter() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    first = parameter.square().sum()
    second = parameter.sum()

    with ng.trace.losses({"first": first, "second": second}) as trace:
        (first + second).backward()

    assert trace.report(parameter).components[0].norm == pytest.approx(4.0)


def test_non_scalar_loss_is_rejected() -> None:
    parameter = torch.nn.Parameter(torch.ones(2))
    with (
        pytest.raises(ValueError, match="must be scalar"),
        ng.trace.losses({"bad": parameter}, parameters=[parameter]),
    ):
        pass
