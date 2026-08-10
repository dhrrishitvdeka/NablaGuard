import torch

import nablaguard as ng
from nablaguard.sanitize import max_ulp_difference
from nablaguard.sanitize.numerical import compare_shadow
from nablaguard.sanitize.shadow import REGISTRY


def test_dispatch_detects_exp_overflow_before_output_issue() -> None:
    with ng.guard(
        shadow=True,
        operations=["aten.exp*"],
        capture_source=False,
    ) as monitor:
        torch.exp(torch.tensor([12.0], dtype=torch.float16))

    assert monitor.issues[0].code == "NG1003"
    assert monitor.issues[0].category == "OVERFLOW_RISK"
    assert monitor.issues[0].evidence["input_max"] == 12.0
    assert any(issue.code == "NG1001" for issue in monitor.issues)
    assert any(issue.code == "NG1002" for issue in monitor.issues)
    assert monitor.events[0].operation.startswith("aten.exp")


def test_shadow_finds_unstable_float32_sum() -> None:
    value = torch.tensor([100_000_000.0, 1.0, -100_000_000.0], dtype=torch.float32)
    with ng.guard(
        shadow=True,
        operations=["aten.sum*"],
        cancellation_threshold=1.0,
        capture_source=False,
    ) as monitor:
        torch.sum(value)

    instability = [issue for issue in monitor.issues if issue.category == "NUMERICAL_INSTABILITY"]
    assert len(instability) == 1
    assert instability[0].evidence["max_absolute_error"] == 1.0
    assert "combined_budget" in instability[0].evidence
    assert "atol + rtol" in instability[0].evidence["criterion"]


def test_sum_cancellation_metric_is_exactly_documented() -> None:
    with ng.guard(
        operations=["aten.sum*"], cancellation_threshold=0.9, capture_source=False
    ) as monitor:
        torch.tensor([1.0, -1.0]).sum()

    cancellation = [issue for issue in monitor.issues if issue.category == "NUMERICAL_CANCELLATION"]
    assert cancellation[0].evidence["cancellation"] == 1.0
    assert cancellation[0].evidence["definition"] == "1 - abs(sum(x)) / sum(abs(x))"


def test_event_propagation_uses_metadata_ids_not_tensor_retention() -> None:
    with ng.guard(operations=["aten.exp*", "aten.log*"], capture_source=False) as monitor:
        first = torch.exp(torch.tensor([1.0]))
        torch.log(first)

    assert len(monitor.events) == 2
    assert monitor.events[1].tags["upstream_event_ids"] == [monitor.events[0].event_id]


def test_operation_filter_prevents_unselected_dispatch_events() -> None:
    with ng.guard(operations=["aten.exp*"], capture_source=False) as monitor:
        torch.log(torch.tensor([2.0]))

    assert monitor.events == []


def test_ulp_difference_for_adjacent_float32_values() -> None:
    value = torch.tensor([1.0], dtype=torch.float32)
    adjacent = torch.nextafter(value, torch.tensor([2.0], dtype=torch.float32))

    assert max_ulp_difference(value, value.to(torch.float64)) == 0
    assert max_ulp_difference(value, adjacent.to(torch.float64)) == 1


def test_shadow_failure_emits_unknown_issue_instead_of_silent_skip() -> None:
    def broken_rule(operation, args, kwargs, dtype):
        del operation, args, kwargs, dtype
        raise RuntimeError("shadow backend unavailable")

    REGISTRY.register("aten::exp", broken_rule)
    try:
        with ng.guard(
            shadow=True,
            operations=["aten.exp*"],
            capture_source=False,
        ) as monitor:
            torch.exp(torch.tensor([1.0]))
    finally:
        REGISTRY.rules.pop("aten::exp", None)

    unsupported = [issue for issue in monitor.issues if issue.category == "SHADOW_UNSUPPORTED"]
    assert len(unsupported) == 1
    assert unsupported[0].code == "NG1005"
    assert unsupported[0].evidence["status"] == "UNKNOWN"


def test_matching_nonfinite_shadow_values_are_not_a_precision_mismatch() -> None:
    real = torch.tensor([float("inf"), float("nan")], dtype=torch.float32)
    shadow = real.to(torch.float64)

    comparison = compare_shadow(real, shadow)

    assert comparison.finite_mismatch_count == 0
    assert comparison.max_absolute_error == 0.0


def test_deep_mode_enables_shadow_by_default() -> None:
    monitor = ng.guard(mode="deep")
    assert monitor.shadow is True


def test_light_mode_does_not_enter_dispatch() -> None:
    with ng.guard(mode="light") as monitor:
        torch.exp(torch.tensor([100.0]))

    assert monitor.events == []
