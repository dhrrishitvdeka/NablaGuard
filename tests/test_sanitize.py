import torch

import nablaguard as ng
from nablaguard.sanitize import compute_statistics


def test_statistics_handle_nonfinite_values_without_retaining_tensor() -> None:
    tensor = torch.tensor([0.0, 2.0, float("nan"), float("inf")])
    statistics = compute_statistics(tensor)

    assert statistics.minimum == 0.0
    assert statistics.maximum == 2.0
    assert statistics.mean == 1.0
    assert statistics.nan_count == 1
    assert statistics.inf_count == 1
    assert statistics.zero_fraction == 0.25


def test_explicit_sanitize_emits_nonfinite_issue() -> None:
    result = ng.sanitize(torch.tensor([1.0, float("nan")]))

    assert len(result.events) == 1
    assert result.events[0].nan_count == 1
    assert result.issues[0].code == "NG1001"


def test_extreme_threshold_is_configuration_not_hidden_heuristic() -> None:
    result = ng.sanitize(torch.tensor([101.0]), extreme_value_threshold=100.0)

    assert [issue.code for issue in result.issues] == ["NG1003"]
    assert result.issues[0].evidence["threshold"] == 100.0


def test_guard_module_filter_and_event_bound() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1))
    with ng.guard(model, modules=["0", "2"], max_events=1) as monitor:
        model(torch.ones(1, 2))

    assert len(monitor.events) == 1
    assert monitor.dropped_events == 1
    assert monitor.events[0].module_path == "0"


def test_guard_observes_a_root_only_module() -> None:
    model = torch.nn.Linear(2, 1)
    with ng.guard(model) as monitor:
        model(torch.ones(1, 2))

    assert [event.module_path for event in monitor.events] == ["<root>"]
