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


def test_light_statistics_bound_descriptive_sample_but_scan_nonfinite_values() -> None:
    tensor = torch.arange(10_000, dtype=torch.float32).reshape(100, 100).t()
    tensor[-1, -1] = float("nan")

    statistics = compute_statistics(tensor, max_samples=32)

    assert statistics.sampled_elements <= 32
    assert statistics.total_elements == 10_000
    assert statistics.nan_count == 1


def test_light_guard_records_sampling_evidence() -> None:
    model = torch.nn.Linear(128, 128)

    with ng.guard(model, mode="light", light_sample_elements=16) as monitor:
        model(torch.ones(8, 128))

    assert monitor.events
    assert all(event.tags["statistics_sampled_elements"] <= 16 for event in monitor.events)
    assert all(event.tags["statistics_total_elements"] == 1024 for event in monitor.events)
    assert all(event.tags["statistics_sampled"] for event in monitor.events)


def test_light_guard_defaults_to_root_boundary_but_honors_selection() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())

    with ng.guard(model, mode="light") as root_monitor:
        model(torch.ones(2, 4))
    with ng.guard(model, mode="light", modules=["0"]) as selected_monitor:
        model(torch.ones(2, 4))

    assert [event.module_path for event in root_monitor.events] == ["<root>"]
    assert [event.module_path for event in selected_monitor.events] == ["0"]
