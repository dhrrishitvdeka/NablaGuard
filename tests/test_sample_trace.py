import pytest
import torch

import nablaguard as ng


def test_per_sample_gradients_find_dominance_conflict_and_duplicates() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    inputs = torch.tensor([[1.0], [2.0], [-1.0]])

    report = ng.trace.samples(
        model,
        lambda output: output.squeeze(-1),
        inputs,
        microbatch_size=3,
        cancellation_threshold=0.4,
    )

    assert [sample.norm for sample in report.samples] == pytest.approx([1.0, 2.0, 1.0])
    assert [sample.magnitude_share for sample in report.samples] == pytest.approx([0.25, 0.5, 0.25])
    assert report.samples[2].cosine_to_batch == pytest.approx(-1.0)
    assert report.dominant_samples[0].index == 1
    assert [sample.index for sample in report.conflicting_samples] == [2]
    assert report.duplicate_pairs[0].left == 0
    assert report.duplicate_pairs[0].right == 1
    assert report.cancellation == pytest.approx(0.5)
    assert {issue.code for issue in report.issues} == {"NG2003", "NG2004"}


def test_layer_and_sample_subset_selection() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    inputs = torch.randn(5, 2)

    report = ng.trace.samples(
        model,
        lambda output: output.squeeze(-1),
        inputs,
        layers=["1.weight"],
        sample_indices=[1, 4],
    )

    assert report.selected_parameters == ("1.weight",)
    assert [sample.index for sample in report.samples] == [1, 4]
    assert report.gradient_elements == model[1].weight.numel() * 2


def test_reduced_loss_falls_back_to_individual_forwards() -> None:
    model = torch.nn.Linear(2, 1)
    inputs = torch.randn(4, 2)
    targets = torch.randn(4, 1)

    report = ng.trace.samples(
        model,
        torch.nn.MSELoss(),
        (inputs, targets),
        microbatch_size=4,
    )

    assert len(report.samples) == 4
    assert any("falling back" in warning for warning in report.warnings)


def test_memory_bound_rejects_large_request_before_gradient_allocation() -> None:
    model = torch.nn.Linear(100, 100)

    with pytest.raises(MemoryError, match="max_gradient_elements"):
        ng.trace.samples(
            model,
            lambda output: output.sum(dim=1),
            torch.randn(10, 100),
            max_gradient_elements=100,
        )


def test_analysis_restores_rng_buffers_and_existing_gradients() -> None:
    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(2),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(2, 1),
    )
    selected = model[2].weight
    selected.grad = torch.ones_like(selected)
    original_mean = model[0].running_mean.clone()
    inputs = torch.randn(4, 2)
    torch.manual_seed(19)
    original_rng = torch.get_rng_state()

    ng.trace.samples(
        model,
        lambda output: output.squeeze(-1),
        inputs,
        parameters={"linear.weight": selected},
        microbatch_size=2,
    )
    observed_random = torch.rand(3)
    torch.set_rng_state(original_rng)
    expected_random = torch.rand(3)

    assert torch.equal(model[0].running_mean, original_mean)
    assert torch.equal(selected.grad, torch.ones_like(selected))
    assert torch.equal(observed_random, expected_random)


def test_per_sample_issues_emit_into_shared_session() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with ng.Session() as session:
        ng.trace.samples(
            model,
            lambda output: output.squeeze(-1),
            torch.tensor([[1.0], [-1.0]]),
            cancellation_threshold=0.9,
        )

    assert {issue.code for issue in session.issues} == {"NG2004"}
