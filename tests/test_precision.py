import torch

import nablaguard as ng


class UnstableSum(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum()


def test_precision_audit_promotes_unstable_sum_to_float64() -> None:
    model = UnstableSum()
    value = torch.tensor([100_000_000.0, 1.0, -100_000_000.0], dtype=torch.float64)

    result = ng.precision.audit(
        model,
        value,
        candidate_dtypes=(torch.float32,),
        max_relative_error=1e-6,
        absolute_tolerance=1e-8,
    )

    root = next(entry for entry in result.entries if entry.module_path == "<root>")
    assert root.recommended_dtype == "torch.float64"
    assert root.measurements[0].max_absolute_error == 1.0
    assert result.issues[0].category == "PRECISION_BUDGET_EXCEEDED"


def test_precision_audit_selects_first_candidate_within_budget() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Tanh())
    value = torch.tensor([[0.25, -0.5, 1.0]], dtype=torch.float32)

    result = ng.precision.audit(
        model,
        value,
        candidate_dtypes=(torch.float32,),
        max_relative_error=1e-4,
        absolute_tolerance=1e-5,
    )

    assert result.entries
    assert {entry.recommended_dtype for entry in result.entries} == {"torch.float32"}
    assert "Recommended dtype" in result.format()


def test_precision_audit_does_not_mutate_original_model() -> None:
    model = torch.nn.Linear(2, 1, dtype=torch.float32)
    model.train()

    ng.precision.audit(
        model,
        torch.ones(1, 2),
        candidate_dtypes=(torch.float32,),
        max_relative_error=1e-3,
    )

    assert model.training
    assert model.weight.dtype == torch.float32


def test_precision_audit_has_explicit_capture_bound() -> None:
    result = ng.precision.audit(
        torch.nn.Identity(),
        torch.ones(10),
        candidate_dtypes=(torch.float32,),
        max_capture_elements=5,
    )

    assert result.entries == ()
    assert result.skipped_modules == ("<root>",)
    assert result.captured_elements == 0
