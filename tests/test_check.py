from pathlib import Path

import torch

import nablaguard as ng


class BadSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x**2

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * x


def test_bad_backward_is_detected_while_forward_passes() -> None:
    result = ng.check.operator(
        candidate=BadSquare.apply,
        reference=lambda x: x**2,
        inputs=[ng.tensor(shape=(32,), dtype=torch.float64)],
        seed=7,
    )

    assert all(item.passed for item in result.forward)
    assert not all(item.passed for item in result.backward)
    assert [issue.code for issue in result.issues] == ["NG3002"]
    assert result.backward[0].max_relative_error == 0.5
    assert "BACKWARD_MISMATCH" in result.format()
    assert "NablaGuard seed: 7" in result.format()
    assert result.candidate_name.endswith("BadSquare.apply")


def test_correct_operator_passes_forward_and_backward() -> None:
    result = ng.check.operator(
        candidate=lambda x: torch.sin(x) * x,
        reference=lambda x: torch.sin(x) * x,
        inputs=[ng.tensor(shape=(3, 5), dtype=torch.float64)],
    )

    assert result.passed
    assert result.issues == ()


def test_forward_mismatch_reports_precise_worst_element() -> None:
    value = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    result = ng.check.operator(
        candidate=lambda x: x + torch.tensor([0.0, 0.0, 2.0]),
        reference=lambda x: x,
        inputs=[value],
        check_backward=False,
    )

    assert not result.passed
    assert result.forward[0].failing_index == (2,)
    assert result.forward[0].candidate_value == 5.0
    assert result.forward[0].reference_value == 3.0
    assert result.issues[0].code == "NG3001"


def test_generated_inputs_do_not_perturb_global_rng() -> None:
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    ng.check.operator(
        candidate=lambda x: x.square(),
        reference=lambda x: x.square(),
        inputs=[ng.tensor(shape=(8,))],
    )
    observed = torch.rand(3)

    assert torch.equal(observed, expected)


def test_failed_check_writes_reproducible_artifact(tmp_path: Path) -> None:
    result = ng.check.operator(
        candidate=BadSquare.apply,
        reference=lambda x: x**2,
        inputs=[ng.tensor(shape=(4,), dtype=torch.float64)],
        artifact_dir=tmp_path,
    )

    assert result.artifact_path is not None
    assert (result.artifact_path / "metadata.json").is_file()
    assert (result.artifact_path / "inputs.pt").is_file()
    assert (result.artifact_path / "environment.json").is_file()
    assert (result.artifact_path / "reproduction.py").is_file()
    saved = torch.load(result.artifact_path / "inputs.pt", weights_only=True)
    assert saved[0].shape == (4,)


def test_check_emits_into_shared_session() -> None:
    with ng.Session() as session:
        ng.check.operator(
            candidate=lambda x: x + 1,
            reference=lambda x: x,
            inputs=[ng.tensor(shape=(2,))],
            check_backward=False,
        )

    assert [issue.code for issue in session.issues] == ["NG3001"]


def test_detached_candidate_is_reported_as_missing_gradient() -> None:
    result = ng.check.operator(
        candidate=lambda value: value.detach().square(),
        reference=lambda value: value.square(),
        inputs=[ng.tensor(shape=(4,), dtype=torch.float64)],
    )

    assert not result.passed
    assert [issue.category for issue in result.issues] == ["MISSING_GRADIENT"]


def test_non_differentiable_float_inputs_preserve_requires_grad() -> None:
    value = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(2.0, dtype=torch.float64, requires_grad=False)

    result = ng.check.operator(
        candidate=lambda input_value, factor: input_value * factor,
        reference=lambda input_value, factor: input_value * factor,
        inputs=[value, scale],
    )

    assert result.passed
    assert result.metadata["input_requires_grad"] == [True, False]
    assert len(result.backward) == 2


def test_complex_imaginary_mismatch_is_detected() -> None:
    value = torch.tensor([1 + 2j], dtype=torch.complex128)

    result = ng.check.operator(
        candidate=lambda input_value: input_value + 1j,
        reference=lambda input_value: input_value,
        inputs=[value],
        check_backward=False,
    )

    assert not result.passed
    assert result.forward[0].max_absolute_error == 1.0
    assert result.forward[0].candidate_value == "(1+3j)"


def test_candidate_and_reference_receive_identical_rng_state() -> None:
    def stochastic(value: torch.Tensor) -> torch.Tensor:
        return value + torch.rand_like(value)

    result = ng.check.operator(
        candidate=stochastic,
        reference=stochastic,
        inputs=[ng.tensor(shape=(8,), dtype=torch.float64)],
    )

    assert result.passed


def test_operator_check_restores_module_buffers() -> None:
    module = torch.nn.BatchNorm1d(3, dtype=torch.float64)
    original_batches = module.num_batches_tracked.detach().clone()

    result = ng.check.operator(
        candidate=module,
        reference=module,
        inputs=[ng.tensor(shape=(4, 3), dtype=torch.float64)],
    )

    assert result.passed
    assert torch.equal(module.num_batches_tracked, original_batches)
