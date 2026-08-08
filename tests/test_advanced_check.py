import torch

import nablaguard as ng


class BadSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.square()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * x


def test_correct_operator_passes_jvp_double_backward_and_finite_difference() -> None:
    result = ng.check.operator(
        candidate=lambda x: x**3,
        reference=lambda x: x**3,
        inputs=[ng.tensor(shape=(3,), dtype=torch.float64)],
        check_jvp=True,
        check_double_backward=True,
        check_finite_difference=True,
        vjp_cotangent="random",
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
    )

    assert result.passed
    assert all(value.passed for value in result.jvp)
    assert all(value.passed for value in result.double_backward)
    assert all(value.passed for value in result.finite_difference)
    assert "JVP: PASS" in result.format()


def test_bad_custom_backward_fails_jvp_and_finite_difference() -> None:
    result = ng.check.operator(
        candidate=BadSquare.apply,
        reference=lambda x: x.square(),
        inputs=[ng.tensor(shape=(3,), dtype=torch.float64)],
        check_backward=False,
        check_jvp=True,
        check_finite_difference=True,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
    )

    assert not result.passed
    assert {issue.category for issue in result.issues} == {
        "JVP_MISMATCH",
        "FINITE_DIFFERENCE_MISMATCH",
    }


def test_double_backward_mismatch_has_explicit_issue() -> None:
    result = ng.check.operator(
        candidate=BadSquare.apply,
        reference=lambda x: x.square(),
        inputs=[ng.tensor(shape=(3,), dtype=torch.float64)],
        check_backward=False,
        check_double_backward=True,
    )

    assert not result.passed
    assert result.issues[0].category == "DOUBLE_BACKWARD_MISMATCH"


def test_stateful_nondeterminism_is_detected_under_identical_rng() -> None:
    calls = 0

    def candidate(x: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return x + calls

    result = ng.check.operator(
        candidate=candidate,
        reference=lambda x: x + 1,
        inputs=[ng.tensor(shape=(2,))],
        check_backward=False,
        check_determinism=True,
    )

    assert not result.passed
    assert [issue.code for issue in result.issues] == ["NG3004"]


def test_operator_analysis_restores_global_rng_state() -> None:
    def candidate(x: torch.Tensor) -> torch.Tensor:
        torch.rand(3)
        return x

    torch.manual_seed(81)
    state = torch.get_rng_state()
    ng.check.operator(
        candidate=candidate,
        reference=lambda x: x,
        inputs=[ng.tensor(shape=(2,))],
        check_determinism=True,
    )
    observed = torch.rand(3)
    torch.set_rng_state(state)
    expected = torch.rand(3)

    assert torch.equal(observed, expected)


def test_finite_difference_has_explicit_element_budget() -> None:
    try:
        ng.check.operator(
            candidate=lambda x: x.square(),
            reference=lambda x: x.square(),
            inputs=[ng.tensor(shape=(10,), dtype=torch.float64)],
            check_finite_difference=True,
            max_finite_difference_elements=4,
        )
    except ValueError as error:
        assert "max_finite_difference_elements" in str(error)
    else:
        raise AssertionError("finite difference budget was not enforced")
