import torch

import nablaguard as ng


def test_nested_session_restores_outer_context() -> None:
    with ng.Session() as outer:
        ng.sanitize(torch.tensor([float("nan")]))
        with ng.Session() as inner:
            ng.check.operator(
                candidate=lambda x: x + 1,
                reference=lambda x: x,
                inputs=[ng.tensor(shape=(1,))],
                check_backward=False,
            )
        ng.check.operator(
            candidate=lambda x: x + 1,
            reference=lambda x: x,
            inputs=[ng.tensor(shape=(1,))],
            check_backward=False,
        )

    assert len(inner.issues) == 1
    assert len(outer.issues) == 1


def test_issue_is_machine_readable() -> None:
    issue = ng.NablaIssue(
        code="NG3001",
        category="FORWARD_MISMATCH",
        severity=ng.Severity.HIGH,
        message="different",
    )

    assert issue.to_dict()["severity"] == "high"
