import torch

import nablaguard as ng
from nablaguard.core import NablaConfig, NablaIssue, Severity
from nablaguard.core.serialization import dumps_json


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


def test_session_bounds_issues_as_well_as_events() -> None:
    session = ng.Session(config=NablaConfig(max_events=1, max_issues=1))
    session.emit_issue(NablaIssue("NG0001", "TEST", Severity.LOW, "one"))
    session.emit_issue(NablaIssue("NG0002", "TEST", Severity.LOW, "two"))

    assert len(session.issues) == 1
    assert session.dropped_issues == 1
    assert session.to_dict()["summary"]["dropped_issues"] == 1


def test_dumps_json_is_strict_and_summarizes_tensors() -> None:
    payload = {
        "ok": 1,
        "nan": float("nan"),
        "tensor": torch.zeros(2, 3),
    }
    rendered = dumps_json(payload)
    assert '"nan": "NaN"' in rendered
    assert '"_type": "torch.Tensor"' in rendered
    assert "0.0" not in rendered
