from __future__ import annotations

from xml.etree import ElementTree

import torch

import nablaguard as ng
from nablaguard.check.fuzz_results import FuzzFailure, FuzzResult
from nablaguard.core import NablaIssue, Severity


def _session(*issues: NablaIssue) -> ng.Session:
    session = ng.Session()
    session.issues.extend(issues)
    return session


def test_session_json_html_and_junit_reports_escape_content() -> None:
    issue = NablaIssue(
        code="NG5001",
        category="CONTRACT_FAILED",
        severity=Severity.HIGH,
        message="x <script>alert(1)</script>",
        evidence={"value": torch.tensor(2.0)},
    )
    session = _session(issue)

    assert '"code": "NG5001"' in ng.report.dumps(session)
    html = ng.report.html(session)
    assert "&lt;script&gt;" in html
    assert "<script>alert" not in html
    xml = ng.report.junit(session)
    root = ElementTree.fromstring(xml)
    assert root.attrib["failures"] == "1"
    assert root.find("testcase/failure") is not None


def test_report_diff_detects_regression_and_resolution() -> None:
    old_issue = NablaIssue("NG1001", "NAN_DETECTED", Severity.CRITICAL, "nan")
    new_issue = NablaIssue("NG3002", "BACKWARD_MISMATCH", Severity.HIGH, "gradient")
    difference = ng.report.compare(_session(old_issue), _session(old_issue, new_issue))

    assert difference.regressed
    assert [issue["code"] for issue in difference.added] == ["NG3002"]
    assert difference.removed == ()


def test_empty_junit_report_is_a_passing_testcase() -> None:
    root = ElementTree.fromstring(ng.report.junit(_session()))
    assert root.attrib == {
        "name": "nablaguard",
        "tests": "1",
        "failures": "0",
        "errors": "0",
    }


def test_fuzz_failure_is_normalized_for_junit() -> None:
    spec = ng.TensorSpec((1,), torch.float64)
    result = FuzzResult(
        seed=1,
        requested_trials=1,
        cases_run=1,
        skipped_cases=0,
        failures=(FuzzFailure(0, 1, "mismatch", (spec,), (spec,)),),
        elapsed_seconds=0.1,
    )

    root = ElementTree.fromstring(ng.report.junit(result))
    assert root.attrib["failures"] == "1"
    assert root.find("testcase").attrib["name"] == "NG3000 FUZZ_FAILURE"  # type: ignore[union-attr]
