"""JUnit XML output for CI systems."""

from __future__ import annotations

from xml.etree import ElementTree

from nablaguard.core.serialization import dumps_json

from .json import SerializableReport
from .normalize import issues as report_issues


def dumps(report: SerializableReport, *, suite_name: str = "nablaguard") -> str:
    """Serialize report issues as JUnit test failures."""

    data = dict(report.to_dict())
    issues = report_issues(data)
    suite = ElementTree.Element(
        "testsuite",
        name=suite_name,
        tests=str(max(1, len(issues))),
        failures=str(len(issues)),
        errors="0",
    )
    if issues:
        for issue in issues:
            code = str(issue.get("code", "UNKNOWN"))
            category = str(issue.get("category", "ISSUE"))
            case = ElementTree.SubElement(suite, "testcase", name=f"{code} {category}")
            failure = ElementTree.SubElement(
                case,
                "failure",
                message=str(issue.get("message", category)),
                type=category,
            )
            failure.text = dumps_json(dict(issue))
    else:
        ElementTree.SubElement(suite, "testcase", name="verification")
    return ElementTree.tostring(suite, encoding="unicode", xml_declaration=True)
