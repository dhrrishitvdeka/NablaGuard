"""Self-contained HTML reports for local inspection and CI artifacts."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from typing import Any

from .json import SerializableReport
from .normalize import issues as report_issues


def render(report: SerializableReport, *, title: str = "NablaGuard report") -> str:
    """Render a structured report as dependency-free, self-contained HTML."""

    data = dict(report.to_dict())
    issues = report_issues(data)
    cards = "".join(_issue_card(issue) for issue in issues)
    if not cards:
        cards = '<section class="pass"><h2>PASS</h2><p>No issues were reported.</p></section>'
    raw = html.escape(json.dumps(data, indent=2, sort_keys=True, default=str))
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
:root {{ color-scheme: light dark; font-family: ui-monospace, Consolas, monospace; }}
body {{ max-width: 72rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.45; }}
section {{ border: 1px solid #8886; border-left: .4rem solid #c33; padding: 1rem; margin: 1rem 0; }}
.pass {{ border-left-color: #2a6; }} h1, h2 {{ font-family: system-ui, sans-serif; }}
dt {{ font-weight: 700; margin-top: .6rem; }} dd {{ margin-left: 0; }}
pre {{ overflow: auto; padding: 1rem; background: #8881; }}
</style>
</head>
<body><h1>{safe_title}</h1><p>{len(issues)} issue(s)</p>{cards}
<details><summary>Structured report</summary><pre>{raw}</pre></details></body></html>"""


def _issue_card(issue: Mapping[str, Any]) -> str:
    code = html.escape(str(issue.get("code", "UNKNOWN")))
    category = html.escape(str(issue.get("category", "ISSUE")))
    message = html.escape(str(issue.get("message", "")))
    severity = html.escape(str(issue.get("severity", "unknown")))
    evidence = html.escape(json.dumps(issue.get("evidence", {}), indent=2, default=str))
    suggestion = issue.get("suggestion")
    suggestion_html = (
        f"<dt>Suggestion</dt><dd>{html.escape(str(suggestion))}</dd>" if suggestion else ""
    )
    return (
        f"<section><h2>{code} {category}</h2><dl><dt>Severity</dt><dd>{severity}</dd>"
        f"<dt>Observation</dt><dd>{message}</dd><dt>Evidence</dt><dd><pre>{evidence}</pre></dd>"
        f"{suggestion_html}</dl></section>"
    )
