"""Internal normalization shared by machine-readable report encoders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def issues(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Extract or synthesize issue records from any public result shape."""

    direct = data.get("issues", [])
    found = (
        [issue for issue in direct if isinstance(issue, Mapping)]
        if isinstance(direct, list)
        else []
    )
    failures = data.get("failures", [])
    if isinstance(failures, list):
        for failure in failures:
            if not isinstance(failure, Mapping):
                continue
            nested = failure.get("issues", [])
            nested_issues = (
                [issue for issue in nested if isinstance(issue, Mapping)]
                if isinstance(nested, list)
                else []
            )
            if nested_issues:
                found.extend(nested_issues)
            else:
                found.append(
                    {
                        "code": "NG3000",
                        "category": "FUZZ_FAILURE",
                        "severity": "high",
                        "message": str(failure.get("reason", "A fuzz trial failed.")),
                        "evidence": {
                            "trial": failure.get("trial"),
                            "seed": failure.get("seed"),
                        },
                    }
                )
    if not found and data.get("passed") is False:
        found.append(
            {
                "code": "NG0001",
                "category": "VERIFICATION_FAILED",
                "severity": "high",
                "message": "The verification result did not pass.",
                "evidence": {},
            }
        )
    return found
