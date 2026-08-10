"""Unit tests for shared JSON redaction helpers."""

from __future__ import annotations

import pytest

from nablaguard.core.redaction import normalize_json, redact_value


def test_normalize_json_special_floats() -> None:
    assert normalize_json(float("nan")) == "NaN"
    assert normalize_json(float("inf")) == "Infinity"
    assert normalize_json(float("-inf")) == "-Infinity"
    assert normalize_json(1.5) == 1.5
    assert normalize_json({"x": [True, None]}) == {"x": [True, None]}


def test_normalize_json_rejects_objects() -> None:
    with pytest.raises(TypeError, match="unsupported value type"):
        normalize_json(object())


def test_redact_value_secret_keys_and_nested() -> None:
    value = redact_value(
        {
            "api_key": "secret",
            "nested": {"password": "x", "ok": "y"},
            "list": [{"authorization": "z"}],
        }
    )
    assert value["api_key"] == "<REDACTED>"
    assert value["nested"]["password"] == "<REDACTED>"
    assert value["nested"]["ok"] == "y"
    assert value["list"][0]["authorization"] == "<REDACTED>"
