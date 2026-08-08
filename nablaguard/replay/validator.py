"""Tensor-fingerprint replay validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from nablaguard.capture.fingerprints import fingerprint


@dataclass(frozen=True, slots=True)
class FingerprintMismatch:
    """Exact fingerprint discrepancy for one named tensor."""

    name: str
    reason: str
    expected_checksum: str | None
    observed_checksum: str | None
    expected: dict[str, Any] | None = None
    observed: dict[str, Any] | None = None


def validate_fingerprints(
    expected: dict[str, dict[str, Any]],
    observed: dict[str, torch.Tensor],
    *,
    max_samples: int,
) -> tuple[FingerprintMismatch, ...]:
    """Compare captured fingerprints with tensors returned by a replay callback."""

    mismatches: list[FingerprintMismatch] = []
    for name, expected_value in expected.items():
        tensor = observed.get(name)
        if tensor is None:
            mismatches.append(
                FingerprintMismatch(
                    name,
                    "observed tensor is missing",
                    expected_value.get("checksum"),
                    None,
                    expected_value,
                    None,
                )
            )
            continue
        observed_value = fingerprint(tensor, max_samples=max_samples).to_dict()
        if observed_value["checksum"] != expected_value.get("checksum"):
            mismatches.append(
                FingerprintMismatch(
                    name,
                    "checksum mismatch",
                    expected_value.get("checksum"),
                    str(observed_value["checksum"]),
                    expected_value,
                    observed_value,
                )
            )
    return tuple(mismatches)
