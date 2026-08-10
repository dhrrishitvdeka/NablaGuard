"""Operator-check integration with the versioned NGF artifact format."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from nablaguard.artifact import ArtifactPolicy, create_failure_artifact


def write_failure_artifact(
    root: Path,
    *,
    metadata: dict[str, Any],
    inputs: Sequence[torch.Tensor],
    minimized_inputs: Sequence[torch.Tensor] | None = None,
    policy: ArtifactPolicy | None = None,
) -> Path:
    """Persist a private-by-default, bounded NGF operator failure."""

    return create_failure_artifact(
        root,
        issue=metadata,
        inputs=inputs,
        minimized_inputs=minimized_inputs,
        policy=policy,
        provenance={"subsystem": "operator_check"},
    )
