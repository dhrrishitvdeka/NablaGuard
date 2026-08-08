"""Portable evidence artifacts for failed operator checks."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch


def write_failure_artifact(
    root: Path,
    *,
    metadata: dict[str, Any],
    inputs: Sequence[torch.Tensor],
) -> Path:
    """Persist a failed experiment and return its content-addressed directory."""

    encoded = json.dumps(metadata, sort_keys=True, default=str).encode("utf-8")
    failure_id = f"NGF-{hashlib.sha256(encoded).hexdigest()[:8]}"
    destination = root / failure_id
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    torch.save([value.detach().cpu() for value in inputs], destination / "inputs.pt")
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    (destination / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "reproduction.py").write_text(_REPRODUCTION, encoding="utf-8")
    return destination


_REPRODUCTION = '''"""Inspect the exact inputs saved by a NablaGuard failure artifact."""
from pathlib import Path
import json
import torch

HERE = Path(__file__).resolve().parent
metadata = json.loads((HERE / "metadata.json").read_text(encoding="utf-8"))
inputs = torch.load(HERE / "inputs.pt", weights_only=True)
print(json.dumps(metadata, indent=2))
print("Loaded input shapes:", [tuple(value.shape) for value in inputs])
print("Re-run the named candidate and reference from metadata with these inputs.")
'''
