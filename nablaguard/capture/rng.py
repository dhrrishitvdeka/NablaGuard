"""Capture, restore, and fingerprint process RNG state."""

from __future__ import annotations

import hashlib
import importlib
import random
from typing import Any

import torch

np: Any = importlib.import_module("numpy")


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, PyTorch CPU, and available CUDA RNG state."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [value.cpu() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore all captured RNG sources available in the current environment."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda", [])
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def rng_digest(state: dict[str, Any] | None = None) -> str:
    """Hash RNG state for divergence checks without serializing it into JSON."""

    captured = capture_rng_state() if state is None else state
    digest = hashlib.sha256()
    digest.update(repr(captured["python"]).encode("utf-8"))
    numpy_state = captured["numpy"]
    digest.update(str(numpy_state[0]).encode("utf-8"))
    digest.update(numpy_state[1].tobytes())
    digest.update(repr(tuple(numpy_state[2:])).encode("utf-8"))
    digest.update(captured["torch_cpu"].contiguous().numpy().tobytes())
    for cuda_state in captured.get("torch_cuda", []):
        digest.update(cuda_state.contiguous().numpy().tobytes())
    return digest.hexdigest()
