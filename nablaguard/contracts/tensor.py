"""Built-in tensor contracts."""

from __future__ import annotations

from fnmatch import fnmatch

import torch

from .base import Contract, ContractContext


def finite(*, module: str = "*", raise_on_failure: bool = False) -> Contract:
    """Require selected observed tensors to contain only finite values."""

    def predicate(context: ContractContext) -> bool:
        if context.module_path is not None and not fnmatch(context.module_path, module):
            return True
        return context.tensor is not None and bool(torch.isfinite(context.tensor).all().item())

    return Contract(
        "tensor.finite",
        predicate,
        code="NG1001",
        category="NAN_DETECTED",
        message="Selected tensor contains non-finite values.",
        raise_on_failure=raise_on_failure,
        evidence_factory=lambda context: {
            "shape": list(context.tensor.shape) if context.tensor is not None else None,
            "dtype": str(context.tensor.dtype) if context.tensor is not None else None,
            "module_pattern": module,
        },
    )
