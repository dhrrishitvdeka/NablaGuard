"""Non-interfering callable execution helpers for operator verification."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Any, cast

import torch


def call_with_isolated_module_state(
    function: Callable[..., Any], inputs: Sequence[Any]
) -> Any:
    """Execute a callable on an independent module copy when one owns it."""

    return isolated_callable(function)(*inputs)


def isolated_callable(function: Callable[..., Any]) -> Callable[..., Any]:
    """Clone an ``nn.Module`` callable or bound module method.

    Free functions and callables that close over external mutable state are
    returned unchanged; callers must treat them as potentially interfering.
    Module parameters and buffers are deep-copied so candidate and reference
    module evaluations do not share trainable state.
    """

    if isinstance(function, torch.nn.Module):
        return cast(Callable[..., Any], _clone_module(function))
    owner = getattr(function, "__self__", None)
    if isinstance(owner, torch.nn.Module):
        cloned_owner = _clone_module(owner)
        method_name = getattr(function, "__name__", None)
        if isinstance(method_name, str):
            cloned_method = getattr(cloned_owner, method_name)
            if callable(cloned_method):
                return cast(Callable[..., Any], cloned_method)
    return function


def _clone_module(module: torch.nn.Module) -> torch.nn.Module:
    cloned = copy.deepcopy(module)
    cloned.train(module.training)
    return cloned


def leaf_copy(value: torch.Tensor) -> torch.Tensor:
    """Independent leaf that preserves strides, broadcasts, and requires_grad.

    Broadcasted (zero-stride) dimensions are re-expanded from a narrowed base
    so the copy does not materialize a dense tensor first. Custom strides are
    reconstructed with ``empty_strided`` so layout-sensitive operators see the
    same memory geometry as the original leaf.
    """

    detached = value.detach()
    if any(
        stride == 0 and size > 1 for stride, size in zip(value.stride(), value.shape, strict=True)
    ):
        base = detached
        for dimension, (stride, size) in enumerate(zip(value.stride(), value.shape, strict=True)):
            if stride == 0 and size > 1:
                base = base.narrow(dimension, 0, 1)
        copy = base.clone(memory_format=torch.preserve_format).expand(value.shape)
    else:
        copy = torch.empty_strided(
            value.shape, value.stride(), dtype=value.dtype, device=value.device
        )
        copy.copy_(detached)
    if value.requires_grad and (copy.is_floating_point() or copy.is_complex()):
        copy.requires_grad_(True)
    return copy
