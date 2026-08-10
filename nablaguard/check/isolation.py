"""Non-interfering callable execution helpers for operator verification."""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

import torch

IsolationKind = Literal["module_clone", "passthrough"]


def call_with_isolated_module_state(
    function: Callable[..., Any], inputs: Sequence[Any]
) -> Any:
    """Execute a callable on an independent module copy when one owns it."""

    return isolated_callable(function)(*inputs)


def isolation_kind(function: Callable[..., Any]) -> IsolationKind:
    """Report whether a callable will be module-cloned or left shared.

    ``module_clone`` means parameters and buffers are deep-copied. ``passthrough``
    means free functions, closures, and non-module callables share any external
    mutable state with the caller (and between candidate/reference if both are
    passthrough).
    """

    if isinstance(function, torch.nn.Module):
        return "module_clone"
    owner = getattr(function, "__self__", None)
    if isinstance(owner, torch.nn.Module) and isinstance(
        getattr(function, "__name__", None), str
    ):
        return "module_clone"
    return "passthrough"


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
