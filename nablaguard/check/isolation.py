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
    """Clone an ``nn.Module`` callable or bound module method."""

    if isinstance(function, torch.nn.Module):
        return cast(Callable[..., Any], copy.deepcopy(function))
    owner = getattr(function, "__self__", None)
    if isinstance(owner, torch.nn.Module):
        cloned_owner = copy.deepcopy(owner)
        method_name = getattr(function, "__name__", None)
        if isinstance(method_name, str):
            cloned_method = getattr(cloned_owner, method_name)
            if callable(cloned_method):
                return cast(Callable[..., Any], cloned_method)
    return function
