"""Built-in parameter update contracts."""

from __future__ import annotations

import torch

from .base import Contract, ContractContext


def change(
    *, min_relative: float = 0.0, max_relative: float | None = None, raise_on_failure: bool = False
) -> Contract:
    """Require relative L2 parameter change to stay within configured bounds."""

    if min_relative < 0 or (max_relative is not None and max_relative < min_relative):
        raise ValueError("invalid parameter relative-change bounds")

    def observed(context: ContractContext) -> float | None:
        if context.parameters is None or context.previous_parameters is None:
            return None
        changes = []
        for name, current in context.parameters.items():
            previous = context.previous_parameters.get(name)
            if previous is None:
                continue
            difference = torch.linalg.vector_norm(
                (current.detach() - previous.detach()).reshape(-1)
            )
            denominator = torch.linalg.vector_norm(previous.detach().reshape(-1)).clamp_min(1e-30)
            changes.append(float((difference / denominator).item()))
        return max(changes) if changes else None

    def predicate(context: ContractContext) -> bool:
        value = observed(context)
        return (
            value is not None
            and value >= min_relative
            and (max_relative is None or value <= max_relative)
        )

    return Contract(
        "parameter.change",
        predicate,
        code="NG4001",
        category="PARAMETER_CHANGE_CONTRACT",
        message="Relative parameter change is outside the configured interval.",
        raise_on_failure=raise_on_failure,
        requires=frozenset({"previous_parameters"}),
        evidence_factory=lambda context: {
            "observed_max_relative_change": observed(context),
            "min_relative": min_relative,
            "max_relative": max_relative,
        },
    )
