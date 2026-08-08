"""Built-in gradient contracts."""

from __future__ import annotations

from .base import Contract, ContractContext


def norm(
    *,
    min: float | None = None,
    max: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    raise_on_failure: bool = False,
) -> Contract:
    """Bound the combined L2 norm of named gradients or parameter ``.grad`` values."""

    if min is not None:
        if minimum is not None:
            raise ValueError("pass either min or minimum, not both")
        minimum = min
    if max is not None:
        if maximum is not None:
            raise ValueError("pass either max or maximum, not both")
        maximum = max
    if minimum is None and maximum is None:
        raise ValueError("gradient.norm requires minimum, maximum, or both")
    if minimum is not None and minimum < 0:
        raise ValueError("gradient norm minimum must be non-negative")
    if maximum is not None and maximum < 0:
        raise ValueError("gradient norm maximum must be non-negative")

    def observed(context: ContractContext) -> float | None:
        gradients = context.gradients
        if gradients is None and context.parameters is not None:
            gradients = {
                name: value.grad
                for name, value in context.parameters.items()
                if value.grad is not None
            }
        if not gradients:
            return None
        squared = sum(
            float(value.detach().to("cpu").square().sum().item()) for value in gradients.values()
        )
        return float(squared**0.5)

    def predicate(context: ContractContext) -> bool:
        value = observed(context)
        return (
            value is not None
            and (minimum is None or value >= minimum)
            and (maximum is None or value <= maximum)
        )

    return Contract(
        "gradient.norm",
        predicate,
        code="NG2001" if maximum is not None else "NG2002",
        category="GRADIENT_NORM_CONTRACT",
        message="Combined selected gradient norm is outside the configured interval.",
        raise_on_failure=raise_on_failure,
        evidence_factory=lambda context: {
            "observed_norm": observed(context),
            "minimum": minimum,
            "maximum": maximum,
        },
    )
