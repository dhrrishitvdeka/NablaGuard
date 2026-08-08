"""Find and shrink a shape-dependent differentiable operator bug."""

import torch

import nablaguard as ng


def candidate(x: torch.Tensor) -> torch.Tensor:
    """Deliberately fail at a vector width boundary."""

    return x + 1 if x.shape[-1] >= 17 else x


inputs = [
    ng.tensor(
        shape=ng.shapes(ranks=(1, 2, 3), dimensions=(7, 8, 16, 17, 32)),
        dtype=[torch.float64, torch.float32],
        distribution=["normal", "zeros", "mixed_magnitude"],
        layout=["contiguous", "transposed", "strided"],
    )
]

result = ng.check.fuzz(
    candidate=candidate,
    reference=lambda x: x,
    inputs=inputs,
    trials=50,
    seed=81927183,
)
result.print()
