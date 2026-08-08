"""Detect the first unsafe FP16 exponential before downstream propagation."""

import torch

import nablaguard as ng

scores = torch.tensor([2.0, 8.0, 12.0], dtype=torch.float16)

with ng.guard(
    mode="deep",
    operations=["aten.exp*", "aten.sum*", "aten.div*"],
) as monitor:
    exponentials = torch.exp(scores)
    probabilities = exponentials / exponentials.sum()
    loss = -torch.log(probabilities[0])

monitor.print()
print("Final loss:", loss)
