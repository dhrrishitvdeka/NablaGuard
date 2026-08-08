"""Decompose an update into two deliberately opposing loss gradients."""

import torch

import nablaguard as ng

weight = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float64))
classification = weight.sum()
regularization = -0.75 * weight.sum()

with ng.trace.losses(
    {"classification": classification, "regularization": regularization},
    parameters=[weight],
) as trace:
    (classification + regularization).backward()

trace.report(weight, name="layer2.weight").print()
