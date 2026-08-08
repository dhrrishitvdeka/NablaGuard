"""Find dominant and opposing examples in a tiny batch."""

import torch

import nablaguard as ng

model = torch.nn.Linear(1, 1, bias=False)
inputs = torch.tensor([[1.0], [2.0], [-1.0], [2.0]])

report = ng.trace.samples(
    model,
    lambda output: output.squeeze(-1),
    inputs,
    parameters={"linear.weight": model.weight},
    microbatch_size=4,
)
report.print()
