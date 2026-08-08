"""Locate and diagnose a captured monotonic loss boundary."""

from pathlib import Path

import torch

import nablaguard as ng
from nablaguard.bisect import metric_greater_than

model = torch.nn.Linear(1, 1, bias=False)
with ng.capture(
    model,
    root=Path(".nabla/example-runs"),
    checkpoint_every=4,
) as recorder:
    for step in range(1, 17):
        loss = float(step) if step < 9 else float(step * 10)
        with torch.no_grad():
            model.weight.add_(0.01)
        recorder.record_step(
            step=step,
            loss=loss,
            batch_indices=[step * 100 + index for index in range(4)],
            tensors={"layer.weight": model.weight},
        )

result = ng.bisect(
    recorder.run_path,
    metric_greater_than("loss", 50),
    known_good=0,
    known_bad=16,
)
result.print()
