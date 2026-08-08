"""Audit module-boundary dtype error against a float64 reference."""

import torch

import nablaguard as ng

model = torch.nn.Sequential(
    torch.nn.Linear(8, 8),
    torch.nn.LayerNorm(8),
    torch.nn.Softmax(dim=-1),
)
sample = torch.randn(4, 8)

report = ng.precision.audit(
    model,
    sample,
    candidate_dtypes=(torch.float16, torch.bfloat16, torch.float32),
    max_relative_error=1e-3,
)
report.print()
