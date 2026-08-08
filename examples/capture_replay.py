"""Capture and exactly replay a tiny deterministic training interval."""

from pathlib import Path

import torch

import nablaguard as ng

ROOT = Path(".nabla/example-runs")
torch.manual_seed(42)
model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

with ng.capture(
    model,
    optimizer,
    root=ROOT,
    checkpoint_every=2,
    hyperparameters={"learning_rate": 0.1},
) as recorder:
    for step in range(1, 5):
        features = torch.randn(4, 2, dtype=torch.float64)
        loss = model(features).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        recorder.record_step(
            step=step,
            loss=loss,
            batch_indices=[step],
            tensors={"weight": model.weight},
        )

replay_model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
replay_optimizer = torch.optim.SGD(replay_model.parameters(), lr=0.1)


def replay_step(step: int, metadata: dict):
    del step, metadata
    features = torch.randn(4, 2, dtype=torch.float64)
    loss = replay_model(features).square().mean()
    replay_optimizer.zero_grad()
    loss.backward()
    replay_optimizer.step()
    return {"weight": replay_model.weight}


result = ng.replay(
    recorder.run_path,
    model=replay_model,
    optimizer=replay_optimizer,
    step_fn=replay_step,
    to_step=4,
)
result.print()
