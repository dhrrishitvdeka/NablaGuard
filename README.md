<div align="center">
  <img src="docs/assets/nablaguard-hero.svg" alt="NablaGuard — Debug the math, not just the code" width="100%">
  <br>
  <a href="https://github.com/dhrrishitvdeka/NablaGuard/actions/workflows/ci.yml"><img src="https://github.com/dhrrishitvdeka/NablaGuard/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/version-1.0.0-6366f1" alt="Version 1.0.0">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776ab" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c" alt="PyTorch 2.2+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22c55e" alt="MIT license"></a>
</div>

## Debug the math, not just the code

NablaGuard is a PyTorch-first toolkit for checking differentiable operators,
finding numerical instability, explaining gradient geometry, capturing and
replaying training state, and bisecting the first bad step. Its diagnostics are
computed algorithmically and carry stable issue codes, evidence, and explicit
limitations.

Version 1.0 provides a stable public API and a tested CPU-eager baseline.
Hardware-conditional and external-project scope is stated explicitly rather
than implied by the version number.

| Verify | Observe | Reproduce | Explain |
|:--|:--|:--|:--|
| Forward, VJP, JVP, higher-order and finite-difference checks | Eager numerical sanitizer and precision shadows | Layered capture, deterministic replay and first-bad bisection | Loss and per-sample gradient geometry |

## Install

```bash
pip install -e ".[dev]"
nabla --version
```

## Check a differentiable operator

```python
import torch
import nablaguard as ng


class WrongSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.square()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * x  # expected 2 * x


result = ng.check.operator(
    candidate=WrongSquare.apply,
    reference=lambda x: x.square(),
    inputs=[ng.tensor(shape=(32,), dtype=torch.float64)],
    vjp_cotangent="random",
    check_jvp=True,
    check_finite_difference=True,
)
result.print()
```

Forward, VJP, JVP, double-backward, finite-difference, and same-RNG
nondeterminism checks are available. Every generated case has a private seed;
checks restore caller Python, NumPy, CPU, and CUDA RNG state.

Fuzz boundary-heavy shapes, values, dtypes, and layouts with `ng.check.fuzz`.
Failures can be persisted and greedily minimized to the smallest reproducing
case found within the configured budget.

## Monitor numerical behavior

```python
with ng.guard(
    model,
    mode="deep",
    modules=["transformer.blocks.10.*"],
    operations=["aten.exp*", "aten.sum*", "aten._softmax*"],
) as monitor:
    loss = model(batch).sum()
    loss.backward()

monitor.print()
```

Light mode observes explicit/module boundaries, standard mode intercepts a
curated set of sensitive eager ATen operations, and deep mode additionally
compares selected operations with higher-precision shadow execution. Events
retain bounded scalar metadata, not full activations.

## Assert training contracts

```python
checks = [
    ng.contracts.loss.finite(),
    ng.contracts.gradient.norm(max=100),
    ng.contracts.tensor.finite(module="attention.*"),
    ng.contracts.parameter.change(min_relative=1e-8),
    ng.contracts.training.loss_not_exploding(max_ratio=10, window=20),
]

with ng.capture(model, optimizer, contracts=checks) as recorder:
    loss = train_step()
    recorder.record_step(loss=loss)
```

Contracts emit ordinary `NablaIssue` objects, can fail fast, and can invoke an
artifact callback. Capture persists contract failures with each step.

## Explain gradients

`ng.trace.losses` separates several losses at selected parameters and reports
norms, pairwise cosines, and exact cancellation. `ng.trace.samples` computes
bounded per-sample VJPs, ranks dominant/opposing samples, and restores model
buffers, RNG, and existing gradients afterward.

Cancellation is always

```text
1 - norm(sum_i g_i) / sum_i norm(g_i)
```

It describes lost component magnitude; it does not assign causality.

## Capture, replay, and bisect

```python
with ng.capture(model, optimizer, checkpoint_every=1000) as recorder:
    for step, batch in enumerate(loader, start=1):
        loss = train_step(batch)
        recorder.record_step(
            step=step,
            loss=loss,
            batch_indices=batch.indices,
            tensors={"layer.weight": model.layer.weight},
        )
```

Capture records full state boundaries, RNG state, environment limitations,
batch identity, and bounded tensor fingerprints. `ng.replay` restores the
nearest checkpoint and reports `MATCH`, `DIVERGENCE`, or `UNVERIFIED` for each
boundary. `ng.bisect` performs logarithmic search under an explicit monotonic
good-to-bad assumption and labels adjacent-boundary changes as observations,
not causes.

## CLI and CI reports

```bash
nabla run train.py
nabla run train.py --capture --format json --output run.json
nabla sanitize train.py --mode deep
nabla trace train.py --format html --output trace.html
nabla check package.ops:kernel --reference package.refs:kernel --trials 100
nabla check tests/test_kernel.py
nabla replay .nabla/runs/run-id --model-factory app:model --step-function app:step
nabla bisect .nabla/runs/run-id --metric loss --greater-than 10
nabla inspect .nabla/failures/NGF-example
```

Console, JSON, self-contained HTML, and JUnit XML outputs are supported. Exit
code zero means the requested verification passed; detected issues or failed
checks return nonzero.

## Development and evidence

```bash
ruff check .
mypy nablaguard
pytest --cov=nablaguard --cov-report=term-missing
python benchmarks/suite.py --output benchmark.json
```

See [architecture](ARCHITECTURE.md), [roadmap](ROADMAP.md), the
[mathematical guide](docs/math/README.md), and [measured performance
characteristics](docs/performance.md). CUDA paths are tested only when hardware
is available. `torch.compile(..., backend="eager")` has smoke coverage; native
Triton introspection, distributed training, and arbitrary external state are
outside the current correctness claim.
