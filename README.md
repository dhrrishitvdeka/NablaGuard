# NablaGuard

Debug the math, not just the code.

NablaGuard is a PyTorch-first toolkit for numerical verification, gradient
analysis, differentiable operator testing, deterministic training replay, and
failure bisection. Version 0.2 implements the verification foundation and
tensor-aware diffcheck; replay and bisection remain later roadmap work.

## Catch a broken backward

This custom function computes `x²` correctly in its forward pass and returns
the wrong derivative in its backward pass:

```python
import torch
import nablaguard as ng


class BadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x**2

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * x  # wrong: expected 2 * x


result = ng.check.operator(
    candidate=BadFunction.apply,
    reference=lambda x: x**2,
    inputs=[ng.tensor(shape=(32,), dtype=torch.float64)],
)
result.print()
```

NablaGuard reports a passing forward comparison and `NG3002
BACKWARD_MISMATCH`, including the worst absolute and relative VJP errors. Every
generated experiment prints its seed; pass the same seed to repeat it.

## Trace competing losses

```python
with ng.trace.losses(
    {"classification": classification, "regularization": regularization},
    parameters=[model.layer2.weight],
) as trace:
    (classification + regularization).backward()

trace.report(model.layer2.weight, name="layer2.weight").print()
```

Cancellation has one documented definition:

```text
1 - ||sum(g_i)||₂ / sum(||g_i||₂)
```

It measures lost component magnitude. It does not claim that cancellation is a
bug or identify a root cause.

## Monitor tensor health

```python
with ng.guard(model, modules=["transformer.blocks.10.*"]) as monitor:
    loss = model(batch).sum()
    loss.backward()

print(monitor.issues)
```

The Version 0.1 guard records bounded scalar metadata (min, max, mean, standard
deviation, absolute max, zero fraction, NaN count, and Inf count). It never
retains module outputs. Magnitude warnings require an explicit threshold, which
keeps model-specific heuristics out of the correctness engine.

## Install and develop

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy nablaguard
python examples/bad_backward.py
python examples/gradient_trace.py
```

PyTorch eager execution on CPU is the supported baseline. CUDA operations work
where ordinary PyTorch autograd works, but this release makes no claim of full
CUDA, `torch.compile`, distributed, Triton, replay, or bisection compatibility.
See [ARCHITECTURE.md](ARCHITECTURE.md) and [ROADMAP.md](ROADMAP.md).

## Fuzz and minimize an operator

```python
strategy = ng.tensor(
    shape=ng.shapes(ranks=(1, 2, 3), dimensions=(7, 8, 16, 17, 32)),
    dtype=[torch.float64, torch.float32, torch.float16],
    distribution=["normal", "tiny", "huge", "mixed_magnitude"],
    layout=["contiguous", "transposed", "strided", "broadcasted"],
)

result = ng.check.fuzz(
    candidate=my_operator,
    reference=reference_operator,
    inputs=[strategy],
    trials=100,
    seed=81927183,
    artifact_dir=".nabla/failures",
)
result.print()
```

Every failing case records its trial seed and concrete recipes. The shrinker
re-executes each proposed shape, dtype, layout, or distribution simplification;
it calls the result “minimal known,” because a bounded greedy search cannot
prove global minimality.

Reference-free properties can return a boolean or an `(actual, expected)` pair:

```python
@ng.property
def softmax_translation_invariance(x):
    return ng.equivalent(torch.softmax(x + 3, -1), torch.softmax(x, -1))
```

Importable callables can be checked in CI with meaningful exit codes:

```bash
nabla check package.ops:my_op --reference torch:sin --shape 32 --trials 100
```
