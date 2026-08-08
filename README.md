# NablaGuard

Debug the math, not just the code.

NablaGuard is a PyTorch-first toolkit for numerical verification, gradient
analysis, differentiable operator testing, deterministic training replay, and
failure bisection. Version 0.3 implements the verification foundation,
tensor-aware diffcheck, selective eager numerical instrumentation, and
experimental precision auditing, layered training capture, and deterministic
replay validation, and checkpoint-aware training bisection.

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

Deep mode re-executes only a curated registry of numerically sensitive ATen
operations in higher precision:

```python
with ng.guard(
    mode="deep",
    shadow_dtype=torch.float64,
    max_relative_error=1e-3,
    operations=["aten.exp*", "aten.sum*", "aten._softmax*"],
) as monitor:
    loss = model(batch).sum()
    loss.backward()

monitor.print()
```

The dispatch engine can identify an FP16 `exp` input beyond `log(finfo.max)`
before reporting its infinite output. Events carry metadata-only upstream IDs;
full tensors are not retained. Standard/deep dispatch is officially eager-only.
Light mode avoids dispatch interception.

Precision recommendations are experiments against a copied float64 model, not
static guesses:

```python
report = ng.precision.audit(
    model,
    sample_input,
    candidate_dtypes=(torch.float16, torch.bfloat16, torch.float32),
    max_relative_error=1e-4,
)
report.print()
```

The audit is bounded by `max_capture_elements`, leaves the original model
unchanged, and does not automatically rewrite it.

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
nabla sanitize train.py --mode deep
```

## Capture and replay training boundaries

```python
with ng.capture(
    model,
    optimizer,
    checkpoint_every=1000,
    metadata_every=1,
) as recorder:
    for step, (x, y) in enumerate(loader, start=1):
        loss = train_step(x, y)
        recorder.record_step(
            step=step,
            loss=loss,
            batch_indices=batch_indices,
            tensors={"layer.weight": model.layer.weight},
        )
```

Capture stores a full state boundary at step 0 and periodically thereafter,
plus step metadata, Python/NumPy/PyTorch RNG state, batch identity, and bounded
tensor fingerprints. The manifest always lists determinism limitations.

```python
result = ng.replay(
    recorder.run_path,
    model=fresh_model,
    optimizer=fresh_optimizer,
    step_fn=replay_one_step,
    from_step=0,
    to_step=500,
)
result.print()
```

The callback must reconstruct data from captured batch metadata and returns the
same named tensors that were fingerprinted. Replay reports exact checksum and
RNG matches, the first divergence, environment differences, or `UNVERIFIED`
when the callback returns no tensors. Restoration alone is never called proof
of determinism.

## Bisect the first bad training boundary

Captured scalar metadata can be searched directly:

```python
from nablaguard.bisect import metric_greater_than

result = ng.bisect(
    run_path,
    metric_greater_than("loss", 10),
    known_good=0,
    known_bad=14281,
)
result.print()
```

Or pass `model_factory` and `step_fn` to restore the nearest checkpoint and
replay each midpoint before applying a Python predicate to `BoundaryState`.
Binary search assumes one monotonic good-to-bad transition; it verifies endpoint
labels but cannot prove unobserved monotonicity in logarithmic time.

Boundary diagnosis compares captured loss, trigger batch, and tensor fingerprint
statistics at N−1 and N. Changes are labeled `OBSERVED`; causality remains
`UNKNOWN` unless established outside this report.

```bash
nabla bisect .nabla/runs/run-id --metric loss --greater-than 10
```

## Analyze individual sample gradients

```python
report = ng.trace.samples(
    model,
    loss_fn,
    (inputs, targets),
    layers=["layer4.*"],
    sample_indices=[0, 4, 8, 12],
    microbatch_size=4,
    max_gradient_elements=5_000_000,
)
report.print()
```

The report ranks sample gradient magnitude share, cosine to the selected batch
gradient, opposing samples, nearly duplicate directions, and exact cancellation
`1 - ||Σgᵢ||₂ / Σ||gᵢ||₂`. It does not call norm share an additive contribution.

Per-sample analysis performs one VJP per selected sample and retains one flattened
gradient vector per selected sample. NablaGuard calculates this allocation before
the first gradient and raises if it exceeds `max_gradient_elements`. Existing
parameter gradients, module buffers, and RNG state are restored afterward.
