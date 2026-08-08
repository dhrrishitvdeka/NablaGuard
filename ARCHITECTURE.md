# Architecture

## Product boundary

PyTorch computes derivatives. NablaGuard verifies and explains them. The engine
is deterministic numerical tooling; it has no LLM, database, web service, or
dashboard dependency.

## Shared core

Every subsystem emits immutable `NablaIssue` values. Context-local `Session`
objects combine issues from checking, tracing, and sanitizing without global
mutable state. `TensorEvent` stores scalar metadata only, and `max_events`
bounds session memory.

## Operator verification

`check.operator` materializes independent leaf copies for candidate and
reference computations. Forward outputs are compared elementwise. Backward
verification then applies identical all-ones cotangents and compares the
resulting vector-Jacobian products for every input. This validates a useful VJP;
it does not yet prove the full Jacobian, higher derivatives, or nondeterminism.

Random inputs use a private `torch.Generator`, so checks do not perturb global
RNG state. Failures can persist exact inputs, environment metadata, structured
evidence, and the seed under an `NGF-*` artifact directory.

## Gradient provenance

`trace.losses` evaluates `torch.autograd.grad` once per named scalar loss before
the user's ordinary `backward`. Detached per-loss gradients are retained only
for selected parameters. Cost is therefore approximately one backward traversal
per named loss plus memory for one gradient copy per loss and selected
parameter. Users should select parameters for large models.

Pairwise cosine similarity is undefined when either gradient has zero norm and
is reported as `NaN`. Cancellation is
`1 - ||sum(g_i)||₂ / sum(||g_i||₂)` and says nothing about causality.

## Numerical sanitizer

The initial sanitizer uses module forward hooks or explicit `observe` calls.
Hooks see module outputs, not every internal ATen operation; therefore the first
reported module is the first observed boundary, not necessarily the originating
operator. Torch dispatch experiments are deferred until their eager, compiled,
and overhead behavior is measured.

## Diffcheck search and shrinking

`TensorStrategy` resolves through a private Python RNG into concrete
`TensorSpec` recipes. Tensor values use per-input private `torch.Generator`
instances. Recipes cover boundary-heavy shapes, floating dtypes, value
distributions, and contiguous, transposed, sliced, strided, or broadcasted
layouts. Candidate and reference receive independent stride-preserving leaves.

The reference defines whether a generated domain is valid: if it rejects a
case, that trial is skipped. If the reference accepts it and the candidate
raises or mismatches, the trial fails. Properties may add reference-free
invariants to the same trial.

Shrinking is bounded and greedy. It tries dimension removal, landmark dimension
sizes, contiguous layout, simpler value distributions, and simpler dtypes. A
candidate is accepted only after the original failure predicate reproduces.
The output is therefore the smallest case found, not a mathematical proof of
minimality.
