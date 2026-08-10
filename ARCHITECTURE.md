# Architecture

## Product boundary

PyTorch computes derivatives. NablaGuard verifies and explains them. The engine
is deterministic numerical tooling; it has no LLM, database, web service, or
dashboard dependency.

## Shared core

Every subsystem emits immutable `NablaIssue` values. Context-local `Session`
objects combine issues from checking, tracing, and sanitizing without global
mutable state. `TensorEvent` stores scalar metadata only. `max_events` and
`max_issues` bound session memory; dropped counts are reported when limits hit.

## Contracts and reports

Contracts are explicit predicates over a `ContractContext`; they do not infer
model-specific thresholds. A failed predicate becomes the same immutable
`NablaIssue` used by every subsystem. Guards evaluate tensor contracts at
observed boundaries. Capture evaluates loss, gradient, parameter, and history
contracts after a completed step and persists their issue dictionaries beside
the step metadata. Optional fail-fast behavior and artifact callbacks are
caller-controlled.

Every public result intended for reporting exposes `to_dict()`. JSON is the
canonical machine representation. HTML escapes all user-derived content and is
self-contained; JUnit maps issue identities to failing test cases. Report diffs
compare stable diagnostic and location identity fields, not mutable message or
evidence wording.

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

The sanitizer combines module hooks, explicit `observe` calls, and a curated
eager `TorchDispatchMode`. The mode sees selected sensitive ATen operations and
records scalar output statistics, source positions, module context, and
metadata-only upstream event IDs. Light mode disables dispatch. Standard mode
instruments without shadow execution. Deep mode promotes floating operands and
re-executes registered operations in FP64 by default.

Shadow comparison reports absolute, relative, finite-state, and supported ULP
errors. Pre-operation heuristics use mathematical dtype limits, such as
`log(finfo.max)` for exponential overflow. Scalar sum cancellation uses
`1 - abs(sum(x)) / sum(abs(x))`. Heuristics are labeled observations or risks,
not root-cause claims.

The dispatch mechanism uses a private PyTorch API and is officially eager-only.
No correctness claim is made for `torch.compile`, distributed execution, or
custom kernels until compatibility tests establish one.

## Precision audit

Precision auditing deep-copies the model for each user-ordered candidate dtype
and compares selected module outputs with a copied FP64 reference. It recommends
the first measured dtype satisfying both absolute and relative budgets. The
capture has an element bound and reports skipped modules. Because a whole model
runs at each dtype, a module's observed error can include upstream propagation;
the result is empirical placement guidance, not an isolated kernel proof.

## Capture and replay boundaries

A full checkpoint at step N means model, optimizer, scheduler, scaler, extra
state, and RNG *after* step N. Capture entry writes boundary 0. Completed step
records contain loss, batch identity, user-selected tensor fingerprints,
data-loader metadata, and the post-step RNG digest/state. Periodic full
checkpoints bound restoration cost while per-step metadata stays small.

Fingerprints compute statistics and a SHA-256 checksum over at most a configured
number of evenly spaced elements. Each fingerprint records `checksum_scope`
(`full` or `sampled`) and `statistics_scope`. Sampling non-contiguous large
tensors uses coordinate indexing and does not first materialize a complete
contiguous copy. A sampled checksum match does not prove full-tensor equality.

Replay restores the nearest full checkpoint at or before the requested boundary,
then calls user code for every intervening step. The callback is responsible for
reconstructing data and external state from metadata. Fingerprints and RNG
digests produce `MATCH`, `DIVERGENCE`, or `UNVERIFIED`. `ReplayResult.passed`
requires every requested step to be `MATCH`. Trusted local checkpoint loading
uses Python pickle through `torch.load`; untrusted run directories must not be
replayed.

## Training bisection and boundary diagnosis

The generic search primitive verifies one known-good and one known-bad endpoint,
then performs a logarithmic false-to-true binary search. It assumes the supplied
predicate is monotonic. After search, small intervals are fully re-checked and
larger intervals are spot-checked; inconsistent outcomes become
`BISECT_NON_MONOTONIC` evidence and mark the result as failed.

Captured-run bisection either evaluates JSON metadata directly or constructs
fresh model/optimizer objects for every probe, restores the nearest full
checkpoint, and replays to the midpoint. Reusing mutable model state between
probes is forbidden by the factory API.

Diagnosis compares N−1 with N: loss, trigger batch identity, checksum presence,
and min/max/mean/std/norm fingerprint changes. It ranks relative scalar changes
but labels each `OBSERVED`, not causal. Uncaptured gradient, activation,
optimizer, data, and external state remain explicitly `UNKNOWN`.

## Per-sample gradient geometry

Per-sample tracing selects named parameters directly or through parameter-name
globs. It calculates the exact storage requirement `samples × selected parameter
elements` before allocating and refuses requests beyond `max_gradient_elements`.
Microbatch forwards are shared when the loss function returns an unreduced
leading batch dimension; scalar-reduced losses fall back to individual forwards
and report that cost.

Each scalar sample loss produces one `autograd.grad` VJP. Temporary vectors yield
norm share, cosine to the summed batch gradient, exact cancellation, and bounded
pairwise duplicate-direction checks. The final report stores scalar evidence only.
RNG, buffers, and existing selected `.grad` values are restored in `finally`.
Stochastic per-sample gradients are an analysis run and need not equal the draws
that occurred in a separate training forward.

## Advanced derivative verification

The base backward check compares a VJP under either all-ones or seeded random
cotangents. Opt-in JVP uses `torch.autograd.functional.jvp` with seeded tangents.
Double-backward comparison differentiates the summed first-gradient vector.
Central finite differences compare the candidate's analytical VJP with its own
perturbed forward objective under an element budget. Determinism repeats the
candidate after restoring identical Python, NumPy, CPU, and CUDA RNG state.

Each advanced check builds fresh leaves and graphs. One failure cannot consume
another check's autograd graph, and the entire operator analysis restores caller
RNG state. Passing sampled JVP/VJP directions does not prove complete Jacobian or
Hessian equality. Finite differences remain sensitive to epsilon and conditioning.

Triton and CUDA-extension wrappers already fit the callable candidate boundary;
no native Triton dependency or unsupported introspection is introduced. The
`triton` extra is Linux-only. CPU eager remains the baseline; compile-eager and
hardware-conditional CUDA tests provide scoped compatibility evidence only.

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
