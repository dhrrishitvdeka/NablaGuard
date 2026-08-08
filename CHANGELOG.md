# Changelog

## 0.5.0 — Unreleased

- Added generic monotonic first-bad-step binary search.
- Added captured metadata and checkpoint-aware replay probe modes.
- Added fresh model/optimizer factories so probes cannot share mutated state.
- Added git-bisect-style probe history and restoration-cost evidence.
- Added adjacent-boundary loss, batch, and tensor-fingerprint diagnosis.
- Labeled boundary changes observed and causality unknown.
- Added metric predicates, `nabla bisect`, examples, and accuracy benchmark.

## 0.4.0

- Added layered full checkpoints, step metadata, and batch/data-state capture.
- Added Python, NumPy, PyTorch CPU, and available CUDA RNG capture and restoration.
- Added environment metadata with explicit determinism limitations.
- Added bounded-content tensor fingerprints for divergence localization.
- Added nearest-checkpoint restore and callback-driven interval replay.
- Added exact fingerprint/RNG validation, environment diffs, and first divergence.
- Added replay accuracy benchmark and `nabla replay` factory/callback integration.

## 0.3.0

- Added selective eager ATen instrumentation for sensitive numerical operations.
- Added FP64 shadow execution with absolute, relative, finite-state, and ULP evidence.
- Added mathematical exponential overflow, underflow, and sum-cancellation checks.
- Added source locations, module context, and metadata-only propagation links.
- Added bounded, experimental model precision auditing against an FP64 reference.
- Added zero-modification `nabla sanitize` script execution and overhead benchmarks.

## 0.2.0

- Added bounded, boundary-heavy shape and tensor input strategies.
- Added dtype, value-distribution, and non-contiguous layout generation.
- Added deterministic operator fuzzing with invalid-reference case skipping.
- Added reusable reference-free numerical properties.
- Added bounded shape/layout/distribution/dtype failure minimization.
- Added minimized failure artifacts and the `nabla check` command.

## 0.1.0

- Added the shared event, issue, configuration, and session foundation.
- Added seeded forward and VJP operator verification with failure artifacts.
- Added multi-loss gradient magnitude, cosine, and cancellation reports.
- Added bounded tensor statistics, module monitoring, and NaN/Inf detection.
- Added console/JSON reporting, examples, tests, and an overhead benchmark.
