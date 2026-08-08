# Changelog

## 0.3.0 — Unreleased

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
