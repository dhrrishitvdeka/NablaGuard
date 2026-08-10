# Changelog

## Unreleased

- Hardened audit follow-ups for the CPU-eager baseline: shadow failures emit
  `NG1005` / `SHADOW_UNSUPPORTED` instead of silent skips; tensor provenance
  drops stale object ids via weak references; fingerprints record
  `checksum_scope` (`full` | `sampled`) and `statistics_scope`.
- Session issue lists are bounded (`max_issues`, `dropped_issues`); reports use
  strict JSON (`normalize_json`, no `default=str`); tensors in evidence are
  summarized without raw values; report files write atomically.
- Advanced checks preserve input `requires_grad` (finite differences only cover
  inputs that already require grad). Loss traces expose `release()` to drop
  retained gradient copies after `gradient()` use.
- Bisection verifies monotonicity on small intervals and reports
  `NG4004` / `BISECT_NON_MONOTONIC` with `passed=false` when outcomes oscillate.
- CLI exit taxonomy is `0` pass, `1` check-fail, `2` usage, `3` error; pytest
  path checks map into that contract; `bisect` fails on non-monotonic results;
  `replay` warns about pickle checkpoints and requires `--i-trust-this-run`
  outside `.nabla/runs`.
- Package ships `py.typed`; CI pins GitHub Actions to commit SHAs and enforces
  coverage `fail_under = 85`.
- Prior NGF v1, run_id containment, replay MATCH-only pass, selective capture
  snapshots, redaction, and non-interference work remains as previously listed.

## 1.0.0 — 2026-08-09

- Declared the documented top-level API and issue/report schemas stable.
- Shipped operator diffcheck and minimization, eager numerical sanitizing,
  precision auditing, capture/replay, bisection, and gradient tracing together.
- Added runtime training contracts and console, JSON, HTML, JUnit, and diff reports.
- Completed the seven-command CLI, GitHub Actions evidence artifacts, controlled
  regression fixtures, benchmark suite, performance baseline, and math guide.
- Validated on Python 3.12 and PyTorch 2.13 CPU with 99 tests and 88% coverage;
  CUDA remains hardware-conditional and was unavailable for this release run.

## 0.9.0

- Added custom and built-in training contracts integrated with guards and capture.
- Added JSON, self-contained HTML, JUnit XML, and structured report diff support.
- Completed the `run`, `sanitize`, `trace`, `check`, `replay`, `bisect`, and `inspect` CLI.
- Added controlled broken-model regression fixtures and a unified benchmark matrix.
- Documented measured performance, API boundaries, and the underlying mathematics.
- Marked the package as a release candidate pending independent project validation.

## 0.7.0 — Unreleased

- Added seeded random-cotangent VJP and random-tangent JVP checks.
- Added second-order VJP comparison and bounded central finite differences.
- Added same-RNG repeated execution for NG3004 nondeterminism evidence.
- Preserved caller Python, NumPy, CPU, and CUDA RNG state across operator checks.
- Added advanced derivative CLI flags and per-check terminal status.
- Added optional Linux Triton extra and callable-kernel compatibility boundary.
- Added compile-eager, CPU, and hardware-conditional CUDA compatibility tests.
- Added advanced derivative example, cost benchmark, tests, and documentation.

## 0.6.0

- Added bounded per-sample selected-parameter gradient analysis.
- Added layer globs, sample subsets, and microbatch forwards.
- Added dominant magnitude share, cosine-to-batch, conflicts, and duplicates.
- Added exact per-sample cancellation with shared NG2003/NG2004 issues.
- Added pre-allocation gradient-element limits and performance warnings.
- Restored RNG, module buffers, and existing gradients after analysis.
- Added per-sample example, benchmark, tests, and research documentation.

## 0.5.0

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
