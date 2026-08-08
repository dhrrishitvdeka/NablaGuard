# Roadmap

## 0.1 — Foundation

- Shared event, issue, configuration, and session primitives
- Forward and VJP comparison for differentiable operators
- Optional portable failure artifacts
- Exact multi-loss gradient norms, cosines, and cancellation
- Bounded tensor statistics and NaN/Inf detection
- Console and JSON reports

## 0.2 — Diffcheck (implemented)

Boundary-heavy shape, dtype, value-distribution, and layout generation;
reference-aware property fuzzing; failure shrinking; seed persistence; CLI
checks; minimized failure artifacts.

## 0.3 — Numerical sanitizer (implemented)

Evidence-backed eager dispatch instrumentation, selected high-precision shadow
rules, mathematical overflow/underflow/cancellation checks, propagation metadata,
zero-modification CLI execution, and bounded precision auditing against explicit
error budgets.

## 0.4 — Capture and replay (implemented)

Layered full checkpoints and step metadata; Python, NumPy, CPU/CUDA RNG capture;
environment and determinism limitations; bounded tensor fingerprints; nearest-
checkpoint restoration; exact fingerprint/RNG replay validation and first-
divergence reporting.

## 0.5 — Training bisection (implemented)

Monotonic first-bad binary search; metadata and checkpoint-aware replay probes;
fresh-state factories; git-bisect-style history; trigger-batch and adjacent-step
fingerprint diagnosis with observed/inferred/unknown labeling.

## 0.6 — Advanced gradient trace (implemented)

Selected-parameter and layer-glob per-sample VJPs; sample subsets; microbatch
forwards; dominant, opposing, and duplicate-direction samples; exact cancellation;
pre-allocation memory bounds and performance warnings.

## 0.7 — Kernel engineering (implemented)

Seeded random VJP and JVP comparison; double backward; bounded central finite
differences; identical-RNG nondeterminism checks; reusable metamorphic properties;
non-contiguous layouts; callable kernel boundary and optional Triton extra;
eager, compile-eager, CPU, and hardware-conditional CUDA compatibility tests.

## Toward 1.0

Version 1.0 requires contracts, final reporting/CI integrations, a stable API,
credible benchmarks, documented cost, and real-world compatibility evidence.
