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

## 0.3 — Numerical sanitizer

Evidence-backed dispatch experiments, selected high-precision shadow rules, and
precision auditing against explicit error budgets.

## Later releases

Training capture and deterministic replay (0.4), checkpoint-aware training
bisection (0.5), per-sample gradient analysis (0.6), and kernel/JVP/higher-order
verification (0.7). Version 1.0 requires a stable API, credible benchmarks,
documented cost, and real-world compatibility evidence.
