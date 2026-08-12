# Compatibility evidence

## Production-supported baseline

CPU **eager** execution with Python 3.10+ and PyTorch ≥2.2 is the production
support baseline. Local validation metrics (Windows, Python 3.13, PyTorch CPU):

- Unit suite plus coverage gate (`fail_under = 85`). Counts live in CI.
- BugBench checked-in corpus: detection and false-positive rates are fixture
  metrics, not a population estimate.
- `benchmarks/suite.py` records controlled detection rates and guard-mode
  overhead on representative CPU workloads.
- Observation non-interference: deterministic regressions for light/standard/deep
  guards and capture (losses, parameters, buffers, gradients, optimizer, RNG).
- Fuzz trials keep `requires_grad` so backward mismatches are reported.
- `nabla replay` loads pickle checkpoints only with `--i-trust-this-run`.
- NGF inspect refuses drive-qualified and parent-escaping inventory paths and
  does not hash files above the inspection size cap.

Representative workloads include a `TransformerEncoderLayer` backward under light
monitoring and an AdamW CNN step with capture + contracts. Operator checks cover
non-contiguous layouts, mixed `requires_grad`, stochastic pairing, and
hardware-conditional CUDA when present.

## Experimental / unsupported

| Capability | Status | Notes |
|---|---|---|
| CUDA | EXPERIMENTAL | Smoke/conditional tests only; no multi-GPU or full sanitizer/capture matrix |
| FP16 / BF16 / AMP | EXPERIMENTAL | No production claim |
| `torch.compile` / Inductor | EXPERIMENTAL | Compile-eager smoke only |
| Triton internals | UNSUPPORTED | Callable wrappers may still be checked as ordinary ops |
| DDP / FSDP / multi-process | UNSUPPORTED | Single-process only |
| Untrusted checkpoints | UNSUPPORTED | Capture/replay uses pickle; load only trusted local runs |
| Long-run soak (10k+ steps) | DEFERRED | Not required for CPU-eager production claim |

Replay validates captured observables and cannot reconstruct arbitrary external
services or omitted data-loader state. Fingerprint checksums may be sampled;
see `checksum_scope` on each fingerprint record.
