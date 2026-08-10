# Compatibility evidence

## Production-supported baseline

CPU **eager** execution with Python 3.10+ and PyTorch ≥2.2 is the production
support baseline. Local validation metrics (Windows 11, Python 3.13.14,
PyTorch 2.13.0+cpu):

- Unit suite: 149 passed, 1 CUDA-conditional skip, **88%** line coverage of
  `nablaguard`.
- BugBench checked-in corpus: exit 0, detection rate 1.0, false-positive rate
  0.0 (controlled fixtures; not a population estimate).
- `benchmarks/suite.py`: controlled detection FP/FN rates 0.0; standard-mode
  overhead ratio ≈11.9× (comparable to the checked-in baseline).
- Observation non-interference: deterministic regressions for light/standard/deep
  guards and capture (losses, parameters, buffers, gradients, optimizer, RNG).

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
services or omitted data-loader state.
