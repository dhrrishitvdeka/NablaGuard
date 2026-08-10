# NablaGuard industrialization audit

**Original audit date:** 2026-08-09  
**Re-audit date:** 2026-08-11  
**Audited revision:** local main (v1.0.0 hardening)  
**Scope:** public APIs, internal abstractions, events and issues, numerical tracing,
operator verification, gradient tracing, capture and replay, bisection, contracts,
reporting, CLI, tests, packaging, documentation, security, and performance assumptions.

## Executive verdict

CPU-eager production baseline is supported with measured unit, BugBench, and suite
evidence (**156** tests, ~88% coverage, BugBench detection 1.0 on the checked-in corpus,
suite standard-mode ≈11.9×). NGF private-by-default artifacts, verified-only replay
pass, selective capture snapshots, light-mode sample bounds, and non-interference
regressions close the original P0 table for that baseline.

### 2026-08-11 residual closure

| Prior residual | Status after re-audit |
|---|---|
| Silent shadow failures | **CLOSED** — `NG1005` / `SHADOW_UNSUPPORTED` |
| `id(tensor)` provenance reuse | **CLOSED** — weakref cleanup of producer map |
| Advanced `_leaf` invents `requires_grad` | **CLOSED** — preserves caller flag; FD only on grad inputs |
| Fingerprint sampling ambiguity | **CLOSED** — `checksum_scope` / `statistics_scope` fields |
| Unbounded issue lists | **CLOSED** — `max_issues` + `dropped_issues` |
| `default=str` JSON | **CLOSED** — strict `normalize_json` / `dumps_json` |
| Non-atomic report writes | **CLOSED** — `atomic_write_text` |
| Bisect always exit 0 / no mono check | **CLOSED** — `NG4004`, `passed`, CLI exit 1 |
| Pytest exit codes leak | **CLOSED** — mapped to 0/1/3 |
| Missing `py.typed` | **CLOSED** |
| Mutable CI action tags | **CLOSED** — pinned commit SHAs + coverage gate |
| Capture pickle residual | **ACCEPTED** with CLI trust warning + `--i-trust-this-run` |
| Evidence graph / DDP / CUDA matrix | **OPEN** (deferred / experimental) |

NablaGuard remains a functional PyTorch debugging library with a coherent issue type,
operator checker, deterministic input generation, bounded events, layered checkpoints,
and a broad unit-test foundation. CUDA, AMP, BF16, Inductor, DDP, FSDP, Triton,
activation checkpointing, and long-run soak still lack production evidence and are
explicitly experimental/unsupported.

Historical note: the original audit of `e1e5b15` correctly rejected an unproven
“fully industrial” claim. Later hardening closed CPU-eager P0s without upgrading
experimental hardware paths. The published `v1.0.0` tag was a functional baseline;
current `main` is evidence that the
industrial release gates below have passed. Semantic versions cannot be rolled back.
Until the gates are satisfied, project documentation and package metadata must not
describe NablaGuard as production-ready. Backward-compatible hardening can ship in
`1.x`; changes that cannot preserve the public or persisted-data contract require a
documented migration and may require `2.0`.

### Evidence labels

- **OBSERVED** — directly established by code, tests, or checked-in benchmark output.
- **INFERRED** — a technically supported risk that still needs a targeted experiment.
- **UNKNOWN** — no adequate evidence exists.

### System map

| Concern | Current implementation | Audit conclusion |
|---|---|---|
| Shared runtime | `core.Session`, `TensorEvent`, `NablaIssue` | Useful in-process base; not a versioned evidence model |
| Numerical guard | module hooks plus private `TorchDispatchMode` | Eager-oriented and synchronization-heavy |
| Operator verification | forward, VJP, JVP, double backward, finite difference, determinism, fuzzing | Strongest subsystem; correctness isolation gaps remain |
| Gradient provenance | multi-loss and per-sample VJPs | Mathematically useful; expensive storage and lifetime behavior |
| Capture | periodic checkpoints plus per-step JSON, RNG and fingerprints | Correct direction; current parameter-copy and I/O cost is unsafe at scale |
| Replay | checkpoint restore and fingerprint/RNG comparison | Partial behavioral verifier; no rigorous fidelity levels |
| Bisect | scalar monotonic binary search, optional replay probes | Useful primitive; not hierarchical failure localization |
| Reporting and CLI | console, JSON, HTML, JUnit | Good breadth; schemas and CI exit taxonomy are unstable |

## Current strengths

1. **OBSERVED — shared issue vocabulary.** `NablaIssue` and `Severity` are reused by
   sanitizer, trace, check, replay, bisection, and contracts. Reports do not depend on
   an LLM, and diagnostic messages generally distinguish observations from suggestions.

2. **OBSERVED — operator verification has real numerical depth.** The checker supports
   forward comparison, seeded VJPs, JVPs, double backward, finite differences,
   determinism checks, shape/value/dtype/layout generation, and deterministic greedy
   shrinking. It preserves non-contiguous strides for many inputs and records seeds.

3. **OBSERVED — randomized analyses try to avoid global RNG interference.** Operator
   checks capture and restore Python, NumPy, CPU, and CUDA RNG state, while input
   generation uses private `torch.Generator` instances.

4. **OBSERVED — capture separates full checkpoints from per-step metadata.** The design
   already uses periodic state checkpoints, per-step RNG evidence, selected fingerprints,
   and batch indices instead of saving a complete checkpoint each step.

5. **OBSERVED — bounded event count and metadata-only event fields.** `Session` caps the
   number of tensor events, records dropped-event count, and `TensorEvent` itself does not
   directly own tensor storage.

6. **OBSERVED — atomic persistence primitives exist.** Core JSON and PyTorch writers use
   temporary files followed by replacement. This is the correct basis for crash-safe
   manifests and artifacts.

7. **OBSERVED — failure-focused automated tests are broad for the current scope.** The
   release baseline had 99 passing tests, one hardware-conditional CUDA skip, strict
   Ruff and mypy checks, and 88.19% line coverage. Deliberately broken backward,
   numerical, replay, bisection, tracing, and contract fixtures exist.

8. **OBSERVED — limitations are sometimes stated honestly.** Benchmark documentation
   scopes its false-positive and false-negative values to controlled fixtures, reports
   CUDA memory as unmeasured, and does not claim perfect deterministic reproduction.

## Architectural weaknesses

1. **OBSERVED — no unified evidence graph.** Events and issues are independent lists.
   There are no typed operation, tensor, sample, batch, rank, checkpoint, or training-step
   nodes and no versioned `produced_by`, `gradient_from`, `occurred_before`, or
   `reproduced_by` edges. Sanitizer provenance is a local dictionary keyed by Python
   object identity, while replay, bisect, and trace maintain separate models.

2. **OBSERVED — persisted and machine-readable records are not schema contracts.** Event,
   issue, report, checkpoint, replay-step, benchmark, and failure-artifact data use
   ad-hoc dictionaries and `default=str`. Only some objects include `format_version`,
   and no validators or migrations exist.

3. **OBSERVED — public/private boundaries are porous.** Package `__init__` modules export
   many implementation types and helpers. There is no deprecation decorator or warning
   mechanism, compatibility test for public signatures, migration framework, or
   `py.typed` marker.

4. **OBSERVED — runtime state is process-local.** A `ContextVar` plus Python lists cannot
   aggregate ranks, preserve global event order, coordinate worker processes, or build a
   distributed evidence stream. Issue storage is unbounded even when events are capped.

5. **OBSERVED — observation, analysis, and persistence execute inline.** GPU reductions,
   Python issue construction, shadow computation, callbacks, JSON encoding, and disk
   writes occur on the training thread. There is no summary queue, ring buffer, async
   writer, backpressure policy, or failure-isolated worker.

6. **OBSERVED — internal and user failures lack a structural boundary.** Exceptions from
   contract callbacks, reporters, shadow rules, candidate/reference functions, and
   replay callbacks are handled inconsistently. No `NG-INTERNAL-*` model proves whether
   NablaGuard changed user training state before failing.

7. **OBSERVED — compatibility is prose rather than a tested capability registry.** The
   existing compatibility page names a few smoke tests but does not use the required
   `SUPPORTED`, `EXPERIMENTAL`, `PARTIAL`, and `UNSUPPORTED` matrix or link every claim to
   a test environment.

## Correctness risks

| Severity | Finding | Evidence and consequence |
|---|---|---|
| Critical | Unverified replay can pass | `ReplayResult.passed` only rejects `DIVERGENCE` and `ERROR`; a sequence containing only `UNVERIFIED` steps returns `True`. Downstream bisection can accept a boundary that was never numerically verified. |
| Critical | Training non-interference is unproven | There is no suite comparing complete loss, gradients, parameters, optimizer state, buffers, and RNG progression with and without every observation-only mode. Instrumentation that changes RNG or state could silently change the job it diagnoses. |
| High | Candidate and reference do not begin from identical RNG state | The base operator check executes candidate then reference sequentially. The outer RNG snapshot restores state only after the complete check, so stochastic but equivalent implementations can compare different random draws. |
| High | Candidate and reference state are not isolated | Mutable modules, global state, aliased inputs, or custom operators can alter state before the reference runs. This can create false positives or hide bugs. |
| High | Complex comparisons discard information | `compare_tensors` casts complex tensors to `float64`, dropping the imaginary component. Complex forward or gradient mismatches are therefore not trustworthy. |
| High | Input gradient semantics are changed | `_leaf_copy` sets `requires_grad=True` for every floating or complex tensor, even if a supplied tensor did not require gradients. Non-differentiable arguments and mixed differentiability cannot be modeled accurately. |
| High | Shadow execution is not semantically isolated | Generic FP64 shadow rules re-execute operations without snapshotting RNG or mutable state. Exceptions are silently ignored rather than recorded as `UNKNOWN` or `UNSUPPORTED`. |
| High | Tensor provenance uses reusable object IDs | `_tensor_producers[id(tensor)]` may associate a later tensor with an old producer after Python object-ID reuse. It is not a durable graph identity. |
| High | Fingerprint sampling can miss divergence | The checksum samples elements, but the field name does not distinguish a sampled digest from a full digest. A change outside sampled positions may be missed; scalar statistics do not prove equality. |
| Medium | Broad exception paths blur responsibility | Fuzzing/reference rejection and callback failures can classify NablaGuard errors as operator failures or omit unsupported regions without a machine-readable reason. |
| Medium | Bisect assumes a monotonic predicate | The scalar binary search is valid only for a good-to-bad transition. Oscillating losses, intermittent nondeterminism, and divergent ranks violate the assumption and are not detected. |

PyTorch itself states that bitwise equality is not guaranteed across releases,
platforms, or CPU/GPU even with identical inputs. Replay must therefore report a
measured fidelity level rather than a Boolean promise; see the official
[reproducibility note](https://docs.pytorch.org/docs/stable/notes/randomness.html) and
[numerical accuracy note](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).

## Performance risks

| Severity | Location | Observed cost | Required direction |
|---|---|---|---|
| Critical | `capture/recorder.py` | Clones every named parameter on entry and after every recorded step, even when no parameter-change contract needs it | Select only contracted parameters; use fingerprints or sampled deltas; never copy a whole large model per step by default |
| Critical | `sanitize/statistics.py` | Scans full tensors, allocates finite masks and filtered tensors, promotes to FP64, and calls multiple `.item()` reductions | Device-side fused summaries, bounded sampling, configurable cadence, and one deliberate synchronization point |
| High | `capture/recorder.py` | Captures all RNG states and writes JSON plus a `.pt` RNG file synchronously at every metadata step | Batched metadata, asynchronous bounded writer, retention policy, measured flush latency |
| High | `capture/fingerprints.py` | Hash sampling is bounded, but min/max/mean/std/norm still scan the entire tensor | Make full reductions optional and distinguish sampled from full evidence |
| High | `trace/samples.py` | Stores a flattened full gradient vector per selected sample and clones model buffers and existing gradients | Parameter/module selection, random sample selection, top-k streaming, byte budgets, and chunked CPU summaries |
| High | sanitizer deep mode | Re-executes selected operations after dtype promotion and compares full outputs | Explicit eligible-op registry, byte/shape budgets, sampled comparisons, and measured memory overhead |
| Medium | gradient reports and contracts | Repeated `.item()` and repeated norm/statistic computations introduce CPU/GPU synchronization | Cache shared summaries per evidence node and batch scalar transfers |
| Medium | `Session` and reports | Events are count-bounded but issue count and serialized report size are not | Byte-aware ring buffers, issue aggregation, truncation evidence, and size caps |
| Medium | checkpoint-aware bisection | Probe-state caching can retain several full model and optimizer states | Disk-backed or single-state replay with an explicit memory budget |

The official [CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html)
documentation explains that CUDA work is asynchronous and that CPU/GPU copies trigger
necessary synchronization. The current scalar extraction pattern makes light-mode cost
especially sensitive to tensor count and device latency. The existing 11.94x standard
CPU ratio is a useful warning signal, not a GPU overhead estimate.

No current benchmark measures all required quantities: wall-clock slowdown, GPU
utilization, CPU time, RAM, VRAM, disk I/O, time-to-detection, and instrumentation loss
for a tiny MLP, CNN, Transformer, mixed-precision workload, and large synthetic network.

## Missing abstractions

1. A versioned evidence graph shared by guard, trace, check, capture, replay, bisect,
   sanitization, and reporting.
2. A versioned NGF manifest, strict validators, compatibility readers, migrations, and
   explicit completion state for atomically constructed artifacts.
3. Replay fidelity levels L0–L4 with separate model, optimizer, batch, RNG, tolerance,
   fingerprint, and bitwise evidence.
4. Runtime policies for sampling, cadence, synchronization budget, byte budget, ring
   buffers, asynchronous sinks, retention, and degradation behavior.
5. Structured `USER_MODEL_FAILURE`, `INVALID_CONFIGURATION`, and
   `NABLAGUARD_INTERNAL_FAILURE` results with stable error codes.
6. A data backend protocol for streaming batches, projection, predicate pushdown,
   provenance identity, and optional Arrow/Polars/Vaex/IterableDataset adapters.
7. A privacy policy object, secret/path redactors, raw-tensor opt-in, sanitizer, and
   artifact-size enforcement.
8. A distributed event envelope containing run, rank, local sequence, monotonic time,
   step, batch, collective boundary, and summary/hash fields.
9. A capability registry that binds every compatibility status to tested versions,
   hardware, execution mode, and limitations.
10. A limited, stable plugin protocol for checks, predicates, generators, reporters,
    data engines, and capture policies without exposing core internals.
11. Regression baseline schemas for numerical error, gradients, runtime, memory, and
    determinism.

## Technical debt

- `default=str` appears in core serialization and all report encoders. It hides invalid
  schema values instead of rejecting them and can persist sensitive object
  representations.
- The sanitizer depends on private `torch.utils._python_dispatch.TorchDispatchMode`.
  Private PyTorch APIs have no compatibility promise from NablaGuard's declared
  `torch>=2.2` minimum through unbounded future versions.
- `LossTrace` deliberately leaves the latest trace in a context variable, retaining
  detached gradient copies beyond the context lifetime.
- Loss histories and contract issue lists grow without a retention bound.
- Contract predicates and evidence factories can calculate the same norm twice and run
  arbitrary callbacks inline.
- Artifact writers bypass the existing atomic serialization helpers.
- Failure artifact IDs are content-derived and directories use `exist_ok=True`, so
  concurrent identical failures can write the same incomplete directory.
- The CLI uses `parse_known_args` for script forwarding, which can consume NablaGuard-
  shaped options intended for the user script.
- `nabla check path.py` delegates to pytest and returns pytest's exit codes, conflating
  test-runner failures with NablaGuard's required 0/1/2/3 CI contract.
- Output reports are written non-atomically. SARIF, baseline, benchmark, artifact
  inspection, and artifact sanitization commands are absent.
- Strict mypy skips following `torch.*` imports, weakening checks at the most important
  integration boundary. Pyright and a PEP 561 marker are absent.
- CI installs unconstrained latest dependencies on Ubuntu/Python 3.10–3.12 only. It has
  no minimum/maximum PyTorch matrix, Windows/macOS, Python 3.13, CUDA, AMP, multiprocess
  distributed, or Inductor jobs and no enforced coverage threshold.
- GitHub Actions are pinned to mutable major tags rather than immutable commit SHAs.
  Release automation, provenance, signing, dependency review, and security scanning are
  absent.

## Compatibility limitations

The current evidence supports only a narrow claim: CPU eager execution is the primary
tested baseline. One `torch.compile(..., backend="eager")` callable smoke test does not
establish TorchDynamo, AOTAutograd, or Inductor correctness and non-interference. One
conditional CUDA `sin` check does not establish CUDA sanitizer, capture, replay, AMP, or
memory behavior.

| Capability | Audited status | Evidence required to improve status |
|---|---|---|
| PyTorch eager / CPU / FP32 | PARTIAL | representative end-to-end non-interference, version matrix, overhead bounds |
| FP64 operator checks | PARTIAL | complex fix, stochastic/state isolation, reference-domain accounting |
| CUDA | EXPERIMENTAL | real GPU CI across sanitizer, check, capture, replay, and overhead |
| FP16 / BF16 / AMP | EXPERIMENTAL | numerical, scaler, autocast, replay, and non-interference suites on capable hardware |
| `torch.compile` | EXPERIMENTAL | eager-vs-compiled semantics, gradient equality, graph-break reporting, AOTAutograd and Inductor tests |
| custom `autograd.Function` | PARTIAL | state/RNG isolation, edge cases, higher-order capability reporting |
| `torch.library` custom ops | UNSUPPORTED | integrate `torch.library.opcheck`, FakeTensor, schema, AOT-dispatch and device coverage |
| Triton | UNSUPPORTED | optional real-kernel fixtures across masks, strides, edge shapes, dtypes, forward/backward |
| DDP | UNSUPPORTED | multiprocess CPU protocol tests plus optional multi-GPU integration |
| FSDP | UNSUPPORTED | real sharded parameter/gradient/state tests and documented hook limitations |
| activation checkpointing | UNSUPPORTED | RNG, recomputation, hook-count, gradient, and non-interference tests |
| out-of-core data | UNSUPPORTED | backend protocol and 1M/10M/100M-row measurements |

PyTorch's [`torch.library.opcheck`](https://docs.pytorch.org/docs/stable/library.html)
tests schema, autograd registration, FakeTensor, and AOT-dispatch behavior; it is
complementary to mathematical gradient checking and should be incorporated rather than
reimplemented. The [`torch.compile` programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html)
documents graph breaks and custom-operator boundaries that NablaGuard must surface
without silently changing semantics.

Distributed support cannot be inferred from single-process mocks. PyTorch DDP launches a
separate training process per rank, as described by the
[`torch.distributed` documentation](https://docs.pytorch.org/docs/stable/distributed).
FSDP replaces managed parameters with views and has explicit backward-hook and
double-backward limitations; see the official
[FSDP reference](https://docs.pytorch.org/docs/stable/fsdp.html). Those behaviors affect
parameter identity, gradient observation, capture, and replay design.

For out-of-core data, PyTorch documents that each worker receives a replica of an
`IterableDataset` and naive multiprocess loading can duplicate data. Any `ng.data`
provenance contract must therefore include worker/rank sharding and seed evidence; see
[`torch.utils.data`](https://docs.pytorch.org/docs/stable/data.html).

## Security concerns

1. **Critical — raw tensors are stored by default for checker failures.** Supplying an
   artifact directory writes every input and minimized input to `.pt` files. There is no
   `raw_tensors=False` default, consent boundary, redactor, tensor-count limit, byte cap,
   compression policy, or retention policy.

2. **Critical — replay loads unrestricted pickle.** Checkpoint restore explicitly calls
   `torch.load(..., weights_only=False)`. PyTorch warns never to load data from an
   untrusted source because `torch.load` uses unpickling. The
   [official `torch.load` reference](https://docs.pytorch.org/docs/stable/generated/torch.load.html)
   and [serialization security note](https://docs.pytorch.org/docs/main/notes/serialization.html)
   also clarify that `weights_only=True` narrows but does not eliminate denial-of-service
   or memory-corruption risks. Artifact inspection must not execute artifact code.

3. **High — user-controlled run IDs can escape the capture root.** `Path(root) / run_id`
   accepts absolute paths and parent traversal. Run IDs require a strict identifier
   grammar plus resolved-path containment checks.

4. **High — secrets can enter JSON through unrestricted metadata.** Hyperparameters,
   extra state, data state, event tags, issue evidence, callback errors, and object
   representations are serialized without key-based secret filtering. Environment data
   includes executable and platform identity; replay errors may expose paths or data.

5. **High — artifacts have no size controls.** A failure on large or numerous tensors can
   exhaust disk or create an accidental diagnostic dump large enough to disrupt a job.

6. **Medium — no authenticity or integrity boundary exists.** Manifests do not include a
   complete file inventory, digests, completion marker, or optional signature. A partial
   or modified artifact can be mistaken for trusted evidence.

7. **Medium — CI supply-chain controls are incomplete.** Mutable action tags and
   unconstrained dependency resolution reduce reproducibility and expand the build
   trust surface.

## Production blockers

Priority is ordered by industrialization sequence and silent-evidence risk.
Production-ready means the **CPU-eager baseline** only (156 tests, ~88% coverage,
BugBench exit 0). CUDA/AMP/compile/distributed remain experimental unless noted.

| Priority | Blocker | Exit evidence | Status (CPU-eager) |
|---|---|---|---|
| P0 | Replay can pass without verified observables | No Boolean pass for L0/unverified runs; negative tests | **CLOSED** — `passed` requires all `MATCH` |
| P0 | Observation non-interference is unproven | Deterministic matrix across observation-only modes | **CLOSED** — guard modes + capture regressions |
| P0 | Light mode is not designed or measured as a low-overhead runtime | Bounded stats + measured ratios | **CLOSED** — sample bounds; light ≈1.8× on `tiny_mlp` |
| P0 | Capture copies a full model per step | Selective parameter snapshot | **CLOSED** — snapshot only when contracts require it |
| P0 | Raw artifacts and unrestricted pickle are unsafe defaults | NGF private-by-default, redaction, size, safe inspect | **CLOSED** for NGF; checkpoints remain **trusted-local pickle only** with CLI trust gate |
| P0 | Operator candidate/reference isolation is insufficient | Stochastic/stateful/complex regressions | **CLOSED** — isolation + pairing + requires_grad preservation |
| P1 | No credible BugBench | Ground-truth corpus + runner metrics | **CLOSED** for checked-in corpus (detection 1.0 on fixtures) |
| P1 | No shared evidence graph | Versioned graph across subsystems | OPEN (deferred) |
| P1 | No real modern-execution evidence | compile/AMP/CUDA matrix | OPEN — labeled experimental |
| P1 | CI result contract is incomplete | Exit 0–3, JSON/JUnit | **CLOSED** for 0/1/2/3 + JSON/JUnit; SARIF still absent |
| P1 | Distributed behavior is absent | DDP/FSDP | OPEN — unsupported |
| P2 | No out-of-core data/provenance layer | Backend protocol | OPEN (deferred) |
| P2 | No soak or self-fuzz evidence | Long-run suites | OPEN (deferred) |
| P2 | No confirmed real-world discoveries/corpus | Licensed corpus | OPEN (deferred) |
| P2 | Public API and persisted schemas are unstable | Manifests + migrations | **PARTIAL** — public API + NGF v1 + `py.typed`; broader migrations deferred |

### Industrial release-gate assessment

| Required gate | Status (current main) |
|---|---|
| Stable public API | **PASS** (documented surface; CPU-eager) |
| Credible BugBench results | **PASS** (checked-in corpus; fixture-scoped metrics) |
| Bounded light-mode overhead | **PASS** (CPU measured; CUDA null) |
| Tested CUDA support | EXPERIMENTAL / unproven |
| Tested mixed precision | EXPERIMENTAL / unproven |
| Tested `torch.compile` behavior | EXPERIMENTAL (eager backend smoke only) |
| Custom operator verification | **PASS** for ordinary callables / Autograd.Function |
| Triton integration | UNSUPPORTED |
| Failure artifact compatibility and migration | **PASS** (NGF v1 inspect/sanitize/migrate) |
| Replay validation and fidelity | **PASS** for verified MATCH; trusted checkpoints only |
| CI exit codes and machine-readable integration | **PASS** for 0/1/2/3 + JSON/JUnit (SARIF deferred) |
| Security review and private defaults | **PASS** for NGF private defaults + redaction + run_id containment + replay trust gate; residual pickle threat on capture |
| Long-run stability | DEFERRED |
| Real-world bug discoveries | DEFERRED |
| Documented false positives and false negatives | PARTIAL; controlled-fixture caveat retained |
| Documented compatibility and limitations | **PASS** (CPU-eager claim; experimental elsewhere) |

No gate with `MISSING` / unproven experimental evidence may be converted into a
support claim by documentation alone. `PARTIAL` means useful implementation exists,
not that every industrial sub-gate has passed.

## Required execution order

The audit supports the requested phase order without exception:

1. Build BugBench first so later changes have honest detection and localization measures.
2. Redesign and benchmark light/standard/deep policies, then prove non-interference.
3. Specify versioned, private-by-default NGF artifacts and rigorous replay fidelity.
4. Harden operator verification before adding compile, `torch.library`, or Triton claims.
5. Add `torch.compile` and Triton through isolated official-API prototypes and real tests.
6. Build the backend-neutral out-of-core data and provenance layer.
7. Stabilize CI exit/report contracts, then harden security boundaries.
8. Add real multiprocess DDP followed by FSDP and distributed evidence aggregation.
9. Run soak tests and self-fuzzing before real-world bug hunting and corpus publication.
10. Re-audit every industrial release gate before any production-ready claim.

This document deliberately makes no large implementation change. It establishes the
baseline, risks, and proof obligations that subsequent focused commits must satisfy.
