# Deterministic replay research log

## Problem

Reproduce the interval in which training first diverged without storing a full
checkpoint after every step or claiming impossible universal determinism.

## Existing PyTorch mechanisms

`state_dict()` covers model, optimizer, schedulers, and GradScaler. Python,
NumPy, PyTorch CPU, and CUDA expose RNG state. PyTorch deterministic-algorithm
mode reduces but does not eliminate environment and custom-operator differences.

## Approaches

Full checkpoints every step are prohibitively large. Metadata-only capture
cannot restore model state. A layered design uses periodic full boundaries plus
per-step RNG, batch identity, and selected tensor fingerprints.

## Experiment and decision

Boundary 0 is saved when capture begins. A checkpoint at N represents state
after N. Replay restores the nearest boundary, then invokes user code for every
next step using recorded metadata. Exact bounded-content fingerprints and a
content-stable RNG digest validate each boundary. A deterministic CPU fixture
replays exact weights and RNG across periodic checkpoints.

## Limitations

The callback must reconstruct data and all external side effects. Data-loader
worker process state is not automatically captured. CUDA, custom kernels,
asynchronous execution, distributed collectives, files, services, and mutable
globals may diverge. Fingerprints sample large tensors and can miss unsampled
changes. Checkpoints are trusted pickle-based local artifacts and must not be
loaded from untrusted sources.

