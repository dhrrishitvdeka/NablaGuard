# Why deterministic replay is difficult

Reproducing a training step requires more than model weights. Optimizer,
scheduler, scaler, Python/NumPy/PyTorch RNGs, batch identity, data-loader state,
library versions, devices, and nondeterministic kernels can all change the
result. External services and filesystem ordering add further state.

NablaGuard records explicit boundaries and restores captured state, then
compares bounded tensor fingerprints and RNG digests. A match is evidence for
the captured observables; an omitted tensor remains unverified. A mismatch
locates the first observed divergence but does not by itself name its cause.
