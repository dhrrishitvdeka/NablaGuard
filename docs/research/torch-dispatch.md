# Torch dispatch instrumentation research log

## Problem

Find the first selected unsafe eager operation, not merely the next module whose
output is non-finite.

## Existing mechanisms

Forward hooks observe module inputs/outputs; autograd hooks observe backward
gradients; FX sees traceable graphs; `TorchDispatchMode` sees runtime ATen calls.

## Experiments

Against the installed PyTorch 2.13 runtime, dispatch exposed `aten.exp.default`,
`aten.sum` overloads, `aten._softmax`, division, norms, and matrix operations.
Calling the received overload inside `__torch_dispatch__` executes beneath the
current mode, allowing post-operation evidence collection without recursion.

## Decision

Use dispatch only for a curated sensitive registry in standard/deep eager mode.
Use module pre/post hooks to attach module context. A reentrancy flag prevents
NablaGuard's own statistic operations from generating events. Event propagation
stores tensor identity to event-ID mappings, never tensor references.

## Limitations

Tensor identity can only describe observed runtime flow; it is not a durable
autograd graph. Source capture and Python dispatch add overhead. Light mode is
the lower-cost alternative. `torch.compile` and distributed behavior are
explicitly unverified.

