# Higher-precision shadow execution research log

## Problem

Finite low-precision outputs may already contain unacceptable error before a NaN
or Inf appears. The sanitizer needs an experimental reference without doubling
an entire model's storage or execution by default.

## Existing PyTorch mechanisms

Module hooks expose boundaries but not internal `exp`, reduction, or division
operations. `TorchDispatchMode` observes ATen overloads in eager execution.
Autocast changes execution policy rather than providing a simultaneous reference.

## Approaches and experiments

Whole-model FP64 replay is broad but expensive and often unsupported. A curated
dispatch registry can promote operands only for `sum`, `mean`, variance, norms,
`exp`, `log`, softmax, division, matrix operations, and related reductions.
Experiments on PyTorch 2.13 confirmed the internal softmax and reduction ATen
overloads are visible to a mode.

## Decision

Deep mode re-executes registered eager operations in a configured shadow dtype.
It compares absolute error, relative error, finite state, and ULP distance where
the real dtype has a supported integer representation. Real training tensors and
shadow outputs are released after scalar evidence is recorded.

## Limitations

The Python dispatch API is private and adds material runtime cost. Promotion can
change overload support. Random, stateful, mutating, distributed, compiled, and
custom-kernel operations are not shadowed. Passing one operation does not prove
end-to-end stability.

