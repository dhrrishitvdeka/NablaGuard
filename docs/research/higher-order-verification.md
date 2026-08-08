# JVP, VJP, and higher-order verification research log

## Problem

One all-ones VJP can miss Jacobian errors orthogonal to its cotangent, and a
first-order pass says nothing about forward-mode or higher-order correctness.

## Existing PyTorch mechanisms

Autograd computes VJPs; `torch.autograd.functional.jvp` computes JVPs through
supported operators; `create_graph=True` makes first gradients differentiable;
finite differences offer an implementation-independent local approximation.

## Approaches and experiments

Version 0.7 adds a seeded random VJP direction, seeded JVP tangent, an
all-ones-vector Hessian product, and central finite differences of the summed
candidate output. The deliberately incorrect `x²` backward fails JVP, second
order, and finite-difference checks while its forward still passes.

## Decision

Keep advanced checks opt-in and separately reported. Rebuild fresh graphs per
check. Bound finite-difference input elements. Repeat determinism checks after
restoring identical RNG state, so ordinary seeded randomness is not mislabeled.

## Limitations

Sampled directions do not prove full Jacobian/Hessian equality. Functional JVP
and double backward require operator support. Finite differences depend on dtype,
epsilon, conditioning, and smoothness. Same-process repetition cannot expose all
cross-device or distributed nondeterminism.

