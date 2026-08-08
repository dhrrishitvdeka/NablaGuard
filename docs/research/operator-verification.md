# Differentiable operator verification research log

## Problem

Detect operators with correct-looking forward values but incorrect custom
backward formulas.

## Existing PyTorch mechanisms

PyTorch autograd can compute VJPs; `gradcheck` uses finite differences but does
not compare an implementation to a separately trusted reference across a common
developer report and artifact format.

## Possible approaches

Compare full Jacobians, sampled VJPs, all-ones VJPs, or finite differences.

## Experiments and results

An all-ones VJP immediately detects the deliberately incorrect `x²` backward,
is inexpensive, and works for non-scalar outputs. It samples only one cotangent
and can miss errors orthogonal to that direction.

## Decision

Version 0.1 performs deterministic all-ones VJP comparison. Later fuzzing will
add seeded random cotangents and finite-difference/reference cross-checks.

## Limitations

Passing does not prove Jacobian equality, nondeterminism, double backward,
layout support, or compiled compatibility.

