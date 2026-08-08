# Tensor operator fuzzing research log

## Problem

Custom differentiable operators often fail only at vector-width boundaries,
particular layouts, low precision, or high dynamic range.

## Existing PyTorch mechanisms

`torch.autograd.gradcheck` provides finite-difference checks for chosen inputs.
`torch.testing.assert_close` compares known outputs. Neither defines a bounded
tensor-specific search space plus portable failure evidence.

## Approaches and experiments

The initial search space emphasizes dimensions near powers of two, primes,
multiple floating dtypes, thirteen distributions, and five layouts. Private
Python and PyTorch RNGs make trials repeatable without altering a training
process's global RNG state. Unsupported reference domains are skipped.

## Decision

Use small native strategies around the shared operator checker. Avoid exposing a
general-purpose property-testing dependency as NablaGuard's API.

## Limitations

Version 0.2 samples finite combinations rather than guaranteeing coverage.
CUDA and compiled execution are not claimed. Multiple arguments are sampled
independently and do not yet express relational shape constraints such as
matrix-multiplication inner dimensions.

