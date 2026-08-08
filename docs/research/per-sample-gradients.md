# Per-sample gradient analysis research log

## Problem

Identify samples that dominate, oppose, cancel, or nearly duplicate the selected
batch update without implying that per-sample gradients are cheap.

## Existing PyTorch mechanisms

Repeated `torch.autograd.grad` computes exact selected VJPs. `torch.func` and
`vmap` can vectorize functional models but have operator and mutation constraints.

## Approaches and experiment

Version 0.6 shares a microbatch forward when `loss_fn` returns an unreduced
leading sample dimension, then runs one VJP per scalar sample loss. Reduced losses
fall back to individual forwards. A controlled batch with gradients `[1, 2, -1]`
reports magnitude shares `[25%, 50%, 25%]`, cosine −1 for the opposing sample,
and 50% cancellation.

## Decision

Prefer the broadly compatible autograd path first. Require parameter/layer and
sample selection for large models. Calculate and enforce retained-gradient
element count before execution. Bound the quadratic duplicate search separately.

## Limitations

Runtime scales with sample count. Memory scales with selected parameter elements
times selected samples. Dropout and other stochastic layers produce analysis-run
draws; RNG restoration prevents side effects but cannot reconstruct a prior
training forward. Distributed sharding and compiled graphs are unverified.

