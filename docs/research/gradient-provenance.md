# Gradient provenance research log

## Problem

Explain how named scalar losses contribute to a selected parameter gradient.

## Existing PyTorch mechanisms

`torch.autograd.grad` computes selected VJPs without populating `.grad`.
Backward hooks observe accumulated gradients but do not identify loss terms.

## Possible approaches

Run one VJP per loss, hook backward traversals, or symbolically inspect the
autograd graph. Separate VJPs have the clearest mathematical meaning.

## Experiments and results

For losses already built into one graph, calling `autograd.grad` with
`retain_graph=True` preserves the user's subsequent backward. Detached copies
avoid retaining the graph. Time scales with the number of losses; memory scales
with selected parameter size times loss count.

## Decision

Version 0.1 computes exact separate VJPs. Parameter selection is explicit but
can fall back to reachable autograd leaf discovery for small examples.

## Limitations

Shared stochastic forward computations are not rerun, gradient hooks may alter
the VJP, and many losses or whole-model selection can be expensive. A component
norm share is not an additive contribution to the final norm.

## References

- PyTorch `torch.autograd.grad` documentation
- The triangle inequality underlying the cancellation metric

