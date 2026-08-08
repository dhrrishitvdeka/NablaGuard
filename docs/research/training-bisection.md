# Training bisection research log

## Problem

Given a good training boundary and a later bad boundary, locate the first bad
step without replaying the entire interval for every investigation.

## Existing mechanisms

Git bisect searches monotonic commit predicates. NablaGuard capture supplies
periodic restorable checkpoints and per-step metadata; replay can reconstruct a
midpoint from the nearest earlier full boundary.

## Approaches and experiment

Metadata predicates are cheap but limited to captured evidence. Arbitrary Python
predicates require model state, so each probe must construct fresh objects,
restore the nearest checkpoint, and replay forward. A controlled threshold at
step 5 is found exactly in metadata and replay modes. The generic search uses at
most `ceil(log2(interval))` midpoint probes plus endpoint checks.

## Decision

Expose both modes behind one `ng.bisect` API. Require the user to label a known
good and known bad endpoint. Require factories for replay probes so mutable state
cannot leak between decisions.

## Limitations

Binary search assumes one monotonic false-to-true transition and cannot prove
that assumption logarithmically. Replay limitations still apply. Boundary
fingerprint differences are observations, not causal explanations. Sparse
metadata cannot support direct step bisection.

