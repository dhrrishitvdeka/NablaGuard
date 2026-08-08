# Finite-difference gradient checking

For a scalar objective `phi`, central differences estimate one component as

```text
(phi(x + epsilon e_i) - phi(x - epsilon e_i)) / (2 epsilon).
```

Central differences cancel the leading truncation error, but an epsilon that is
too large measures curvature while one that is too small loses the subtraction
to rounding. The useful scale depends on dtype and conditioning. NablaGuard
compares the candidate's analytical VJP with bounded central differences and
refuses inputs beyond its element budget. This is independent evidence, not an
exact oracle.
