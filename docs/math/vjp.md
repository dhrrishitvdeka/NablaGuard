# What is a VJP?

For `y = f(x)` with Jacobian `J`, a vector-Jacobian product is `v^T J`. Reverse
mode computes this product without materializing `J`; calling backward on `y`
with cotangent `v` is precisely that operation. It is efficient when there are
many inputs and few scalar objectives.

Checking only an all-ones cotangent can miss errors whose columns cancel in
that direction. NablaGuard therefore supports seeded random cotangents. Several
passing directions increase confidence but do not prove equality of every
Jacobian element.
