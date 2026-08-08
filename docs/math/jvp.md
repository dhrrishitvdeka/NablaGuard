# What is a JVP?

A Jacobian-vector product is `J v`: the output change predicted for an input
direction `v`. Forward-mode differentiation propagates this tangent alongside
the primal computation and is attractive when the input dimension is small.

VJPs and JVPs view the same Jacobian from opposite sides. Comparing both catches
different implementation mistakes. NablaGuard generates JVP tangents from a
private seed, re-runs candidate and reference on fresh leaves, and reports the
same absolute and relative error evidence used by forward checks.
