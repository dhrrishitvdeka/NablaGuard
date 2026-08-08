# Property-based testing for tensor operators

Example-based tests sample shapes and values chosen by a person. Tensor bugs
often hide at zero dimensions, prime lengths, vectorization boundaries, mixed
magnitudes, unusual dtypes, or non-contiguous strides. Property-based testing
generates those domains from a reproducible recipe and checks a reference or an
invariant such as translation symmetry.

When a trial fails, NablaGuard greedily tries simpler shapes, layouts, value
distributions, and dtypes, retaining a change only when the failure reproduces.
The result is the smallest case found within the budget, not a proof of global
minimality. The seed and concrete recipe remain part of the artifact.
