# Why FP16 overflows

IEEE binary16 has five exponent bits. Its largest finite value is 65,504, so
`exp(x)` overflows when `x` is greater than approximately
`log(65504) = 11.09`. A logit of 20 is unremarkable in float32 but becomes
infinity when exponentiated directly in float16.

Stable softmax subtracts the largest logit before exponentiation; log-sum-exp
uses the same identity. Mixed precision commonly performs selected reductions
or normalization in float32. NablaGuard checks an exponential input against the
mathematical limit of its actual dtype before inspecting the output, separating
an overflow risk from the later non-finite propagation.
