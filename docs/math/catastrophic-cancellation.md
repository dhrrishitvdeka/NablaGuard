# What is catastrophic cancellation?

When large positive and negative floating-point values are added, their leading
digits can cancel and leave a small result whose remaining digits contain much
less information. For `x = [10^8, 1, -10^8]`, the mathematical sum is one, but
an evaluation in limited precision may lose the middle term.

For scalar reductions NablaGuard reports

```text
1 - abs(sum(x)) / sum(abs(x))
```

Values near one mean that most input magnitude disappeared in the result. This
is an observation, not proof of error: centered data and conservation laws can
cancel legitimately. A higher-precision shadow disagreement provides stronger
evidence. Remedies include pairwise or compensated summation, rescaling, and
performing only the sensitive reduction in a wider dtype.
