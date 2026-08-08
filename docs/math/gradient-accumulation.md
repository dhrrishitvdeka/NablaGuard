# How backpropagation accumulates gradients

If a parameter influences several paths or losses, reverse-mode automatic
differentiation adds their vector-Jacobian products. For losses `L_i` and a
parameter `theta`,

```text
grad_theta sum_i L_i = sum_i grad_theta L_i.
```

PyTorch also adds each backward result into `parameter.grad`; it does not clear
that buffer automatically. This is useful for deliberate microbatch
accumulation and dangerous when `zero_grad()` is forgotten. NablaGuard's loss
trace computes each component with `autograd.grad`, preserves the ordinary
backward path, and reports norms and pairwise geometry without treating a large
component as an additive percentage of the final vector.
