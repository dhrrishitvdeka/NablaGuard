"""First NablaGuard vertical slice: a correct forward with a broken backward."""

import torch

import nablaguard as ng


class BadFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x**2

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # Deliberately wrong: derivative of x**2 is 2*x.
        return grad_output * x


result = ng.check.operator(
    candidate=BadFunction.apply,
    reference=lambda x: x**2,
    inputs=[ng.tensor(shape=(32,), dtype=torch.float64)],
)
result.print()

if result.passed:
    raise SystemExit("The deliberately broken backward was not detected")
