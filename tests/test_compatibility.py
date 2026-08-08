import pytest
import torch

import nablaguard as ng


def test_eager_cpu_is_supported_baseline() -> None:
    result = ng.check.operator(
        candidate=lambda x: torch.log_softmax(x, -1),
        reference=lambda x: torch.log_softmax(x, -1),
        inputs=[ng.tensor(shape=(4, 7), dtype=torch.float64)],
        check_jvp=True,
    )
    assert result.passed


def test_torch_compile_eager_backend_smoke() -> None:
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    try:
        compiled = torch.compile(lambda x: x.sin() * x, backend="eager")
        result = ng.check.operator(
            candidate=compiled,
            reference=lambda x: x.sin() * x,
            inputs=[ng.tensor(shape=(4,), dtype=torch.float64)],
        )
    except Exception as error:
        pytest.skip(f"torch.compile eager backend unsupported here: {error}")
    assert result.passed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_eager_smoke_when_hardware_is_available() -> None:
    result = ng.check.operator(
        candidate=lambda x: x.square(),
        reference=lambda x: x.square(),
        inputs=[ng.tensor(shape=(16,), dtype=torch.float64, device="cuda")],
        check_jvp=True,
    )
    assert result.passed
