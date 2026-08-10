from pathlib import Path

import pytest
import torch

import nablaguard as ng
from nablaguard.check.specs import DEFAULT_DISTRIBUTIONS, DEFAULT_LAYOUTS, TensorSpec


def test_shape_strategy_is_seeded_and_bounded() -> None:
    strategy = ng.shapes(ranks=(2, 3), dimensions=(7, 8, 17), max_elements=1000)
    recipe = ng.tensor(shape=strategy, dtype=[torch.float64])

    first = ng.check.fuzz(
        candidate=lambda x: x,
        reference=lambda x: x,
        inputs=[recipe],
        trials=5,
        seed=41,
    )
    second = ng.check.fuzz(
        candidate=lambda x: x,
        reference=lambda x: x,
        inputs=[recipe],
        trials=5,
        seed=41,
    )

    assert first.to_dict() | {"elapsed_seconds": 0} == second.to_dict() | {"elapsed_seconds": 0}
    assert first.passed


@pytest.mark.parametrize("distribution", DEFAULT_DISTRIBUTIONS)
def test_value_distributions_are_reproducible_and_finite(distribution: str) -> None:
    spec = TensorSpec((4, 5), torch.float16, distribution=distribution)
    first_generator = torch.Generator().manual_seed(7)
    second_generator = torch.Generator().manual_seed(7)

    first = spec.generate(first_generator)
    second = spec.generate(second_generator)

    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


@pytest.mark.parametrize("layout", DEFAULT_LAYOUTS)
def test_layout_recipes_preserve_shape_and_reach_operator(layout: str) -> None:
    spec = TensorSpec((3, 4), torch.float64, layout=layout)
    result = ng.check.operator(
        candidate=lambda x: x.sin(),
        reference=lambda x: x.sin(),
        inputs=[spec],
    )

    assert result.passed
    assert result.metadata["input_shapes"] == [[3, 4]]
    if layout == "broadcasted":
        assert 0 in result.metadata["input_strides"][0]
    elif layout != "contiguous":
        assert result.metadata["input_strides"][0] != [4, 1]


def test_fuzz_finds_and_minimizes_shape_dependent_failure(tmp_path: Path) -> None:
    def candidate(x: torch.Tensor) -> torch.Tensor:
        return x + 1 if x.shape[-1] >= 8 else x

    strategy = ng.tensor(
        shape=ng.shapes((64, 13, 17), ranks=(3,), dimensions=(17,)),
        dtype=[torch.float64],
        distribution=["normal"],
        layout=["contiguous"],
    )
    result = ng.check.fuzz(
        candidate=candidate,
        reference=lambda x: x,
        inputs=[strategy],
        trials=1,
        seed=99,
        artifact_dir=tmp_path,
    )

    assert not result.passed
    failure = result.failures[0]
    assert failure.seed >= 0
    assert failure.minimal_specs[0].shape == (8,)
    assert failure.artifact_path is not None
    assert failure.artifact_error is None
    assert (failure.artifact_path / "manifest.json").is_file()
    assert (failure.artifact_path / "fingerprints.json").is_file()
    # Private-by-default: no raw tensors on disk.
    assert not (failure.artifact_path / "inputs" / "minimized_inputs.pt").exists()
    assert not (failure.artifact_path / "inputs" / "inputs.pt").exists()
    fingerprints = (failure.artifact_path / "fingerprints.json").read_text(encoding="utf-8")
    assert "minimized[0]" in fingerprints
    assert "Minimal known failing shapes" in result.format()


def test_reference_rejections_are_skipped() -> None:
    def unsupported(x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("unsupported domain")

    result = ng.check.fuzz(
        candidate=lambda x: x,
        reference=unsupported,
        inputs=[TensorSpec((3,))],
        trials=3,
    )

    assert result.passed
    assert result.cases_run == 0
    assert result.skipped_cases == 3


def test_reference_free_property_reports_violation() -> None:
    @ng.property
    def impossible_translation(x: torch.Tensor):
        return ng.equivalent(x + 1, x)

    result = ng.check.fuzz(
        candidate=lambda x: x,
        inputs=[TensorSpec((4,))],
        properties=[impossible_translation],
        trials=1,
        minimize=False,
    )

    assert not result.passed
    assert result.failures[0].issues[0].code == "NG3005"


def test_softmax_translation_property_passes() -> None:
    @ng.property(name="softmax translation invariance")
    def translation_invariance(x: torch.Tensor):
        return ng.equivalent(torch.softmax(x + 3, -1), torch.softmax(x, -1))

    result = ng.check.fuzz(
        candidate=lambda x: torch.softmax(x, -1),
        inputs=[TensorSpec((4, 7), torch.float64)],
        properties=[translation_invariance],
        trials=3,
    )

    assert result.passed


def test_fuzz_emits_failure_into_shared_session() -> None:
    with ng.Session() as session:
        ng.check.fuzz(
            candidate=lambda x: x + 1,
            reference=lambda x: x,
            inputs=[TensorSpec((2,))],
            trials=1,
            minimize=False,
        )

    assert [issue.code for issue in session.issues] == ["NG3001"]
