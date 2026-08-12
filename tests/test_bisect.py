from pathlib import Path

import pytest
import torch

import nablaguard as ng
from nablaguard.bisect import first_bad, metric_greater_than
from nablaguard.cli.main import main


def _metadata_run(root: Path) -> Path:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    with ng.capture(
        model,
        root=root,
        run_id="metadata",
        checkpoint_every=4,
    ) as recorder:
        for step in range(1, 9):
            with torch.no_grad():
                model.weight.add_(1)
            recorder.record_step(
                step=step,
                loss=float(step),
                batch_indices=[1000 + step],
                tensors={"layer.weight": model.weight},
            )
    return recorder.run_path


def test_generic_search_checks_endpoints_and_is_logarithmic() -> None:
    evaluated: list[int] = []

    def predicate(step: int) -> bool:
        evaluated.append(step)
        return step >= 731

    result = first_bad(0, 1024, predicate)

    assert result.first_bad_step == 731
    assert len(evaluated) <= 20
    assert result.probes
    assert result.monotonicity_ok


def test_non_monotonic_predicate_is_reported() -> None:
    def predicate(step: int) -> bool:
        # Oscillates: bad at 3, good at 4, bad from 5.
        return step == 3 or step >= 5

    result = first_bad(0, 8, predicate)

    assert result.first_bad_step is not None
    assert result.monotonicity_violations
    assert not result.monotonicity_ok


def test_metadata_bisect_finds_first_bad_and_diagnoses_boundary(tmp_path: Path) -> None:
    run_path = _metadata_run(tmp_path)
    result = ng.bisect(
        run_path,
        metric_greater_than("loss", 4),
        known_good=0,
        known_bad=8,
    )

    assert result.first_bad_step == 5
    assert not result.checkpoint_aware
    assert result.diagnosis.trigger_batch == (1005,)
    assert result.diagnosis.observations[0].label == "OBSERVED"
    assert result.diagnosis.unknowns
    assert result.issues[0].code == "NG4003"
    assert "Causality: UNKNOWN" in result.format()


def test_checkpoint_aware_bisect_restores_and_replays_each_probe(tmp_path: Path) -> None:
    run_path = _metadata_run(tmp_path)

    def model_factory() -> torch.nn.Module:
        return torch.nn.Linear(1, 1, bias=False)

    def step_fn(model, optimizer, step, metadata):
        del optimizer, step, metadata
        with torch.no_grad():
            model.weight.add_(1)
        return {"layer.weight": model.weight}

    result = ng.bisect(
        run_path,
        lambda state: bool(state.model is not None and state.model.weight.item() >= 5),
        known_good=0,
        known_bad=8,
        model_factory=model_factory,
        step_fn=step_fn,
    )

    assert result.first_bad_step == 5
    assert result.checkpoint_aware
    assert all(probe.checkpoint_step is not None for probe in result.probes)
    assert any(probe.replayed_steps > 0 for probe in result.probes)


def test_bisect_missing_metadata_is_an_error(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with ng.capture(
        model,
        root=tmp_path,
        run_id="sparse",
        metadata_every=4,
        checkpoint_every=8,
    ) as recorder:
        for step in range(1, 9):
            recorder.record_step(step=step, loss=float(step))

    with pytest.raises(FileNotFoundError, match="metadata missing"):
        ng.bisect(
            recorder.run_path,
            metric_greater_than("loss", 4),
            known_good=0,
            known_bad=8,
        )


def test_bisect_rejects_incorrect_endpoint_labels(tmp_path: Path) -> None:
    run_path = _metadata_run(tmp_path)

    with pytest.raises(ValueError, match="known_good"):
        ng.bisect(run_path, lambda state: state.step >= 0, known_good=0, known_bad=8)
    with pytest.raises(ValueError, match="known_bad"):
        ng.bisect(run_path, lambda state: False, known_good=0, known_bad=8)


def test_bisect_issue_uses_shared_session(tmp_path: Path) -> None:
    run_path = _metadata_run(tmp_path)
    with ng.Session() as session:
        ng.bisect(run_path, metric_greater_than("loss", 4), known_bad=8)

    assert [issue.code for issue in session.issues] == ["NG4003"]


def test_cli_bisect_metric_predicate(tmp_path: Path, capsys) -> None:
    run_path = _metadata_run(tmp_path)

    exit_code = main(
        [
            "bisect",
            str(run_path),
            "--metric",
            "loss",
            "--greater-than",
            "4",
            "--known-bad",
            "8",
        ]
    )

    assert exit_code == 0
    assert "FIRST BAD STEP: 5" in capsys.readouterr().out
