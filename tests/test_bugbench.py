from __future__ import annotations

import json
from pathlib import Path

from nablaguard.benchmark import BugBenchObservation, CaseContext, run_bugbench
from nablaguard.benchmark.bugbench import json_dumps
from nablaguard.benchmark.overhead import run_overhead_benchmark
from nablaguard.cli.main import main


def detected_fixture(context: CaseContext) -> BugBenchObservation:
    return BugBenchObservation(
        True,
        category="BACKWARD_MISMATCH",
        module="custom_square",
        stage="backward",
        evidence={"seed": context.seed, "infinite_error": float("inf")},
        baseline_seconds=1.0,
        instrumented_seconds=1.25,
        original_failure_size=64,
        minimized_failure_size=2,
    )


def safe_fixture(context: CaseContext) -> BugBenchObservation:
    del context
    return BugBenchObservation(False)


def test_bugbench_derives_metrics_from_ground_truth(tmp_path: Path) -> None:
    root = tmp_path / "bugbench"
    category = root / "autograd"
    category.mkdir(parents=True)
    _write_case(
        category / "NG-BUG-0001.yaml",
        identifier="NG-BUG-0001",
        implementation="tests.test_bugbench:detected_fixture",
        detected=True,
        expected_category="BACKWARD_MISMATCH",
        module="custom_square",
        stage="backward",
    )
    _write_case(
        category / "NG-CTRL-0001.yaml",
        identifier="NG-CTRL-0001",
        implementation="tests.test_bugbench:safe_fixture",
        detected=False,
        expected_category=None,
        module=None,
        stage=None,
    )

    report = run_bugbench(root, seed=71)
    metrics = report.metrics

    assert report.passed
    assert metrics["detection_rate"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["localization_accuracy"] == 1.0
    assert metrics["diagnostic_accuracy"] == 1.0
    assert metrics["runtime_overhead"]["median"] == 1.25
    assert metrics["failure_minimization_effectiveness"]["mean_reduction"] == 0.96875
    encoded = json_dumps(report)
    assert '"runtime_overhead"' in encoded
    assert '"infinite_error": "Infinity"' in encoded


def test_bugbench_cli_uses_ci_exit_taxonomy(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing"

    exit_code = main(["benchmark", "bugbench", "--root", str(missing)])

    assert exit_code == 3
    assert "Invalid BugBench configuration" in capsys.readouterr().err


def test_overhead_benchmark_measures_actual_guarded_model_path() -> None:
    report = run_overhead_benchmark(
        quick=True,
        device="cpu",
        workloads=["tiny_mlp"],
        modes=["light"],
    )

    workload = report.workloads["tiny_mlp"]
    assert workload["baseline"]["wall_seconds"] > 0
    assert workload["light"]["wall_clock_ratio"] > 0
    assert workload["light"]["python_heap_overhead_bytes"] >= 0


def _write_case(
    path: Path,
    *,
    identifier: str,
    implementation: str,
    detected: bool,
    expected_category: str | None,
    module: str | None,
    stage: str | None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": {"name": "nablaguard.bugbench.case", "version": 1},
                "id": identifier,
                "benchmark_category": "autograd",
                "description": "test fixture",
                "implementation": implementation,
                "expected": {
                    "detected": detected,
                    "category": expected_category,
                    "module": module,
                    "stage": stage,
                },
                "requirements": {"device": "cpu"},
                "tags": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
