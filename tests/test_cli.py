from pathlib import Path

import torch

import nablaguard as ng
from nablaguard.cli.main import main


def test_cli_check_exit_code_pass(capsys) -> None:
    exit_code = main(
        [
            "check",
            "torch:sin",
            "--reference",
            "torch:sin",
            "--shape",
            "4",
            "--seed",
            "7",
        ]
    )

    assert exit_code == 0
    assert "Result: PASS" in capsys.readouterr().out


def test_cli_check_exit_code_failure(capsys) -> None:
    exit_code = main(
        [
            "check",
            "torch:neg",
            "--reference",
            "torch:positive",
            "--shape",
            "4",
        ]
    )

    assert exit_code == 1
    assert "Result: FAIL" in capsys.readouterr().out


def test_cli_sanitize_runs_script_without_modification(tmp_path: Path, capsys) -> None:
    script = tmp_path / "unstable.py"
    script.write_text(
        "import torch\ntorch.exp(torch.tensor([100.0], dtype=torch.float32))\n",
        encoding="utf-8",
    )

    exit_code = main(["sanitize", str(script)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "OVERFLOW_RISK" in output
    assert str(script) in output


def test_cli_run_accepts_capture_after_script_and_writes_json(tmp_path: Path) -> None:
    script = tmp_path / "healthy.py"
    output = tmp_path / "report.json"
    script.write_text("import torch\ntorch.logsumexp(torch.tensor([1.0]), 0)\n", encoding="utf-8")

    exit_code = main(["run", str(script), "--capture", "--format", "json", "--output", str(output)])

    assert exit_code == 0
    assert '"passed": true' in output.read_text(encoding="utf-8")


def test_cli_trace_passes_remaining_arguments_to_script(tmp_path: Path) -> None:
    script = tmp_path / "arguments.py"
    observed = tmp_path / "observed.txt"
    script.write_text(
        "import pathlib, sys, torch\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.argv[2], encoding='utf-8')\n"
        "torch.exp(torch.tensor([1.0]))\n",
        encoding="utf-8",
    )

    exit_code = main(["trace", str(script), str(observed), "payload"])

    assert exit_code == 0
    assert observed.read_text(encoding="utf-8") == "payload"


def test_cli_bisect_exit_code_pass(tmp_path: Path) -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with ng.capture(model, root=tmp_path, run_id="cli-bisect", checkpoint_every=8) as recorder:
        for step in range(1, 5):
            recorder.record_step(step=step, loss=float(step))

    exit_code = main(
        [
            "bisect",
            str(recorder.run_path),
            "--metric",
            "loss",
            "--greater-than",
            "2",
            "--known-good",
            "0",
            "--known-bad",
            "4",
        ]
    )
    assert exit_code == 0


def test_cli_replay_requires_trust_outside_default_root(tmp_path: Path) -> None:
    exit_code = main(
        [
            "replay",
            str(tmp_path / "missing-run"),
            "--model-factory",
            "torch.nn:Linear",
            "--step-function",
            "torch:sin",
        ]
    )
    assert exit_code == 3


def test_cli_check_writes_junit(tmp_path: Path) -> None:
    output = tmp_path / "check.xml"
    exit_code = main(
        [
            "check",
            "torch:neg",
            "--reference",
            "torch:positive",
            "--shape",
            "4",
            "--format",
            "junit",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 1
    assert 'failures="' in output.read_text(encoding="utf-8")
