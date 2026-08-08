from pathlib import Path

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
