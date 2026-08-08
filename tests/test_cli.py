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
