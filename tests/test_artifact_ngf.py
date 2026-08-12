"""Unit tests for the NGF v1 artifact package."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from nablaguard.artifact import (
    ArtifactPolicy,
    ArtifactSizeLimitError,
    create_failure_artifact,
    inspect_artifact,
    migrate_legacy_artifact,
    parse_size,
    sanitize_artifact,
)
from nablaguard.cli.main import main


def _tiny_issue() -> dict:
    return {
        "passed": False,
        "issues": [{"code": "NG3001", "message": "mismatch"}],
        "home_path": str(Path.home() / "secret-project" / "run.log"),
        "api_key": "should-not-leak",
        "note": f"user={os.environ.get('USERNAME') or os.environ.get('USER') or 'nobody'}",
    }


def test_parse_size_accepts_units() -> None:
    assert parse_size(128) == 128
    assert parse_size("1KB") == 1024
    assert parse_size("2MB") == 2 * 1024 * 1024
    with pytest.raises(ValueError, match="invalid artifact size"):
        parse_size("huge")


def test_policy_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="positive"):
        ArtifactPolicy.create(max_size=0)
    with pytest.raises(ValueError, match="negative"):
        ArtifactPolicy.create(max_stored_tensors=-1)
    with pytest.raises(ValueError, match="positive"):
        ArtifactPolicy.create(fingerprint_samples=0)


def test_create_private_by_default(tmp_path: Path) -> None:
    path = create_failure_artifact(
        tmp_path,
        issue=_tiny_issue(),
        inputs=[torch.randn(4)],
        minimized_inputs=[torch.randn(2)],
    )
    assert (path / "manifest.json").is_file()
    assert (path / "issue.json").is_file()
    assert (path / "fingerprints.json").is_file()
    assert (path / "environment.json").is_file()
    assert (path / "reproduction.py").is_file()
    assert not (path / "inputs" / "inputs.pt").exists()
    assert not (path / "inputs" / "minimized_inputs.pt").exists()
    assert not (path / "state").exists()

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "NGF"
    assert manifest["version"] == 1
    assert manifest["state"] == "complete"
    assert manifest["privacy"]["raw_tensors"] is False
    assert "raw input tensors disabled by policy" in manifest["omitted"]
    assert "raw minimized tensors disabled by policy" in manifest["omitted"]

    fingerprints = json.loads((path / "fingerprints.json").read_text(encoding="utf-8"))
    assert "input[0]" in fingerprints["fingerprints"]
    assert "minimized[0]" in fingerprints["fingerprints"]

    issue = json.loads((path / "issue.json").read_text(encoding="utf-8"))
    assert issue["result"]["api_key"] == "<REDACTED>"
    assert "<HOME>" in issue["result"]["home_path"]
    assert "should-not-leak" not in (path / "issue.json").read_text(encoding="utf-8")

    inspection = inspect_artifact(path)
    assert inspection.valid
    assert inspection.contains_raw_tensors is False


def test_create_with_raw_tensors(tmp_path: Path) -> None:
    path = create_failure_artifact(
        tmp_path,
        issue=_tiny_issue(),
        inputs=[torch.ones(3)],
        minimized_inputs=[torch.zeros(1)],
        policy=ArtifactPolicy.create(raw_tensors=True),
    )
    saved = torch.load(path / "inputs" / "inputs.pt", weights_only=True)
    minimized = torch.load(path / "inputs" / "minimized_inputs.pt", weights_only=True)
    assert saved[0].shape == (3,)
    assert minimized[0].shape == (1,)
    inspection = inspect_artifact(path)
    assert inspection.valid
    assert inspection.contains_raw_tensors is True
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["privacy"]["raw_tensors"] is True


def test_raw_tensors_false_when_none_written(tmp_path: Path) -> None:
    path = create_failure_artifact(
        tmp_path,
        issue=_tiny_issue(),
        inputs=[],
        policy=ArtifactPolicy.create(raw_tensors=True, max_stored_tensors=0),
    )
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["privacy"]["raw_tensors"] is False
    assert manifest["privacy"]["raw_tensors_requested"] is True
    assert inspect_artifact(path).contains_raw_tensors is False


def test_truncation_notes_cover_minimized(tmp_path: Path) -> None:
    path = create_failure_artifact(
        tmp_path,
        issue=_tiny_issue(),
        inputs=[torch.randn(2), torch.randn(2), torch.randn(2)],
        minimized_inputs=[torch.randn(1), torch.randn(1), torch.randn(1)],
        policy=ArtifactPolicy.create(raw_tensors=True, max_stored_tensors=1),
    )
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    assert "input tensors beyond max_stored_tensors" in manifest["omitted"]
    assert "minimized input tensors beyond max_stored_tensors" in manifest["omitted"]
    assert (path / "inputs" / "inputs.pt").is_file()
    assert (path / "inputs" / "minimized_inputs.pt").is_file()


def test_size_budget_rejects_large_tensors(tmp_path: Path) -> None:
    with pytest.raises(ArtifactSizeLimitError):
        create_failure_artifact(
            tmp_path,
            issue=_tiny_issue(),
            inputs=[torch.zeros(1024, 1024)],
            policy=ArtifactPolicy.create(raw_tensors=True, max_size="1KB"),
        )
    assert list(tmp_path.iterdir()) == []


def test_inspect_detects_hash_mismatch(tmp_path: Path) -> None:
    path = create_failure_artifact(tmp_path, issue=_tiny_issue(), inputs=[torch.randn(2)])
    issue_path = path / "issue.json"
    issue_path.write_text(issue_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    inspection = inspect_artifact(path)
    assert not inspection.valid
    assert any("hash mismatch" in error for error in inspection.errors)
    assert inspect_artifact(path, verify_hashes=False).valid is False  # size still mismatches


def test_inspect_rejects_symlinks(tmp_path: Path) -> None:
    path = create_failure_artifact(tmp_path, issue=_tiny_issue(), inputs=[torch.randn(2)])
    link = path / "evil"
    target = tmp_path / "outside.txt"
    target.write_text("secret", encoding="utf-8")
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable on this platform/user")
    inspection = inspect_artifact(path)
    assert not inspection.valid
    assert any("symbolic links are not allowed" in error for error in inspection.errors)


def test_inspect_legacy_includes_metadata_and_migrate_hint(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "metadata.json").write_text(
        json.dumps({"candidate": "torch.sin", "seed": 7}, indent=2) + "\n",
        encoding="utf-8",
    )
    (legacy / "inputs.pt").write_bytes(b"not-a-real-pickle")
    inspection = inspect_artifact(legacy)
    assert inspection.format == "LEGACY"
    assert not inspection.valid
    assert inspection.contains_raw_tensors is True
    assert inspection.manifest is not None
    assert inspection.manifest["metadata"]["candidate"] == "torch.sin"
    assert any("nabla artifact migrate" in error for error in inspection.errors)


def test_sanitize_preserves_fingerprints_without_raw_tensors(tmp_path: Path) -> None:
    source = create_failure_artifact(
        tmp_path,
        issue=_tiny_issue(),
        inputs=[torch.arange(4.0)],
        minimized_inputs=[torch.arange(2.0)],
        policy=ArtifactPolicy.create(raw_tensors=True),
        provenance={"subsystem": "test"},
    )
    original_fp = (source / "fingerprints.json").read_text(encoding="utf-8")
    destination = tmp_path / "sanitized"
    sanitized = sanitize_artifact(source, destination)
    assert sanitized != source
    assert not (sanitized / "inputs" / "inputs.pt").exists()
    assert not (sanitized / "inputs" / "minimized_inputs.pt").exists()
    new_fp = json.loads((sanitized / "fingerprints.json").read_text(encoding="utf-8"))
    old_fp = json.loads(original_fp)
    assert new_fp["fingerprints"] == old_fp["fingerprints"]
    assert "input[0]" in new_fp["fingerprints"]
    assert "minimized[0]" in new_fp["fingerprints"]
    assert inspect_artifact(sanitized).valid
    assert inspect_artifact(sanitized).contains_raw_tensors is False
    # Source is untouched.
    assert (source / "inputs" / "inputs.pt").is_file()


def test_migrate_legacy_does_not_torch_load(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "metadata.json").write_text(
        json.dumps({"seed": 3, "issues": []}, indent=2) + "\n",
        encoding="utf-8",
    )
    (legacy / "inputs.pt").write_bytes(b"pickle-bomb")
    with patch("torch.load", side_effect=AssertionError("must not load tensors")):
        migrated = migrate_legacy_artifact(legacy, tmp_path / "migrated")
    assert inspect_artifact(migrated).valid
    assert not (migrated / "inputs" / "inputs.pt").exists()
    issue = json.loads((migrated / "issue.json").read_text(encoding="utf-8"))
    assert issue["result"]["seed"] == 3


def test_string_redaction_uses_word_boundaries(tmp_path: Path) -> None:
    user = "alice"
    with (
        patch("nablaguard.core.redaction.getpass.getuser", return_value=user),
        patch("nablaguard.core.redaction.socket.gethostname", return_value="devbox"),
    ):
        path = create_failure_artifact(
            tmp_path,
            issue={
                "who": "alice ran the job",
                "tokenish": "malice",  # should not become m<USER>
                "host": "devbox is online",
                "hostish": "devboxen",  # boundary: en continues word? 'devboxen' has suffix
            },
            inputs=[],
        )
    issue = json.loads((path / "issue.json").read_text(encoding="utf-8"))["result"]
    assert issue["who"] == "<USER> ran the job"
    assert issue["tokenish"] == "malice"
    assert issue["host"] == "<HOST> is online"
    assert issue["hostish"] == "devboxen"


def test_operator_survives_artifact_write_failure(tmp_path: Path) -> None:
    import nablaguard as ng

    class BadSquare(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x**2

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            return grad_output * x

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    # Force size limit so write fails after comparisons complete.
    result = ng.check.operator(
        candidate=BadSquare.apply,
        reference=lambda x: x**2,
        inputs=[ng.tensor(shape=(4,), dtype=torch.float64)],
        artifact_dir=blocked,
        artifact_raw_tensors=True,
        artifact_max_size="1B",
    )
    assert not result.passed
    assert result.artifact_path is None
    assert result.artifact_error is not None
    assert "Artifact write failed" in result.format()
    assert result.issues  # failure evidence preserved


def test_fuzz_survives_artifact_write_failure(tmp_path: Path) -> None:
    import nablaguard as ng
    from nablaguard.check.specs import TensorSpec

    result = ng.check.fuzz(
        candidate=lambda x: x + 1,
        reference=lambda x: x,
        inputs=[TensorSpec((2,))],
        trials=1,
        minimize=False,
        artifact_dir=tmp_path,
        artifact_raw_tensors=True,
        artifact_max_size="1B",
    )
    assert not result.passed
    failure = result.failures[0]
    assert failure.artifact_path is None
    assert failure.artifact_error is not None
    assert "Artifact write failed" in result.format()


def test_cli_inspect_and_sanitize_roundtrip(tmp_path: Path, capsys) -> None:
    path = create_failure_artifact(
        tmp_path,
        issue=_tiny_issue(),
        inputs=[torch.randn(3)],
        policy=ArtifactPolicy.create(raw_tensors=True),
    )
    assert main(["inspect", str(path)]) == 0
    assert main(["artifact", "inspect", str(path), "--no-verify-hashes"]) == 0
    out = capsys.readouterr().out
    assert '"format": "NGF"' in out

    exit_code = main(
        ["artifact", "sanitize", str(path), "--output-root", str(tmp_path / "out")]
    )
    assert exit_code == 0
    sanitized_path = Path(capsys.readouterr().out.strip())
    assert inspect_artifact(sanitized_path).valid


def test_cli_inspect_legacy_and_migrate(tmp_path: Path, capsys) -> None:
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "metadata.json").write_text(
        json.dumps({"seed": 9}, indent=2) + "\n", encoding="utf-8"
    )
    assert main(["inspect", str(legacy)]) == 1
    output = capsys.readouterr().out
    assert "LEGACY" in output
    assert "migrate" in output

    assert (
        main(
            [
                "artifact",
                "migrate",
                str(legacy),
                "--output-root",
                str(tmp_path / "migrated"),
            ]
        )
        == 0
    )
    migrated = Path(capsys.readouterr().out.strip())
    assert main(["inspect", str(migrated)]) == 0


def test_cli_invalid_artifact_max_size(capsys) -> None:
    exit_code = main(
        [
            "check",
            "torch:sin",
            "--reference",
            "torch:sin",
            "--shape",
            "2",
            "--artifact-max-size",
            "nope",
        ]
    )
    assert exit_code == 3
    assert "Invalid check configuration" in capsys.readouterr().err


def test_inspect_rejects_windows_drive_inventory_paths(tmp_path: Path) -> None:
    path = create_failure_artifact(tmp_path, issue=_tiny_issue(), inputs=[])
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = [
        {"path": "C:/Windows/win.ini", "size_bytes": 1, "sha256": "00"},
        {"path": r"\\server\share\secret", "size_bytes": 1, "sha256": "00"},
    ]
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    inspection = inspect_artifact(path, verify_hashes=False)
    assert not inspection.valid
    joined = " ".join(inspection.errors)
    assert "unsafe file inventory path" in joined
    assert "win.ini" not in (path / "issue.json").read_text(encoding="utf-8")


def test_inspect_does_not_hash_oversized_listed_files(tmp_path: Path) -> None:
    from nablaguard.artifact import ngf

    path = create_failure_artifact(tmp_path, issue=_tiny_issue(), inputs=[])
    with patch.object(ngf, "_MAX_INSPECT_FILE_BYTES", 32):
        inspection = inspect_artifact(path)
    assert not inspection.valid
    assert any("inspection size cap" in error for error in inspection.errors)


def test_inspect_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    inspection = inspect_artifact(missing)
    assert not inspection.valid
    assert inspection.format == "UNKNOWN"
    assert any("does not exist" in error for error in inspection.errors)


def test_inspect_rejects_invalid_manifest_shapes(tmp_path: Path) -> None:
    path = create_failure_artifact(tmp_path, issue=_tiny_issue(), inputs=[torch.ones(2)])
    # Wrong format/version/state and non-list inventory.
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "format": "NOT_NGF",
                "version": 99,
                "state": "partial",
                "files": "nope",
                "privacy": {"raw_tensors": False},
            }
        ),
        encoding="utf-8",
    )
    inspection = inspect_artifact(path, verify_hashes=False)
    assert not inspection.valid
    joined = " ".join(inspection.errors)
    assert "format is not NGF" in joined
    assert "unsupported NGF version" in joined
    assert "not marked complete" in joined
    assert "files must be a list" in joined

    # Manifest that is not an object.
    (path / "manifest.json").write_text("[]\n", encoding="utf-8")
    inspection = inspect_artifact(path)
    assert not inspection.valid
    assert any("manifest must be an object" in error for error in inspection.errors)

    # Unlisted extra file.
    path = create_failure_artifact(tmp_path / "extra", issue=_tiny_issue(), inputs=[])
    (path / "extra.txt").write_text("noise", encoding="utf-8")
    inspection = inspect_artifact(path)
    assert not inspection.valid
    assert any("unlisted file present" in error for error in inspection.errors)

    # Unsafe inventory path and missing listed file.
    path = create_failure_artifact(tmp_path / "unsafe", issue=_tiny_issue(), inputs=[])
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"] = [
        {"path": "../escape.txt", "size_bytes": 1, "sha256": "00"},
        {"path": "missing.json", "size_bytes": 1, "sha256": "00"},
        {"path": 12, "size_bytes": 1, "sha256": "00"},
    ]
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    inspection = inspect_artifact(path, verify_hashes=False)
    assert not inspection.valid
    joined = " ".join(inspection.errors)
    assert "unsafe file inventory path" in joined
    assert "listed file is missing" in joined


def test_create_rejects_unsupported_metadata_types(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="unsupported value type"):
        create_failure_artifact(
            tmp_path,
            issue={"callback": lambda: None},  # type: ignore[dict-item]
            inputs=[],
        )


def test_light_mode_statistics_are_sampled() -> None:
    from nablaguard.sanitize.statistics import compute_statistics

    value = torch.arange(10_000, dtype=torch.float32)
    stats = compute_statistics(value, max_samples=128)
    assert stats.total_elements == 10_000
    assert 1 <= stats.sampled_elements <= 128
    full = compute_statistics(value)
    assert full.sampled_elements == 10_000
