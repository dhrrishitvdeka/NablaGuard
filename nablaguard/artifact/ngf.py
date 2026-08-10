"""NGF v1 writer, validator, sanitizer, and legacy migration."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import torch

from nablaguard.capture.fingerprints import fingerprint_mapping
from nablaguard.core.redaction import normalize_json, redact_value

_FORMAT = "NGF"
_VERSION = 1
_SIZE_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB)?\s*$", re.I)
# Conservative allowance for pickle/torch.save framing plus JSON/fingerprint payload.
_TENSOR_SERIALIZATION_FACTOR = 1.25
_METADATA_RESERVE_BYTES = 256 * 1024
_MAX_INSPECT_FILE_BYTES = 64 * 1024 * 1024
_MAX_INSPECT_TOTAL_BYTES = 512 * 1024 * 1024


class ArtifactSizeLimitError(ValueError):
    """Raised before an NGF artifact can exceed its configured size budget."""


@dataclass(frozen=True, slots=True)
class ArtifactPolicy:
    """Explicit NGF privacy and retention controls."""

    raw_tensors: bool = False
    max_size_bytes: int = 500 * 1024 * 1024
    max_stored_tensors: int = 16
    fingerprint_samples: int = 4096

    @classmethod
    def create(
        cls,
        *,
        raw_tensors: bool = False,
        max_size: str | int = "500MB",
        max_stored_tensors: int = 16,
        fingerprint_samples: int = 4096,
    ) -> ArtifactPolicy:
        max_size_bytes = parse_size(max_size)
        if max_size_bytes <= 0:
            raise ValueError("artifact maximum size must be positive")
        if max_stored_tensors < 0:
            raise ValueError("max_stored_tensors cannot be negative")
        if fingerprint_samples <= 0:
            raise ValueError("fingerprint_samples must be positive")
        return cls(
            raw_tensors=raw_tensors,
            max_size_bytes=max_size_bytes,
            max_stored_tensors=max_stored_tensors,
            fingerprint_samples=fingerprint_samples,
        )


@dataclass(frozen=True, slots=True)
class NGFInspection:
    """Safe JSON-only artifact validation result."""

    path: Path
    format: str
    version: int | None
    valid: bool
    errors: tuple[str, ...]
    total_size_bytes: int
    contains_raw_tensors: bool
    manifest: Mapping[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "format": self.format,
            "version": self.version,
            "valid": self.valid,
            "errors": list(self.errors),
            "total_size_bytes": self.total_size_bytes,
            "contains_raw_tensors": self.contains_raw_tensors,
            "manifest": self.manifest,
        }


def create_failure_artifact(
    root: str | Path,
    *,
    issue: Mapping[str, Any],
    inputs: Sequence[torch.Tensor] = (),
    minimized_inputs: Sequence[torch.Tensor] | None = None,
    trace: Mapping[str, Any] | None = None,
    policy: ArtifactPolicy | None = None,
    provenance: Mapping[str, Any] | None = None,
    fingerprints: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically create one complete NGF v1 directory."""

    selected_policy = policy or ArtifactPolicy()
    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    identifier = f"NGF-{uuid4().hex[:8].upper()}"
    destination = root_path / identifier
    temporary = Path(tempfile.mkdtemp(prefix=f".{identifier}-", dir=root_path))
    omitted: list[str] = []
    wrote_raw_tensors = False
    try:
        inputs_dir = temporary / "inputs"
        inputs_dir.mkdir()
        safe_issue = redact_value(dict(issue))
        _write_json(
            temporary / "issue.json",
            {"schema": {"name": "nablaguard.issue", "version": 1}, "result": safe_issue},
        )
        _write_json(
            temporary / "trace.json",
            {
                "schema": {"name": "nablaguard.trace", "version": 1},
                "trace": redact_value(dict(trace or {})),
            },
        )
        fingerprint_document = _resolve_fingerprints(
            fingerprints,
            inputs=inputs,
            minimized_inputs=minimized_inputs,
            max_samples=selected_policy.fingerprint_samples,
        )
        _write_json(
            temporary / "fingerprints.json",
            {
                "schema": {"name": "nablaguard.fingerprints", "version": 1},
                "fingerprints": normalize_json(fingerprint_document),
            },
        )
        if environment is None:
            environment_document = _safe_environment()
        else:
            environment_document = redact_value(dict(environment))
            if "schema" not in environment_document:
                environment_document = {
                    "schema": {"name": "nablaguard.environment", "version": 1},
                    **environment_document,
                }
        _write_json(temporary / "environment.json", environment_document)
        (temporary / "reproduction.py").write_text(_REPRODUCTION, encoding="utf-8")

        minimized = tuple(minimized_inputs or ())
        if selected_policy.raw_tensors:
            selected_inputs = tuple(inputs[: selected_policy.max_stored_tensors])
            selected_minimized = tuple(minimized[: selected_policy.max_stored_tensors])
            if len(selected_inputs) < len(inputs):
                omitted.append("input tensors beyond max_stored_tensors")
            if len(selected_minimized) < len(minimized):
                omitted.append("minimized input tensors beyond max_stored_tensors")
            _check_tensor_budget(selected_inputs, selected_minimized, selected_policy)
            if selected_inputs:
                torch.save(
                    [value.detach().to(device="cpu") for value in selected_inputs],
                    inputs_dir / "inputs.pt",
                )
                wrote_raw_tensors = True
            if selected_minimized:
                torch.save(
                    [value.detach().to(device="cpu") for value in selected_minimized],
                    inputs_dir / "minimized_inputs.pt",
                )
                wrote_raw_tensors = True
        else:
            if inputs:
                omitted.append("raw input tensors disabled by policy")
            if minimized:
                omitted.append("raw minimized tensors disabled by policy")

        files = _file_inventory(temporary)
        manifest = {
            "format": _FORMAT,
            "version": _VERSION,
            "id": identifier,
            "state": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "privacy": {
                "raw_tensors": wrote_raw_tensors,
                "raw_tensors_requested": selected_policy.raw_tensors,
                "redaction_applied": True,
            },
            "policy": {
                "max_size_bytes": selected_policy.max_size_bytes,
                "max_stored_tensors": selected_policy.max_stored_tensors,
                "fingerprint_samples": selected_policy.fingerprint_samples,
            },
            "omitted": omitted,
            "provenance": redact_value(dict(provenance or {})),
            "files": files,
        }
        _write_json(temporary / "manifest.json", manifest)
        total_size = _directory_size(temporary)
        if total_size > selected_policy.max_size_bytes:
            raise ArtifactSizeLimitError(
                f"NGF artifact would use {total_size} bytes, above "
                f"{selected_policy.max_size_bytes}"
            )
        os.replace(temporary, destination)
        return destination
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def inspect_artifact(path: str | Path, *, verify_hashes: bool = True) -> NGFInspection:
    """Inspect JSON and file hashes without deserializing tensors or executing code."""

    artifact_path = Path(path).resolve()
    if not artifact_path.exists():
        return NGFInspection(
            artifact_path,
            "UNKNOWN",
            None,
            False,
            ("artifact path does not exist",),
            0,
            False,
            None,
        )

    total_size, walk_errors, on_disk_files, contains_raw = _safe_walk(artifact_path)
    manifest_path = artifact_path / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        if (artifact_path / "metadata.json").is_file() and not (
            artifact_path / "metadata.json"
        ).is_symlink():
            return _inspect_legacy(artifact_path, total_size, contains_raw, walk_errors)
        return NGFInspection(
            artifact_path,
            "UNKNOWN",
            None,
            False,
            ("manifest.json is missing", *walk_errors),
            total_size,
            contains_raw,
            None,
        )

    errors: list[str] = list(walk_errors)
    try:
        if manifest_path.stat().st_size > 1024 * 1024:
            raise ValueError("manifest exceeds 1MB inspection limit")
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return NGFInspection(
            artifact_path,
            "UNKNOWN",
            None,
            False,
            (f"invalid manifest: {error}", *errors),
            total_size,
            contains_raw,
            None,
        )
    if not isinstance(raw_manifest, dict):
        errors.append("manifest must be an object")
        manifest: dict[str, Any] = {}
    else:
        manifest = raw_manifest
    if manifest.get("format") != _FORMAT:
        errors.append("manifest format is not NGF")
    if manifest.get("version") != _VERSION:
        errors.append(f"unsupported NGF version: {manifest.get('version')!r}")
    if manifest.get("state") != "complete":
        errors.append("artifact is not marked complete")
    inventory = manifest.get("files")
    listed: set[str] = set()
    if not isinstance(inventory, list):
        errors.append("manifest files must be a list")
    else:
        for entry in inventory:
            if not isinstance(entry, dict):
                errors.append("file inventory entry must be an object")
                continue
            relative = entry.get("path")
            if not isinstance(relative, str) or not _safe_relative_path(relative):
                errors.append(f"unsafe file inventory path: {relative!r}")
                continue
            listed.add(relative)
            candidate = artifact_path / Path(PurePosixPath(relative))
            if candidate.is_symlink():
                errors.append(f"symbolic links are not allowed: {relative}")
                continue
            if not candidate.is_file():
                errors.append(f"listed file is missing: {relative}")
                continue
            try:
                size = candidate.stat().st_size
            except OSError as error:
                errors.append(f"cannot stat listed file {relative}: {error}")
                continue
            if size != entry.get("size_bytes"):
                errors.append(f"file size mismatch: {relative}")
            if verify_hashes and _sha256(candidate) != entry.get("sha256"):
                errors.append(f"file hash mismatch: {relative}")
    for extra in sorted(on_disk_files - listed):
        errors.append(f"unlisted file present: {extra}")
    return NGFInspection(
        artifact_path,
        str(manifest.get("format", "UNKNOWN")),
        manifest.get("version") if isinstance(manifest.get("version"), int) else None,
        not errors,
        tuple(errors),
        total_size,
        contains_raw,
        manifest,
    )


def sanitize_artifact(path: str | Path, destination_root: str | Path | None = None) -> Path:
    """Create a new metadata-only NGF artifact; never mutate the source."""

    inspection = inspect_artifact(path)
    if not inspection.valid:
        raise ValueError(f"cannot sanitize invalid artifact: {inspection.errors}")
    source = inspection.path
    issue_document = _read_json(source / "issue.json")
    trace_document = _read_json(source / "trace.json")
    fingerprint_document = _read_json(source / "fingerprints.json")
    environment_document = _read_json(source / "environment.json")
    root = Path(destination_root).resolve() if destination_root is not None else source.parent
    fingerprints = fingerprint_document.get("fingerprints", {})
    if not isinstance(fingerprints, Mapping):
        raise ValueError("source fingerprints.json is not a mapping document")
    return create_failure_artifact(
        root,
        issue=redact_value(issue_document.get("result", {})),
        trace=redact_value(trace_document.get("trace", {})),
        fingerprints=dict(fingerprints),
        environment=environment_document,
        policy=ArtifactPolicy(raw_tensors=False),
        provenance={"sanitized_from": source.name},
    )


def migrate_legacy_artifact(path: str | Path, destination_root: str | Path) -> Path:
    """Migrate legacy JSON metadata without loading its pickle tensor files."""

    source = Path(path).resolve()
    metadata_path = source / "metadata.json"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise ValueError(f"not a legacy NablaGuard artifact: {source}")
    metadata = _read_json(metadata_path)
    return create_failure_artifact(
        destination_root,
        issue=metadata,
        policy=ArtifactPolicy(raw_tensors=False),
        provenance={"migrated_from": source.name, "legacy_version": 0},
    )


def parse_size(value: str | int) -> int:
    if isinstance(value, int):
        return value
    match = _SIZE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid artifact size: {value!r}")
    amount = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    multiplier = {
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
    }[unit]
    return int(amount * multiplier)


def _resolve_fingerprints(
    provided: Mapping[str, Any] | None,
    *,
    inputs: Sequence[torch.Tensor],
    minimized_inputs: Sequence[torch.Tensor] | None,
    max_samples: int,
) -> dict[str, Any]:
    if provided is not None:
        return dict(provided)
    named: dict[str, torch.Tensor] = {
        f"input[{index}]": value for index, value in enumerate(inputs)
    }
    if minimized_inputs is not None:
        for index, value in enumerate(minimized_inputs):
            named[f"minimized[{index}]"] = value
    return fingerprint_mapping(named, max_samples=max_samples)


def _inspect_legacy(
    artifact_path: Path,
    total_size: int,
    contains_raw: bool,
    walk_errors: list[str],
) -> NGFInspection:
    legacy_summary: dict[str, Any] | None = None
    errors = [
        (
            "legacy artifact requires migration to NGF v1; "
            f"run: nabla artifact migrate {artifact_path} --output-root <dir>"
        ),
        *walk_errors,
    ]
    metadata_path = artifact_path / "metadata.json"
    try:
        if metadata_path.stat().st_size > 1024 * 1024:
            errors.append("legacy metadata.json exceeds 1MB inspection limit")
        else:
            metadata = _read_json(metadata_path)
            legacy_summary = {
                "legacy": True,
                "migrate": (
                    f"nabla artifact migrate {artifact_path} --output-root <dir>"
                ),
                "metadata": redact_value(metadata),
            }
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as error:
        errors.append(f"legacy metadata unreadable: {error}")
    return NGFInspection(
        artifact_path,
        "LEGACY",
        0,
        False,
        tuple(errors),
        total_size,
        contains_raw or (artifact_path / "inputs.pt").is_file(),
        legacy_summary,
    )


def _check_tensor_budget(
    inputs: Sequence[torch.Tensor],
    minimized_inputs: Sequence[torch.Tensor],
    policy: ArtifactPolicy,
) -> None:
    raw_bytes = sum(
        value.numel() * value.element_size() for value in (*inputs, *minimized_inputs)
    )
    estimated = int(raw_bytes * _TENSOR_SERIALIZATION_FACTOR) + _METADATA_RESERVE_BYTES
    if estimated > policy.max_size_bytes:
        raise ArtifactSizeLimitError(
            f"raw tensor payload requires at least {estimated} bytes "
            f"(including serialization overhead reserve), above {policy.max_size_bytes}"
        )


def _file_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(_iter_files(root), key=lambda item: item.as_posix())
        if path.name != "manifest.json"
    ]


def _safe_walk(
    root: Path,
) -> tuple[int, list[str], set[str], bool]:
    """Walk an untrusted artifact without following symlinks."""

    errors: list[str] = []
    files: set[str] = set()
    total = 0
    contains_raw = False
    for path, is_symlink, is_dir, is_file in _walk_entries(root):
        relative = path.relative_to(root).as_posix()
        if is_symlink:
            errors.append(f"symbolic links are not allowed: {relative}")
            continue
        if is_dir:
            continue
        if not is_file:
            continue
        try:
            size = path.stat().st_size
        except OSError as error:
            errors.append(f"cannot stat {relative}: {error}")
            continue
        if size > _MAX_INSPECT_FILE_BYTES:
            errors.append(
                f"file exceeds inspection size cap ({_MAX_INSPECT_FILE_BYTES} bytes): {relative}"
            )
            continue
        total += size
        if total > _MAX_INSPECT_TOTAL_BYTES:
            errors.append(
                f"artifact exceeds inspection size cap ({_MAX_INSPECT_TOTAL_BYTES} bytes)"
            )
            break
        if path.name != "manifest.json":
            files.add(relative)
        if path.suffix == ".pt":
            contains_raw = True
    return total, errors, files, contains_raw


def _walk_entries(root: Path) -> Iterator[tuple[Path, bool, bool, bool]]:
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    try:
                        is_symlink = entry.is_symlink()
                        is_dir = entry.is_dir(follow_symlinks=False)
                        is_file = entry.is_file(follow_symlinks=False)
                    except OSError:
                        continue
                    yield path, is_symlink, is_dir, is_file
                    if is_dir and not is_symlink:
                        stack.append(path)
        except OSError:
            continue


def _iter_files(root: Path) -> Iterator[Path]:
    for path, is_symlink, _is_dir, is_file in _walk_entries(root):
        if is_file and not is_symlink:
            yield path


def _safe_environment() -> dict[str, Any]:
    return {
        "schema": {"name": "nablaguard.environment", "version": 1},
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


def _write_json(path: Path, value: Any) -> None:
    normalized = normalize_json(value)
    encoded = json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(encoded, encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and value != ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for file_path in _iter_files(path):
        try:
            total += file_path.stat().st_size
        except OSError:
            continue
    return total


_REPRODUCTION = '''"""Safely inspect an NGF artifact without loading tensor pickle data."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
issue = json.loads((HERE / "issue.json").read_text(encoding="utf-8"))
print(json.dumps({"manifest": manifest, "issue": issue}, indent=2))
print("Use `nabla artifact inspect`, then replay only artifacts from trusted sources.")
'''
