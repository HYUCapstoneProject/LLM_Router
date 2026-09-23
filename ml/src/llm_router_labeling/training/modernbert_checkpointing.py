from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..shared.dataset_io import sha256_file, write_json


CHECKPOINT_SCHEMA_VERSION = "modernbert-resume-checkpoint-v1"
CHECKPOINT_MANIFEST_NAME = "checkpoint_manifest.json"
ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"
SUCCESS_MARKER_NAME = "_SUCCESS"


class ModernBertCheckpointError(RuntimeError):
    """Raised when a resumable checkpoint is incomplete or incompatible."""


@dataclass(frozen=True)
class CheckpointOptions:
    checkpoint_dir: Path
    save_every_optimizer_steps: int
    keep_last_checkpoints: int = 2
    resume_from: Path | None = None
    on_checkpoint_saved: Callable[[Path], None] | None = None

    def validate(self) -> None:
        if self.save_every_optimizer_steps <= 0:
            raise ModernBertCheckpointError(
                "save_every_optimizer_steps must be positive"
            )
        if self.keep_last_checkpoints <= 0:
            raise ModernBertCheckpointError("keep_last_checkpoints must be positive")


def training_fingerprint(
    *,
    training_data_sha256: str,
    config: Mapping[str, Any],
    maximum_steps: int | None,
    runtime_version: str,
) -> str:
    payload = {
        "training_data_sha256": training_data_sha256,
        "config": dict(config),
        "maximum_steps": maximum_steps,
        "runtime_version": runtime_version,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_name(optimizer_steps: int) -> str:
    if optimizer_steps < 0:
        raise ModernBertCheckpointError("optimizer_steps must be non-negative")
    return f"step-{optimizer_steps:08d}"


def next_training_cursor(
    *, epoch_index: int, batch_index: int, batches_in_epoch: int
) -> tuple[int, int]:
    next_batch_index = batch_index + 1
    if next_batch_index >= batches_in_epoch:
        return epoch_index + 1, 0
    return epoch_index, next_batch_index


def _artifact_files(directory: Path, manifest_name: str) -> list[Path]:
    ignored = {manifest_name, SUCCESS_MARKER_NAME}
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name not in ignored
    )


def _manifest_payload(
    directory: Path, *, manifest_name: str, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    files = {
        path.relative_to(directory).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _artifact_files(directory, manifest_name)
    }
    return {"metadata": dict(metadata), "files": files}


def seal_artifact_directory(
    directory: Path,
    *,
    manifest_name: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not directory.is_dir():
        raise ModernBertCheckpointError(f"Artifact directory does not exist: {directory}")
    manifest = _manifest_payload(
        directory, manifest_name=manifest_name, metadata=metadata
    )
    write_json(directory / manifest_name, manifest)
    (directory / SUCCESS_MARKER_NAME).write_text("complete\n", encoding="utf-8")
    return manifest


def verify_artifact_directory(
    directory: Path,
    *,
    manifest_name: str,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    manifest_path = directory / manifest_name
    marker_path = directory / SUCCESS_MARKER_NAME
    if not directory.is_dir() or not manifest_path.is_file() or not marker_path.is_file():
        raise ModernBertCheckpointError(f"Incomplete artifact directory: {directory}")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ModernBertCheckpointError(
            f"Cannot read artifact manifest: {manifest_path}"
        ) from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ModernBertCheckpointError(f"Invalid artifact manifest: {manifest_path}")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ModernBertCheckpointError(f"Invalid artifact metadata: {manifest_path}")
    if (
        expected_fingerprint is not None
        and metadata.get("training_fingerprint") != expected_fingerprint
    ):
        raise ModernBertCheckpointError(
            "Checkpoint training fingerprint does not match the current data/config"
        )
    for relative_path, expected in manifest["files"].items():
        path = directory / str(relative_path)
        if not path.is_file():
            raise ModernBertCheckpointError(f"Artifact file is missing: {path}")
        if path.stat().st_size != int(expected["size_bytes"]):
            raise ModernBertCheckpointError(f"Artifact size mismatch: {path}")
        if sha256_file(path) != str(expected["sha256"]):
            raise ModernBertCheckpointError(f"Artifact SHA-256 mismatch: {path}")
    return manifest


def verify_checkpoint(
    directory: Path, *, expected_fingerprint: str | None = None
) -> dict[str, Any]:
    manifest = verify_artifact_directory(
        directory,
        manifest_name=CHECKPOINT_MANIFEST_NAME,
        expected_fingerprint=expected_fingerprint,
    )
    metadata = manifest["metadata"]
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ModernBertCheckpointError(
            f"Unsupported checkpoint schema: {metadata.get('schema_version')!r}"
        )
    if not (directory / "training_state.pt").is_file():
        raise ModernBertCheckpointError(
            f"Checkpoint has no training_state.pt: {directory}"
        )
    return manifest


def create_checkpoint(
    checkpoint_dir: Path,
    *,
    optimizer_steps: int,
    metadata: Mapping[str, Any],
    write_payload: Callable[[Path], None],
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_dir / checkpoint_name(optimizer_steps)
    if destination.exists():
        verify_checkpoint(
            destination,
            expected_fingerprint=str(metadata["training_fingerprint"]),
        )
        return destination
    temporary = checkpoint_dir / f".{destination.name}.partial-{uuid.uuid4().hex}"
    try:
        temporary.mkdir()
        write_payload(temporary)
        seal_artifact_directory(
            temporary,
            manifest_name=CHECKPOINT_MANIFEST_NAME,
            metadata={"schema_version": CHECKPOINT_SCHEMA_VERSION, **dict(metadata)},
        )
        verify_checkpoint(
            temporary,
            expected_fingerprint=str(metadata["training_fingerprint"]),
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def completed_checkpoints(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.is_dir():
        return []
    checkpoints: list[tuple[int, Path]] = []
    for candidate in checkpoint_dir.glob("step-*"):
        try:
            manifest = verify_checkpoint(candidate)
            optimizer_steps = int(manifest["metadata"]["optimizer_steps"])
        except (KeyError, TypeError, ValueError, ModernBertCheckpointError):
            continue
        checkpoints.append((optimizer_steps, candidate))
    return [path for _, path in sorted(checkpoints)]


def latest_completed_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = completed_checkpoints(checkpoint_dir)
    return checkpoints[-1] if checkpoints else None


def prune_checkpoints(checkpoint_dir: Path, *, keep_last: int) -> None:
    if keep_last <= 0:
        raise ModernBertCheckpointError("keep_last must be positive")
    checkpoints = completed_checkpoints(checkpoint_dir)
    for checkpoint in checkpoints[:-keep_last]:
        shutil.rmtree(checkpoint)


def atomic_copy_artifact(
    source: Path,
    destination: Path,
    *,
    manifest_name: str,
    expected_fingerprint: str | None = None,
) -> Path:
    verify_artifact_directory(
        source,
        manifest_name=manifest_name,
        expected_fingerprint=expected_fingerprint,
    )
    if destination.exists():
        verify_artifact_directory(
            destination,
            manifest_name=manifest_name,
            expected_fingerprint=expected_fingerprint,
        )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, temporary)
        verify_artifact_directory(
            temporary,
            manifest_name=manifest_name,
            expected_fingerprint=expected_fingerprint,
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def atomic_copy_checkpoint(
    source: Path, destination: Path, *, expected_fingerprint: str | None = None
) -> Path:
    return atomic_copy_artifact(
        source,
        destination,
        manifest_name=CHECKPOINT_MANIFEST_NAME,
        expected_fingerprint=expected_fingerprint,
    )
