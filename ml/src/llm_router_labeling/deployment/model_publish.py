"""Publish gate-passed model packages and expose the current serving version.

The delivery path is publish-then-pull, not push. The pipeline writes an
immutable package to ``<version>/candidate/``, promotes it to
``<version>/approved/`` only after the deploy gate passes, and atomically moves
``CURRENT.json``. FastAPI polls that pointer, verifies the complete package next
to the active model, and swaps it without a process restart.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ..shared.dataset_io import sha256_file, write_json
from ..shared.object_store import ObjectStore, ObjectStoreError, parse_gcs_uri
from .model_package import (
    EXPECTED_ID2LABEL,
    ModelPackageError,
    validate_model_package,
)

CURRENT_SCHEMA_VERSION = "router-current-model-v1"
CURRENT_POINTER_FILENAME = "CURRENT.json"
MANIFEST_FILENAME = "model_manifest.json"


class ModelPublishError(RuntimeError):
    """Raised when publishing or approving violates the delivery contract."""


def candidate_uri(root_uri: str, model_version: str) -> str:
    return str(parse_gcs_uri(root_uri).child(model_version, "candidate"))


def approved_uri(root_uri: str, model_version: str) -> str:
    return str(parse_gcs_uri(root_uri).child(model_version, "approved"))


def current_pointer_uri(root_uri: str) -> str:
    return str(parse_gcs_uri(root_uri).child(CURRENT_POINTER_FILENAME))


def _package_files(package_dir: Path) -> list[Path]:
    return sorted(path for path in package_dir.iterdir() if path.is_file())


def publish_candidate(
    package_dir: Path,
    *,
    root_uri: str,
    model_version: str,
    object_store: ObjectStore,
    gate_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Upload a validated candidate package to ``<root>/<version>/candidate/``.

    The package is validated locally before a single byte is uploaded, so a
    malformed package can never occupy a candidate path.
    """
    package_dir = Path(package_dir).resolve()
    try:
        validation = validate_model_package(package_dir)
    except ModelPackageError as exc:
        raise ModelPublishError(f"Candidate package is not valid: {exc}") from exc

    if validation["model_version"] != model_version:
        raise ModelPublishError(
            "Package model_version does not match the requested candidate version: "
            f"{validation['model_version']!r} != {model_version!r}"
        )
    if gate_result is not None and not gate_result.get("passed"):
        raise ModelPublishError(
            "Refusing to publish a candidate that failed the deploy gate"
        )

    destination = candidate_uri(root_uri, model_version)
    uploaded = _upload_package(package_dir, destination, object_store)
    return {
        "model_version": model_version,
        "package_uri": destination,
        "uploaded_files": uploaded,
        "gate_status": validation["gate_status"],
    }


def _upload_package(
    package_dir: Path, destination_uri: str, object_store: ObjectStore
) -> list[str]:
    base = parse_gcs_uri(destination_uri)
    uploaded: list[str] = []
    for path in _package_files(package_dir):
        target = str(base.child(path.name))
        object_store.upload_file(path, target)
        uploaded.append(path.name)
    return uploaded


def approve_candidate(
    *,
    root_uri: str,
    model_version: str,
    object_store: ObjectStore,
    approved_by: str,
    pipeline_run_id: str | None = None,
    work_dir: Path,
    allow_pointer_rollback: bool = False,
    logical_model_version: str | None = None,
    training_mode: str | None = None,
    retention_anchor_model_version: str | None = None,
) -> dict[str, Any]:
    """Copy a candidate to ``approved/`` and move the ``CURRENT.json`` pointer.

    Approval is the only step that rewrites ``gate_status``. The package is
    downloaded, re-verified against its own manifest, stamped as approved, and
    re-uploaded, so an approved package is always self-consistent.
    """
    source = candidate_uri(root_uri, model_version)
    destination = approved_uri(root_uri, model_version)
    previous = read_current_pointer(root_uri, object_store=object_store)
    previous_version = previous.get("model_version") if previous else None
    if previous_version == model_version:
        raise ModelPublishError(
            f"{model_version} is already the current approved model"
        )
    staging = Path(work_dir) / model_version
    staging.mkdir(parents=True, exist_ok=True)

    downloaded = download_package(source, staging, object_store=object_store)
    if not downloaded:
        raise ModelPublishError(f"No candidate package found at {source}")

    manifest_path = staging / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ModelPublishError(
            f"Candidate package has no {MANIFEST_FILENAME}: {source}"
        )
    try:
        validate_model_package(staging, manifest_path)
    except ModelPackageError as exc:
        raise ModelPublishError(
            f"Candidate package failed re-validation: {exc}"
        ) from exc

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_version") != model_version:
        raise ModelPublishError(
            f"Candidate manifest model_version is {manifest.get('model_version')!r}, "
            f"expected {model_version!r}"
        )
    approved_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    manifest["gate_status"] = "approved"
    manifest["approved_at"] = approved_at
    manifest["approved_by"] = approved_by
    if pipeline_run_id:
        manifest["pipeline_run_id"] = pipeline_run_id
    if logical_model_version:
        manifest["logical_model_version"] = logical_model_version
    if training_mode:
        manifest["training_mode"] = training_mode
    write_json(manifest_path, manifest)
    # The manifest lists every other file's checksum, so rewriting it does not
    # invalidate the package. Re-validate to prove that.
    validate_model_package(staging, manifest_path)

    if previous and not allow_pointer_rollback:
        _reject_pointer_rollback(previous, manifest)

    _upload_package(staging, destination, object_store)

    pointer_uri = current_pointer_uri(root_uri)
    pointer = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "model_version": model_version,
        "package_uri": destination,
        "previous_model_version": previous_version,
        "previous_package_uri": previous.get("package_uri") if previous else None,
        "approved_at": approved_at,
        "approved_by": approved_by,
        "pipeline_run_id": pipeline_run_id,
        "id2label": EXPECTED_ID2LABEL,
        "manifest_sha256": sha256_file(manifest_path),
    }
    if logical_model_version:
        pointer["logical_model_version"] = logical_model_version
    if training_mode:
        pointer["training_mode"] = training_mode
    protected_versions = {
        str(value)
        for value in (model_version, previous_version, retention_anchor_model_version)
        if value
    }
    pointer["retained_model_versions"] = sorted(protected_versions)
    object_store.write_text(
        pointer_uri,
        json.dumps(pointer, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        if_generation_match=object_store.generation(pointer_uri),
    )
    pointer["retention_warnings"] = _prune_model_packages(
        root_uri=root_uri,
        current_model_version=model_version,
        protected_versions=protected_versions,
        object_store=object_store,
    )
    return pointer


def _prune_model_packages(
    *,
    root_uri: str,
    current_model_version: str,
    protected_versions: set[str],
    object_store: ObjectStore,
) -> list[str]:
    """Remove duplicate candidates and packages outside the rollback boundary.

    CURRENT has already moved before this best-effort cleanup starts. Cleanup
    failures are therefore reported as warnings rather than turning a promoted
    model into a failed pipeline run.
    """
    root = parse_gcs_uri(root_uri)
    root_prefix = root.prefix.rstrip("/") + "/" if root.prefix else ""
    warnings: list[str] = []
    try:
        uris = object_store.list_uris(root_uri)
    except ObjectStoreError as exc:
        return [str(exc)]
    for uri in uris:
        parsed = parse_gcs_uri(uri)
        relative = (
            parsed.prefix[len(root_prefix) :]
            if parsed.prefix.startswith(root_prefix)
            else ""
        )
        parts = PurePosixPath(relative).parts
        if len(parts) < 3:
            continue
        version, package_state = parts[0], parts[1]
        remove_current_candidate = (
            version == current_model_version and package_state == "candidate"
        )
        remove_retired_package = (
            version not in protected_versions
            and package_state in {"candidate", "approved"}
        )
        if not (remove_current_candidate or remove_retired_package):
            continue
        try:
            object_store.delete_uri(uri)
        except ObjectStoreError as exc:  # cleanup cannot invalidate an atomic promotion
            warnings.append(f"{uri}: {exc}")
    return warnings


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp, tolerating both ``Z`` and ``+00:00`` offsets."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _reject_pointer_rollback(
    previous: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    previous_approved = _parse_timestamp(previous.get("approved_at"))
    candidate_created = _parse_timestamp(manifest.get("created_at"))
    if previous_approved is None or candidate_created is None:
        return
    if candidate_created < previous_approved:
        raise ModelPublishError(
            f"Candidate was created at {candidate_created.isoformat()}, before the currently "
            f"approved model was approved at {previous_approved.isoformat()}. Pass "
            "allow_pointer_rollback to deliberately roll back."
        )


def read_current_pointer(
    root_uri: str, *, object_store: ObjectStore
) -> dict[str, Any] | None:
    raw = object_store.read_text(current_pointer_uri(root_uri))
    if raw is None:
        return None
    try:
        pointer = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelPublishError(
            f"{CURRENT_POINTER_FILENAME} is not valid JSON"
        ) from exc
    if not isinstance(pointer, dict):
        raise ModelPublishError(
            f"{CURRENT_POINTER_FILENAME} must contain a JSON object"
        )
    if pointer.get("schema_version") != CURRENT_SCHEMA_VERSION:
        raise ModelPublishError(
            f"Unsupported {CURRENT_POINTER_FILENAME} schema: {pointer.get('schema_version')!r}"
        )
    return pointer


def rollback_current_pointer(
    *,
    root_uri: str,
    object_store: ObjectStore,
    approved_by: str,
) -> dict[str, Any]:
    """Point ``CURRENT.json`` back at the previously approved model."""
    pointer = read_current_pointer(root_uri, object_store=object_store)
    if pointer is None:
        raise ModelPublishError(f"No {CURRENT_POINTER_FILENAME} to roll back")
    previous_version = pointer.get("previous_model_version")
    previous_uri = pointer.get("previous_package_uri")
    if not previous_version or not previous_uri:
        raise ModelPublishError("Current pointer has no previous model to roll back to")

    approved_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    rolled_back = {
        **pointer,
        "model_version": str(previous_version),
        "package_uri": str(previous_uri),
        "previous_model_version": pointer.get("model_version"),
        "previous_package_uri": pointer.get("package_uri"),
        "approved_at": approved_at,
        "approved_by": approved_by,
        "rolled_back_from": pointer.get("model_version"),
    }
    pointer_uri = current_pointer_uri(root_uri)
    object_store.write_text(
        pointer_uri,
        json.dumps(rolled_back, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        if_generation_match=object_store.generation(pointer_uri),
    )
    return rolled_back


def download_package(
    package_uri: str, destination: Path, *, object_store: ObjectStore
) -> list[str]:
    """Download every object under a package prefix into a local directory."""
    base = parse_gcs_uri(package_uri)
    prefix = base.prefix.rstrip("/") + "/"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    try:
        uris = object_store.list_uris(package_uri)
    except ObjectStoreError as exc:
        raise ModelPublishError(f"Failed to list {package_uri}: {exc}") from exc
    for uri in uris:
        object_name = parse_gcs_uri(uri).prefix
        relative = (
            object_name[len(prefix) :]
            if object_name.startswith(prefix)
            else object_name
        )
        if not relative or "/" in relative:
            # Model packages are flat by contract; nested objects are not part of it.
            continue
        object_store.download_file(uri, destination / relative)
        names.append(relative)
    return sorted(names)
