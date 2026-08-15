from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ..shared.dataset_io import sha256_file, write_json
SCHEMA_VERSION = "model-package-v1"
EXPECTED_ID2LABEL = {"0": "easy", "1": "medium", "2": "hard"}
EXPECTED_LABEL2ID = {label: int(index) for index, label in EXPECTED_ID2LABEL.items()}
REQUIRED_METADATA_FIELDS = {
    "model_version",
    "base_model",
    "dataset_version",
    "preprocessing_version",
    "label_rule_version",
    "training_version",
    "evaluation_version",
    "tokenizer_max_length",
    "code_commit",
}
REQUIRED_PACKAGE_FILES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "metrics.json",
    "training_config.json",
}


class ModelPackageError(RuntimeError):
    """Raised when a model package violates the promotion artifact contract."""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ModelPackageError(f"{path.name} must contain a JSON object")
    return value


def _safe_package_file(package_dir: Path, relative_name: str) -> Path:
    relative = PurePosixPath(relative_name)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
        raise ModelPackageError(
            f"Package file must be a top-level relative filename: {relative_name!r}"
        )
    return package_dir / relative_name


def _validate_checkpoint_contract(package_dir: Path) -> dict[str, Any]:
    config_path = package_dir / "config.json"
    tokenizer_config_path = package_dir / "tokenizer_config.json"
    if not config_path.is_file() or not tokenizer_config_path.is_file():
        raise ModelPackageError(
            "Checkpoint must contain config.json and tokenizer_config.json"
        )
    model_config = _load_json(config_path)
    id2label = {
        str(index): str(label)
        for index, label in model_config.get("id2label", {}).items()
    }
    label2id = {
        str(label): int(index)
        for label, index in model_config.get("label2id", {}).items()
    }
    if id2label != EXPECTED_ID2LABEL or label2id != EXPECTED_LABEL2ID:
        raise ModelPackageError("Checkpoint labels must be easy=0, medium=1, hard=2")
    model_files = sorted(
        path.name
        for path in package_dir.iterdir()
        if path.name in {"model.safetensors", "pytorch_model.bin"}
    )
    if not model_files:
        raise ModelPackageError("Checkpoint has no saved model weights")
    return {"model_weight_files": model_files}


def validate_model_package(
    package_dir: Path, manifest_path: Path | None = None
) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    manifest_path = manifest_path or package_dir / "model_manifest.json"
    if not manifest_path.is_file():
        raise ModelPackageError(f"Model manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ModelPackageError(
            f"Unsupported model manifest schema: {manifest.get('schema_version')!r}"
        )
    missing_metadata = sorted(REQUIRED_METADATA_FIELDS - set(manifest))
    if missing_metadata:
        raise ModelPackageError(
            f"Model manifest is missing metadata: {missing_metadata}"
        )
    if manifest.get("id2label") != EXPECTED_ID2LABEL:
        raise ModelPackageError("Model manifest labels must be easy=0, medium=1, hard=2")
    if manifest.get("gate_status") not in {"candidate", "approved"}:
        raise ModelPackageError("gate_status must be candidate or approved")
    if int(manifest["tokenizer_max_length"]) <= 0:
        raise ModelPackageError("tokenizer_max_length must be positive")

    checkpoint_contract = _validate_checkpoint_contract(package_dir)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ModelPackageError("Model manifest files must be a non-empty object")
    required_files = REQUIRED_PACKAGE_FILES | set(
        checkpoint_contract["model_weight_files"]
    )
    missing_files = sorted(required_files - set(files))
    if missing_files:
        raise ModelPackageError(
            f"Model manifest is missing required files: {missing_files}"
        )

    verified: dict[str, str] = {}
    for relative_name, expected_sha256 in sorted(files.items()):
        file_path = _safe_package_file(package_dir, str(relative_name))
        if not file_path.is_file():
            raise ModelPackageError(f"Model package file not found: {relative_name}")
        actual_sha256 = sha256_file(file_path)
        if actual_sha256 != str(expected_sha256).removeprefix("sha256:").lower():
            raise ModelPackageError(f"SHA-256 mismatch for {relative_name}")
        verified[str(relative_name)] = actual_sha256

    return {
        "valid": True,
        "schema_version": SCHEMA_VERSION,
        "model_version": str(manifest["model_version"]),
        "gate_status": str(manifest["gate_status"]),
        "id2label": EXPECTED_ID2LABEL,
        "verified_files": verified,
    }


def create_model_manifest(
    package_dir: Path,
    metadata: Mapping[str, Any],
    output_path: Path | None = None,
) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    missing_metadata = sorted(REQUIRED_METADATA_FIELDS - set(metadata))
    if missing_metadata:
        raise ModelPackageError(
            f"Model package metadata is missing fields: {missing_metadata}"
        )
    checkpoint_contract = _validate_checkpoint_contract(package_dir)
    required_files = REQUIRED_PACKAGE_FILES | set(
        checkpoint_contract["model_weight_files"]
    )
    absent = sorted(name for name in required_files if not (package_dir / name).is_file())
    if absent:
        raise ModelPackageError(f"Model package is missing files: {absent}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        **{field: metadata[field] for field in sorted(REQUIRED_METADATA_FIELDS)},
        "id2label": EXPECTED_ID2LABEL,
        "gate_status": str(metadata.get("gate_status", "candidate")),
        "created_at": str(
            metadata.get(
                "created_at", datetime.now(timezone.utc).isoformat(timespec="seconds")
            )
        ),
        "files": {
            name: sha256_file(package_dir / name) for name in sorted(required_files)
        },
    }
    output_path = output_path or package_dir / "model_manifest.json"
    write_json(output_path, manifest)
    validate_model_package(package_dir, output_path)
    return manifest


def assemble_model_package(
    checkpoint_dir: Path,
    output_dir: Path,
    metadata: Mapping[str, Any],
    evaluation_metrics_path: Path | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Assemble a candidate package from a checkpoint and static metadata.

    ``package_metadata_uri`` is one immutable GCS file shared by every pipeline
    run, so its ``model_version`` cannot vary per run on its own. ``model_version``
    overrides that field with the run's actual candidate version, which is what
    the deploy gate and the publish step both key off of.
    """
    checkpoint_dir = checkpoint_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_contract = _validate_checkpoint_contract(checkpoint_dir)
    filenames = REQUIRED_PACKAGE_FILES | set(
        checkpoint_contract["model_weight_files"]
    )
    for filename in sorted(filenames):
        source = checkpoint_dir / filename
        if not source.is_file():
            raise ModelPackageError(f"Checkpoint file not found: {filename}")
        shutil.copy2(source, output_dir / filename)
    if evaluation_metrics_path is not None:
        if not evaluation_metrics_path.is_file():
            raise ModelPackageError(
                f"Evaluation metrics not found: {evaluation_metrics_path}"
            )
        shutil.copy2(
            evaluation_metrics_path, output_dir / "routerarena_metrics.json"
        )
    candidate_metadata = {**metadata, "gate_status": "candidate"}
    if model_version:
        candidate_metadata["model_version"] = model_version
    manifest = create_model_manifest(output_dir, candidate_metadata)
    if evaluation_metrics_path is not None:
        manifest["files"]["routerarena_metrics.json"] = sha256_file(
            output_dir / "routerarena_metrics.json"
        )
        write_json(output_dir / "model_manifest.json", manifest)
        validate_model_package(output_dir)
    return manifest
