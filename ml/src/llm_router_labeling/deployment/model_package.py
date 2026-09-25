from __future__ import annotations

import json
import re
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from ..shared.dataset_io import sha256_file, write_json
from ..training.routing_policy import (
    LENGTH_CONTRACT_VERSION,
    OUTPUT_LABEL_ORDERS,
    RoutingPolicyError,
    load_routing_policy,
)

SCHEMA_VERSION = "model-package-v1"
POLICY_SCHEMA_VERSION = "model-package-v2"
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
OPTIONAL_METADATA_FIELDS = {
    "evaluation_role",
    "split_version",
    "dataset_manifest_sha256",
    "input_archive_sha256",
    "logical_model_version",
    "training_mode",
    "training_fingerprint",
    "input_contract",
    "length_contract_version",
    "training_max_length",
    "inference_max_length",
}
POLICY_METADATA_FIELDS = {
    "objective",
    "input_contract",
    "routing_policy_file",
    "routing_policy_sha256",
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


def _package_schema(metadata: Mapping[str, Any]) -> str:
    schema = metadata.get("schema_version", SCHEMA_VERSION)
    if not isinstance(schema, str) or schema not in {
        SCHEMA_VERSION,
        POLICY_SCHEMA_VERSION,
    }:
        raise ModelPackageError(f"Unsupported model manifest schema: {schema!r}")
    return str(schema)


def _policy_metadata(metadata: Mapping[str, Any]) -> None:
    missing = sorted(POLICY_METADATA_FIELDS - set(metadata))
    if missing:
        raise ModelPackageError(f"V2 policy metadata is missing fields: {missing}")
    objective = metadata["objective"]
    if not isinstance(objective, str) or objective not in OUTPUT_LABEL_ORDERS:
        raise ModelPackageError("V2 objective must be ordinal, pointwise or pairwise")
    if metadata["routing_policy_file"] != "routing_policy.json":
        raise ModelPackageError("V2 routing_policy_file must be routing_policy.json")
    if (
        not isinstance(metadata["routing_policy_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", metadata["routing_policy_sha256"]) is None
    ):
        raise ModelPackageError(
            "V2 routing_policy_sha256 must be the actual file SHA-256"
        )
    if metadata["input_contract"] != "structure-preserving-v2":
        raise ModelPackageError("V2 input_contract must be structure-preserving-v2")
    allowed_lengths = (
        {512, 8192} if "length_contract_version" in metadata else {512}
    )
    if (
        type(metadata["tokenizer_max_length"]) is not int
        or metadata["tokenizer_max_length"] not in allowed_lengths
    ):
        raise ModelPackageError(
            f"V2 tokenizer_max_length must be in {sorted(allowed_lengths)}"
        )


def _validate_input_length_contract(
    package_dir: Path, metadata: Mapping[str, Any]
) -> None:
    """Validate immutable training history independently from serving length."""
    has_contract = "length_contract_version" in metadata
    if not has_contract and {"training_max_length", "inference_max_length"} & metadata.keys():
        raise ModelPackageError("Separate lengths require length_contract_version")
    if has_contract and metadata["length_contract_version"] != LENGTH_CONTRACT_VERSION:
        raise ModelPackageError("Unsupported package length_contract_version")
    if not has_contract and "input_contract" not in metadata:
        return
    if metadata.get("input_contract") != "structure-preserving-v2":
        raise ModelPackageError("Explicit input contract must be structure-preserving-v2")
    training = _load_json(package_dir / "training_config.json")
    if training.get("input_contract", "structure-preserving-v2") != "structure-preserving-v2":
        raise ModelPackageError("Training input contract differs from package input contract")
    if (
        _package_schema(metadata) == SCHEMA_VERSION
        and training.get("objective_method", "ordinal") != "ordinal"
    ):
        raise ModelPackageError("Explicit v1 input contract requires ordinal training")
    if not has_contract:
        if (
            type(metadata.get("tokenizer_max_length")) is not int
            or metadata["tokenizer_max_length"] != 512
            or type(training.get("max_length")) is not int
            or training["max_length"] != 512
        ):
            raise ModelPackageError("Legacy structured package requires 512-token limits")
        return
    config = _load_json(package_dir / "config.json")
    values = {
        "training_max_length": metadata.get("training_max_length"),
        "inference_max_length": metadata.get("inference_max_length"),
        "tokenizer_max_length": metadata.get("tokenizer_max_length"),
        "training_config.max_length": training.get("max_length"),
        "config.max_position_embeddings": config.get("max_position_embeddings"),
    }
    for name, value in values.items():
        if type(value) is not int or value <= 0:
            raise ModelPackageError(f"{name} must be a positive integer")
    if values["training_max_length"] != values["training_config.max_length"]:
        raise ModelPackageError("training_max_length differs from preserved training history")
    if (
        values["inference_max_length"] not in {512, 8192}
        or values["tokenizer_max_length"] != values["inference_max_length"]
    ):
        raise ModelPackageError("Length contract requires matching 512 or 8192 inference limits")
    if max(values["training_max_length"], values["inference_max_length"]) > values[
        "config.max_position_embeddings"
    ]:
        raise ModelPackageError("Package length exceeds the checkpoint native position limit")


def _validate_checkpoint_contract(
    package_dir: Path,
    *,
    schema_version: str = SCHEMA_VERSION,
    objective: str = "ordinal",
    allow_untagged_ordinal: bool = False,
) -> dict[str, Any]:
    config_path = package_dir / "config.json"
    tokenizer_config_path = package_dir / "tokenizer_config.json"
    if not config_path.is_file() or not tokenizer_config_path.is_file():
        raise ModelPackageError(
            "Checkpoint must contain config.json and tokenizer_config.json"
        )
    model_config = _load_json(config_path)
    if (
        schema_version == SCHEMA_VERSION
        and model_config.get("router_objective", "ordinal") != "ordinal"
    ):
        raise ModelPackageError(
            "The difficulty-classification package contract only supports the ordinal "
            "R0 objective; candidate-success and pairwise heads require their own "
            "serving and evaluation contract"
        )
    if schema_version == POLICY_SCHEMA_VERSION:
        actual_objective = model_config.get("router_objective")
        if (
            allow_untagged_ordinal
            and objective == "ordinal"
            and "router_objective" not in model_config
        ):
            actual_objective = "ordinal"
        if actual_objective != objective:
            raise ModelPackageError(
                "V2 checkpoint router_objective does not match package objective"
            )
        labels = OUTPUT_LABEL_ORDERS[objective]
        expected_id2label = {str(index): label for index, label in enumerate(labels)}
        expected_label2id = {label: index for index, label in enumerate(labels)}
    else:
        expected_id2label = EXPECTED_ID2LABEL
        expected_label2id = EXPECTED_LABEL2ID
    id2label = {
        str(index): str(label)
        for index, label in model_config.get("id2label", {}).items()
    }
    label2id = {
        str(label): int(index)
        for label, index in model_config.get("label2id", {}).items()
    }
    if id2label != expected_id2label or label2id != expected_label2id:
        raise ModelPackageError(f"Checkpoint labels must match {expected_id2label}")
    model_files = sorted(
        path.name
        for path in package_dir.iterdir()
        if path.name in {"model.safetensors", "pytorch_model.bin"}
    )
    if not model_files:
        raise ModelPackageError("Checkpoint has no saved model weights")
    return {"model_weight_files": model_files, "id2label": expected_id2label}


def _validate_policy_contract(package_dir: Path, metadata: Mapping[str, Any]) -> None:
    _policy_metadata(metadata)
    try:
        policy = load_routing_policy(
            package_dir / "routing_policy.json",
            expected_file_sha256=str(metadata["routing_policy_sha256"]),
        )
    except (OSError, RoutingPolicyError) as error:
        raise ModelPackageError(f"Invalid V2 routing policy: {error}") from error
    if policy.objective != metadata["objective"]:
        raise ModelPackageError("V2 policy objective does not match package objective")
    if (
        policy.input_contract != metadata["input_contract"]
        or policy.max_length != metadata["tokenizer_max_length"]
        or policy.length_contract_version != metadata.get("length_contract_version")
    ):
        raise ModelPackageError(
            "V2 policy input contract does not match package metadata"
        )
    training = _load_json(package_dir / "training_config.json")
    if "length_contract_version" not in metadata and (
        type(training.get("max_length")) is not int
        or training["max_length"] != policy.max_length
    ):
        raise ModelPackageError(
            "V2 training max_length must match the 512-token service policy"
        )
    if (
        "input_contract" in training
        and training["input_contract"] != policy.input_contract
    ):
        raise ModelPackageError(
            "V2 training input_contract does not match the service policy"
        )
    if training.get("objective_method", "ordinal") != policy.objective:
        raise ModelPackageError(
            "V2 training objective does not match the service policy"
        )


def validate_model_package(
    package_dir: Path, manifest_path: Path | None = None
) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    manifest_path = manifest_path or package_dir / "model_manifest.json"
    if not manifest_path.is_file():
        raise ModelPackageError(f"Model manifest not found: {manifest_path}")
    manifest = _load_json(manifest_path)

    if "schema_version" not in manifest:
        raise ModelPackageError("Model manifest is missing schema_version")
    schema = _package_schema(manifest)
    missing_metadata = sorted(REQUIRED_METADATA_FIELDS - set(manifest))
    if missing_metadata:
        raise ModelPackageError(
            f"Model manifest is missing metadata: {missing_metadata}"
        )
    if schema == POLICY_SCHEMA_VERSION:
        _policy_metadata(manifest)
    if manifest.get("gate_status") not in {"candidate", "approved"}:
        raise ModelPackageError("gate_status must be candidate or approved")
    if int(manifest["tokenizer_max_length"]) <= 0:
        raise ModelPackageError("tokenizer_max_length must be positive")

    checkpoint_contract = _validate_checkpoint_contract(
        package_dir,
        schema_version=schema,
        objective=str(manifest.get("objective", "ordinal")),
    )
    if manifest.get("id2label") != checkpoint_contract["id2label"]:
        raise ModelPackageError(
            "Model manifest labels do not match checkpoint output labels"
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ModelPackageError("Model manifest files must be a non-empty object")
    required_files = REQUIRED_PACKAGE_FILES | set(
        checkpoint_contract["model_weight_files"]
    )
    if schema == POLICY_SCHEMA_VERSION:
        required_files.add("routing_policy.json")
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

    _validate_input_length_contract(package_dir, manifest)
    if schema == POLICY_SCHEMA_VERSION:
        if files["routing_policy.json"] != manifest["routing_policy_sha256"]:
            raise ModelPackageError("V2 routing policy SHA must match its files entry")
        original_config_sha = manifest.get("checkpoint_config_sha256")
        if (
            not isinstance(original_config_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", original_config_sha) is None
        ):
            raise ModelPackageError(
                "V2 checkpoint_config_sha256 must preserve the original config SHA-256"
            )
        _validate_policy_contract(package_dir, manifest)

    return {
        "valid": True,
        "schema_version": schema,
        "model_version": str(manifest["model_version"]),
        "gate_status": str(manifest["gate_status"]),
        "id2label": checkpoint_contract["id2label"],
        "verified_files": verified,
        **{
            field: manifest[field]
            for field in (
                "input_contract", "length_contract_version",
                "training_max_length", "inference_max_length",
            )
            if field in manifest
        },
        **(
            {
                "objective": manifest["objective"],
                "input_contract": manifest["input_contract"],
                "routing_policy_file": manifest["routing_policy_file"],
                "routing_policy_sha256": manifest["routing_policy_sha256"],
                "deployment_readiness": "candidate_contract_only",
                "automatic_promotion_supported": False,
            }
            if schema == POLICY_SCHEMA_VERSION
            else {}
        ),
    }


def create_model_manifest(
    package_dir: Path,
    metadata: Mapping[str, Any],
    output_path: Path | None = None,
) -> dict[str, Any]:
    package_dir = package_dir.resolve()
    schema = _package_schema(metadata)
    missing_metadata = sorted(REQUIRED_METADATA_FIELDS - set(metadata))
    if missing_metadata:
        raise ModelPackageError(
            f"Model package metadata is missing fields: {missing_metadata}"
        )
    if schema == POLICY_SCHEMA_VERSION:
        _policy_metadata(metadata)
        if metadata.get("gate_status", "candidate") != "candidate":
            raise ModelPackageError(
                "V2 package creation only prepares candidates; objective gate support is still required"
            )
    checkpoint_contract = _validate_checkpoint_contract(
        package_dir,
        schema_version=schema,
        objective=str(metadata.get("objective", "ordinal")),
    )
    required_files = REQUIRED_PACKAGE_FILES | set(
        checkpoint_contract["model_weight_files"]
    )
    if schema == POLICY_SCHEMA_VERSION:
        required_files.add("routing_policy.json")
    absent = sorted(
        name for name in required_files if not (package_dir / name).is_file()
    )
    if absent:
        raise ModelPackageError(f"Model package is missing files: {absent}")
    _validate_input_length_contract(package_dir, metadata)
    if schema == POLICY_SCHEMA_VERSION:
        _validate_policy_contract(package_dir, metadata)

    manifest = {
        "schema_version": schema,
        **{field: metadata[field] for field in sorted(REQUIRED_METADATA_FIELDS)},
        **{
            field: metadata[field]
            for field in sorted(OPTIONAL_METADATA_FIELDS)
            if field in metadata
        },
        **(
            {
                **{field: metadata[field] for field in sorted(POLICY_METADATA_FIELDS)},
                "checkpoint_config_sha256": metadata.get(
                    "checkpoint_config_sha256", sha256_file(package_dir / "config.json")
                ),
            }
            if schema == POLICY_SCHEMA_VERSION
            else {}
        ),
        "id2label": checkpoint_contract["id2label"],
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
    routing_policy_path: Path | None = None,
) -> dict[str, Any]:
    """Assemble a candidate package from a checkpoint and static metadata.

    ``package_metadata_uri`` is one immutable GCS file shared by every pipeline
    run, so its ``model_version`` cannot vary per run on its own. ``model_version``
    overrides that field with the run's actual candidate version, which is what
    the deploy gate and the publish step both key off of.

    V2 assembly validates an explicit service policy and prepares a candidate
    artifact only. It does not supply an objective-compatible production gate
    or establish that public benchmark outcomes measure Claude service quality.
    """
    checkpoint_dir = checkpoint_dir.resolve()
    schema = _package_schema(metadata)
    if schema == POLICY_SCHEMA_VERSION:
        _policy_metadata(metadata)
        if routing_policy_path is None:
            raise ModelPackageError(
                "V2 assembly requires an explicit routing_policy_path"
            )
        if not routing_policy_path.is_file():
            raise ModelPackageError(
                f"Routing policy file not found: {routing_policy_path}"
            )
    elif routing_policy_path is not None:
        raise ModelPackageError(
            "routing_policy_path requires explicit model-package-v2 metadata"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_contract = _validate_checkpoint_contract(
        checkpoint_dir,
        schema_version=schema,
        objective=str(metadata.get("objective", "ordinal")),
        allow_untagged_ordinal=schema == POLICY_SCHEMA_VERSION,
    )
    original_config_sha = sha256_file(checkpoint_dir / "config.json")
    if (
        schema == POLICY_SCHEMA_VERSION
        and metadata.get("checkpoint_config_sha256", original_config_sha)
        != original_config_sha
    ):
        raise ModelPackageError(
            "checkpoint_config_sha256 does not match the source checkpoint"
        )
    filenames = REQUIRED_PACKAGE_FILES | set(checkpoint_contract["model_weight_files"])
    for filename in sorted(filenames):
        source = checkpoint_dir / filename
        if not source.is_file():
            raise ModelPackageError(f"Checkpoint file not found: {filename}")
        shutil.copy2(source, output_dir / filename)
    if schema == POLICY_SCHEMA_VERSION:
        shutil.copy2(routing_policy_path, output_dir / "routing_policy.json")
        model_config = _load_json(output_dir / "config.json")
        if (
            metadata["objective"] == "ordinal"
            and "router_objective" not in model_config
        ):
            # Preserve the source checkpoint; only annotate the new v2 copy.
            model_config["router_objective"] = "ordinal"
            write_json(output_dir / "config.json", model_config)
    if evaluation_metrics_path is not None:
        if not evaluation_metrics_path.is_file():
            raise ModelPackageError(
                f"Evaluation metrics not found: {evaluation_metrics_path}"
            )
        role = str(metadata.get("evaluation_role", "routerarena"))
        evaluation_filename = {
            "routerarena": "routerarena_metrics.json",
            "continual_gate": "gate_metrics.json",
            "development_calibration": "development_metrics.json",
        }.get(role)
        if evaluation_filename is None:
            raise ModelPackageError(f"Unknown evaluation role: {role}")
        metrics = _load_json(evaluation_metrics_path)
        if role != "routerarena" and metrics.get("evaluation_role") != role:
            raise ModelPackageError(
                "Evaluation metrics role does not match package metadata"
            )
        shutil.copy2(evaluation_metrics_path, output_dir / evaluation_filename)
    candidate_metadata = {**metadata, "gate_status": "candidate"}
    if schema == POLICY_SCHEMA_VERSION:
        candidate_metadata["checkpoint_config_sha256"] = original_config_sha
    if model_version:
        candidate_metadata["model_version"] = model_version
    manifest = create_model_manifest(output_dir, candidate_metadata)
    if evaluation_metrics_path is not None:
        manifest["files"][evaluation_filename] = sha256_file(
            output_dir / evaluation_filename
        )
        write_json(output_dir / "model_manifest.json", manifest)
        validate_model_package(output_dir)
    return manifest
