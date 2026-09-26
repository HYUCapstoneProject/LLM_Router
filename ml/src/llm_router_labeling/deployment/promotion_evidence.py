"""Explicit objective contracts for promotion evidence, independent of model loading.

Adding a new objective requires a new gate implementation and an explicit entry;
the generic evidence envelope never upgrades a legacy CE gate to a routing gate.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

PROMOTION_EVIDENCE_SCHEMA = "router-promotion-evidence-v1"
SUPPORTED_PROMOTION_CONTRACTS = frozenset(
    {
        ("ordinal", "model-package-v1", "deploy-gate-result-v1"),
    }
)


class PromotionEvidenceError(ValueError):
    pass


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def promotion_contract(
    manifest: Mapping[str, Any], gate: Mapping[str, Any]
) -> dict[str, str]:
    schema = manifest.get("schema_version")
    # Only the historical v1 schema has an implicit ordinal objective.
    objective = manifest.get(
        "objective", "ordinal" if schema == "model-package-v1" else None
    )
    gate_schema = (
        gate.get("source_schema_version")
        if gate.get("schema_version") == "retraining-observability-v1"
        else gate.get("schema_version")
    )
    if (objective, schema, gate_schema) not in SUPPORTED_PROMOTION_CONTRACTS:
        raise PromotionEvidenceError(
            "No approved objective/package/gate mapping; the CE gate cannot promote a v2 head"
        )
    for key, expected in (
        ("objective", objective),
        ("package_schema_version", schema),
        ("gate_schema_version", gate_schema),
    ):
        if key in gate and gate[key] != expected:
            raise PromotionEvidenceError(
                f"Gate {key} differs from its candidate package"
            )
    return {
        "objective": objective,
        "package_schema_version": schema,
        "gate_schema_version": gate_schema,
    }


def validate_gate_package_binding(
    manifest: Mapping[str, Any], gate: Mapping[str, Any]
) -> None:
    """Check qualification bindings before publication or pointer mutation."""
    promotion_contract(manifest, gate)
    if manifest.get("length_contract_version") is not None and (
        gate.get("length_contract_version") != manifest["length_contract_version"]
        or type(gate.get("inference_max_length")) is not int
        or gate["inference_max_length"] != manifest.get("inference_max_length")
    ):
        raise PromotionEvidenceError(
            "Gate inference length contract differs from package"
        )
    metric_name = {
        "continual_gate": "gate_metrics.json",
        "routerarena": "routerarena_metrics.json",
        "development_calibration": "development_metrics.json",
    }.get(manifest.get("evaluation_role", "routerarena"))
    digest = gate.get("candidate_metrics_sha256")
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or metric_name is None
        or manifest.get("files", {}).get(metric_name) != digest
    ):
        raise PromotionEvidenceError("Gate evaluation is not bound to its package")


def build_promotion_evidence(
    *,
    root_uri: str,
    model_version: str,
    batch_id: str,
    pipeline_run_id: str,
    result: Mapping[str, Any],
    store: Any,
    gate_text: str,
) -> dict[str, Any]:
    """Bind the gate and exact approved manifest to a stable CURRENT read."""

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise PromotionEvidenceError(message)

    pointer_uri = root_uri.rstrip("/") + "/CURRENT.json"
    package_uri = root_uri.rstrip("/") + f"/{model_version}/approved"
    generation = store.generation(pointer_uri)
    text = store.read_text(pointer_uri)
    require(text is not None, "CURRENT is missing after approval")
    pointer = json.loads(text)
    require(isinstance(pointer, dict), "CURRENT is not an object")
    for key in (
        "model_version",
        "package_uri",
        "manifest_sha256",
        "approved_at",
        "pipeline_run_id",
        "candidate_manifest_sha256",
        "previous_pointer_generation",
    ):
        require(
            pointer.get(key) == result.get(key),
            "CURRENT changed or differs from approval result",
        )
    require(
        pointer.get("model_version") == model_version
        and pointer.get("package_uri") == package_uri,
        "CURRENT identity differs from the requested promotion",
    )
    manifest_text = store.read_text(package_uri + "/model_manifest.json")
    require(
        manifest_text is not None
        and text_sha256(manifest_text) == pointer.get("manifest_sha256"),
        "Approved manifest checksum mismatch",
    )
    manifest = json.loads(manifest_text)
    gate = json.loads(gate_text)
    require(
        isinstance(manifest, dict) and isinstance(gate, dict),
        "Promotion documents must be objects",
    )
    contract = promotion_contract(manifest, gate)
    validate_gate_package_binding(manifest, gate)
    length_evidence = {}
    if manifest.get("length_contract_version") is not None:
        require(
            gate.get("length_contract_version") == manifest["length_contract_version"]
            and type(gate.get("inference_max_length")) is int
            and gate["inference_max_length"] == manifest.get("inference_max_length"),
            "Gate inference length contract differs from approved package",
        )
        length_evidence = {
            "length_contract_version": manifest["length_contract_version"],
            "inference_max_length": manifest["inference_max_length"],
        }
    require(
        gate.get("passed") is True
        and gate.get("batch_id") == batch_id
        and gate.get("candidate_model_version") == model_version
        and gate.get("pipeline_job_resource_name") == pipeline_run_id,
        "Gate does not authorize this batch, candidate and pipeline",
    )
    candidate_sha = gate.get("candidate_manifest_sha256")
    metrics_sha = gate.get("candidate_metrics_sha256")
    for digest in (candidate_sha, metrics_sha):
        require(
            isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            "Candidate manifest/metrics checksum missing",
        )
    require(
        pointer.get("candidate_manifest_sha256") == candidate_sha,
        "Gated candidate differs from approved candidate",
    )
    metric_name = {
        "continual_gate": "gate_metrics.json",
        "routerarena": "routerarena_metrics.json",
        "development_calibration": "development_metrics.json",
    }.get(manifest.get("evaluation_role", "routerarena"))
    require(
        metric_name is not None
        and manifest.get("files", {}).get(metric_name) == metrics_sha,
        "Approved manifest does not contain the gated evaluation",
    )
    require(
        manifest.get("model_version") == model_version
        and manifest.get("pipeline_run_id") == pipeline_run_id
        and pointer.get("pipeline_run_id") == pipeline_run_id
        and manifest.get("gate_status") == "approved"
        and manifest.get("approved_at") == pointer.get("approved_at"),
        "Approved manifest identity mismatch",
    )
    require(
        store.read_text(pointer_uri) == text
        and store.generation(pointer_uri) == generation,
        "CURRENT changed during evidence verification",
    )
    return {
        "schema_version": PROMOTION_EVIDENCE_SCHEMA,
        **contract,
        **length_evidence,
        "batch_id": batch_id,
        "candidate_model_version": model_version,
        "outcome": "promoted",
        "pipeline_job_resource_name": pipeline_run_id,
        "model_version": model_version,
        "package_uri": package_uri,
        "candidate_manifest_sha256": candidate_sha,
        "candidate_metrics_sha256": metrics_sha,
        "manifest_sha256": pointer["manifest_sha256"],
        "approved_manifest_sha256": pointer["manifest_sha256"],
        "approved_at": pointer["approved_at"],
        "current_pointer_uri": pointer_uri,
        "current_pointer_sha256": text_sha256(text),
        "current_pointer_generation": generation,
        "previous_pointer_generation": pointer.get("previous_pointer_generation"),
        "gate_result_sha256": text_sha256(gate_text),
    }
