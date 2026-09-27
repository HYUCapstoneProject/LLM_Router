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
        ("ordinal", "model-package-v1", "direct-promotion-result-v1"),
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


def is_direct_promotion(result: Mapping[str, Any]) -> bool:
    schema = result.get("source_schema_version", result.get("schema_version"))
    return schema == "direct-promotion-result-v1"


def validate_direct_promotion(
    manifest: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    promotion_contract(manifest, result)
    if (
        not is_direct_promotion(result)
        or result.get("promotion_mode") != "direct"
        or result.get("quality_gate_skipped") is not True
        or result.get("integrity_verified") is not True
        or manifest.get("inference_max_length") != 8192
        or not manifest.get("length_contract_version")
        or "passed" in result
        or result.get("candidate_metrics_sha256") is not None
        or manifest.get("promotion_mode") != "direct"
        or manifest.get("quality_gate_skipped") is not True
        or manifest.get("evaluation_role") != "not_evaluated"
        or manifest.get("evaluation_version") != "not-evaluated-direct-promotion-v1"
        or any(
            name in manifest.get("files", {})
            for name in (
                "gate_metrics.json",
                "routerarena_metrics.json",
                "development_metrics.json",
            )
        )
    ):
        raise PromotionEvidenceError("Invalid explicit direct-promotion contract")
    for key in ("candidate_manifest_sha256", "baseline_manifest_sha256"):
        if (
            not isinstance(result.get(key), str)
            or re.fullmatch(r"[0-9a-f]{64}", result[key]) is None
        ):
            raise PromotionEvidenceError("Direct promotion identity digest missing")
    for key in (
        "batch_id",
        "candidate_model_version",
        "baseline_model_version",
        "pipeline_job_resource_name",
        "model_publish_root_uri",
    ):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise PromotionEvidenceError("Direct promotion identity missing")
    if (
        type(result.get("expected_current_generation")) is not int
        or result["expected_current_generation"] <= 0
    ):
        raise PromotionEvidenceError(
            "Direct promotion requires a frozen CURRENT generation"
        )
    if manifest.get("model_version") != result["candidate_model_version"]:
        raise PromotionEvidenceError("Direct promotion candidate mismatch")


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
    if is_direct_promotion(gate):
        validate_direct_promotion(manifest, gate)
        return
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
    promotion_mode: str = "quality_gate",
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
    direct = is_direct_promotion(gate)
    require(
        promotion_mode == ("direct" if direct else "quality_gate"),
        "Promotion mode mismatch",
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
        (direct or gate.get("passed") is True)
        and gate.get("batch_id") == batch_id
        and gate.get("candidate_model_version") == model_version
        and gate.get("pipeline_job_resource_name") == pipeline_run_id,
        "Gate does not authorize this batch, candidate and pipeline",
    )
    if direct:
        require(
            pointer.get("previous_pointer_generation")
            == gate.get("expected_current_generation")
            and pointer.get("previous_model_version")
            == gate.get("baseline_model_version")
            and gate.get("model_publish_root_uri") == root_uri.rstrip("/"),
            "Direct baseline generation, version or registry mismatch",
        )
    candidate_sha = gate.get("candidate_manifest_sha256")
    metrics_sha = gate.get("candidate_metrics_sha256")
    for digest in (candidate_sha,) if direct else (candidate_sha, metrics_sha):
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
        direct
        or (
            metric_name is not None
            and manifest.get("files", {}).get(metric_name) == metrics_sha
        ),
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
        **(
            {
                "promotion_mode": "direct",
                "quality_gate_skipped": True,
                "baseline_model_version": gate["baseline_model_version"],
                "baseline_manifest_sha256": gate["baseline_manifest_sha256"],
                "expected_current_generation": gate["expected_current_generation"],
            }
            if direct
            else {}
        ),
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
