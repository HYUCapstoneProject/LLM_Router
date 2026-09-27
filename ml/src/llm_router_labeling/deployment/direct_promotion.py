"""Explicit R0 publication without quality evaluation; integrity checks still apply."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from ..shared.dataset_io import sha256_file, write_json
from ..shared.object_store import build_object_store
from ..shared.observability import write_observability_json
from .model_package import assemble_model_package, validate_model_package
from .model_publish import approve_candidate, publish_candidate, read_current_pointer
from .promotion_evidence import (
    PromotionEvidenceError,
    build_promotion_evidence,
    text_sha256,
    validate_gate_package_binding,
)


def promote_direct(
    *,
    checkpoint_dir,
    output_dir,
    metadata,
    model_version,
    baseline_model_version,
    root_uri,
    batch_id,
    pipeline_run_id,
    observability_run_uri,
    logical_model_version,
    training_mode,
    retention_anchor_model_version,
    object_store,
    promotion_mode,
):
    if promotion_mode != "direct":
        raise PromotionEvidenceError("Direct pipeline requires promotion_mode=direct")
    if training_mode not in {
        "minor_fine_tune",
        "major_consolidation",
        "pristine_rebase",
    }:
        raise PromotionEvidenceError(
            "Direct promotion only supports feedback retraining"
        )
    if not all(
        (batch_id, pipeline_run_id, observability_run_uri, baseline_model_version)
    ):
        raise PromotionEvidenceError("Direct promotion requires complete run identity")
    if (
        metadata.get("schema_version", "model-package-v1") != "model-package-v1"
        or metadata.get("objective", "ordinal") != "ordinal"
    ):
        raise PromotionEvidenceError(
            "Direct promotion only supports ordinal model-package-v1"
        )
    package = Path(output_dir) / "package"
    manifest_path = package / "model_manifest.json"
    direct_metadata = {
        **metadata,
        "promotion_mode": "direct",
        "quality_gate_skipped": True,
        "evaluation_role": "not_evaluated",
        "evaluation_version": "not-evaluated-direct-promotion-v1",
    }
    if manifest_path.exists():
        validate_model_package(package)
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("model_version") != model_version:
            raise PromotionEvidenceError("Existing output belongs to another candidate")
        for name, digest in manifest["files"].items():
            if sha256_file(Path(checkpoint_dir) / name) != digest:
                raise PromotionEvidenceError("Existing package differs from checkpoint")
    else:
        manifest = assemble_model_package(
            Path(checkpoint_dir), package, direct_metadata, model_version=model_version
        )
    pointer_uri = root_uri.rstrip("/") + "/CURRENT.json"
    direct_uri = observability_run_uri.rstrip("/") + "/promotion/direct-result.json"
    current = read_current_pointer(root_uri, object_store=object_store)
    existing = object_store.read_text(direct_uri)
    if existing:
        authorization = json.loads(existing)
        authorization["schema_version"] = authorization.pop(
            "source_schema_version", authorization["schema_version"]
        )
        authorization.pop("updated_at", None)
    else:
        generation = object_store.generation(pointer_uri)
        current = read_current_pointer(root_uri, object_store=object_store)
        if (
            not current
            or current.get("model_version") != baseline_model_version
            or generation is None
            or object_store.generation(pointer_uri) != generation
        ):
            raise PromotionEvidenceError("CURRENT differs from direct baseline")
        baseline_text = object_store.read_text(
            current["package_uri"] + "/model_manifest.json"
        )
        if baseline_text is None or text_sha256(baseline_text) != current.get(
            "manifest_sha256"
        ):
            raise PromotionEvidenceError("Baseline manifest checksum mismatch")
        baseline = json.loads(baseline_text)
        if (
            baseline.get("schema_version") != "model-package-v1"
            or baseline.get("objective", "ordinal") != "ordinal"
            or baseline.get("model_version") != baseline_model_version
            or baseline.get("gate_status") != "approved"
            or baseline.get("inference_max_length")
            != manifest.get("inference_max_length")
        ):
            raise PromotionEvidenceError(
                "Baseline objective, identity or length mismatch"
            )
        authorization = {
            "schema_version": "direct-promotion-result-v1",
            "promotion_mode": "direct",
            "quality_gate_skipped": True,
            "integrity_verified": True,
            "objective": "ordinal",
            "package_schema_version": "model-package-v1",
            "gate_schema_version": "direct-promotion-result-v1",
            "batch_id": batch_id,
            "pipeline_job_resource_name": pipeline_run_id,
            "candidate_model_version": model_version,
            "candidate_manifest_sha256": sha256_file(manifest_path),
            "candidate_metrics_sha256": None,
            "baseline_model_version": baseline_model_version,
            "baseline_manifest_sha256": current["manifest_sha256"],
            "model_publish_root_uri": root_uri.rstrip("/"),
            "expected_current_generation": generation,
            "length_contract_version": manifest.get("length_contract_version"),
            "inference_max_length": manifest.get("inference_max_length"),
        }
    validate_gate_package_binding(manifest, authorization)
    if any(
        authorization.get(key) != value
        for key, value in {
            "candidate_manifest_sha256": sha256_file(manifest_path),
            "batch_id": batch_id,
            "pipeline_job_resource_name": pipeline_run_id,
            "baseline_model_version": baseline_model_version,
            "model_publish_root_uri": root_uri.rstrip("/"),
        }.items()
    ):
        raise PromotionEvidenceError("Direct authorization identity mismatch")
    write_observability_json(direct_uri, authorization, object_store=object_store)
    write_observability_json(
        observability_run_uri.rstrip("/") + "/metrics/packaging.json",
        {
            "batch_id": batch_id,
            "candidate_model_version": model_version,
            "package_file_count": len(manifest["files"]),
            "quality_gate_skipped": True,
        },
        object_store=object_store,
    )
    if current and current.get("model_version") == model_version:
        if (
            current.get("pipeline_run_id") != pipeline_run_id
            or current.get("candidate_manifest_sha256")
            != authorization["candidate_manifest_sha256"]
        ):
            raise PromotionEvidenceError("CURRENT belongs to another promotion")
        result = current
    else:
        publish_candidate(
            package,
            root_uri=root_uri,
            model_version=model_version,
            object_store=object_store,
            gate_result=authorization,
            promotion_mode="direct",
        )
        with tempfile.TemporaryDirectory(prefix="direct-approve-") as staging:
            result = approve_candidate(
                root_uri=root_uri,
                model_version=model_version,
                object_store=object_store,
                approved_by="vertex-pipeline-direct-promotion",
                pipeline_run_id=pipeline_run_id,
                work_dir=Path(staging),
                expected_baseline_model_version=baseline_model_version,
                expected_baseline_manifest_sha256=authorization[
                    "baseline_manifest_sha256"
                ],
                logical_model_version=logical_model_version,
                training_mode=training_mode,
                retention_anchor_model_version=retention_anchor_model_version,
                direct_authorization=authorization,
            )
    evidence = build_promotion_evidence(
        root_uri=root_uri,
        model_version=model_version,
        batch_id=batch_id,
        pipeline_run_id=pipeline_run_id,
        result=result,
        store=object_store,
        gate_text=object_store.read_text(direct_uri),
        promotion_mode="direct",
    )
    write_observability_json(
        observability_run_uri.rstrip("/") + "/result/publish-result.json",
        evidence,
        object_store=object_store,
    )
    write_json(Path(output_dir) / "approval_result.json", result)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkpoint-dir",
        "output-dir",
        "metadata",
        "model-version",
        "baseline-model-version",
        "root-uri",
        "batch-id",
        "pipeline-run-id",
        "observability-run-uri",
        "logical-model-version",
        "training-mode",
        "retention-anchor-model-version",
        "promotion-mode",
    ):
        parser.add_argument("--" + name, required=True)
    args = vars(parser.parse_args())
    args["metadata"] = json.loads(Path(args["metadata"]).read_text())
    args["object_store"] = build_object_store(args["root_uri"])
    result = promote_direct(**args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
