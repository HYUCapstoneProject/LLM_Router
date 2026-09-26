from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..shared.dataset_io import sha256_file, write_json
from ..shared.object_store import build_object_store
from ..shared.observability import write_observability_json
from .deploy_gate import evaluate_deploy_gate
from .model_package import validate_model_package
from .model_publish import approved_uri
from .promotion_evidence import promotion_contract


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate model promotion rules")
    parser.add_argument("--candidate-package", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--candidate-model-version", required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--baseline-model-version")
    parser.add_argument("--observability-run-uri")
    parser.add_argument("--batch-id")
    parser.add_argument("--pipeline-run-id")
    parser.add_argument("--model-publish-root-uri")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.observability_run_uri and (not args.batch_id or not args.pipeline_run_id):
        raise SystemExit("observability requires batch-id and pipeline-run-id")
    validation = validate_model_package(args.candidate_package)
    candidate_manifest = _read_json(args.candidate_package / "model_manifest.json")
    if validation.get("schema_version") != "model-package-v1":
        raise SystemExit("The CE deploy gate accepts R0 model-package-v1 only")
    candidate_sha = sha256_file(args.candidate_metrics)
    candidate_metrics = _read_json(args.candidate_metrics)
    role = candidate_metrics.get("evaluation_role", "routerarena")
    metrics_filename = {
        "continual_gate": "gate_metrics.json",
        "routerarena": "routerarena_metrics.json",
        "development_calibration": "development_metrics.json",
    }.get(role)
    if metrics_filename is None or candidate_sha != validation["verified_files"].get(
        metrics_filename
    ):
        raise SystemExit("Candidate evaluation is not bound to the verified package")
    baseline_evidence = {}
    if args.model_publish_root_uri:
        if role != "continual_gate":
            raise SystemExit(
                "R0 pipeline qualification requires protected continual_gate evaluation"
            )
        if not args.baseline_model_version or not args.baseline_metrics:
            raise SystemExit("Approved baseline identity and metrics are required")
        store = build_object_store(args.model_publish_root_uri)
        uri = approved_uri(args.model_publish_root_uri, args.baseline_model_version)
        baseline_text = store.read_text(uri.rstrip("/") + "/model_manifest.json")
        if baseline_text is None:
            raise SystemExit("Approved baseline manifest is missing")
        baseline_manifest = json.loads(baseline_text)
        if (
            baseline_manifest.get("schema_version") != "model-package-v1"
            or baseline_manifest.get("objective", "ordinal") != "ordinal"
            or baseline_manifest.get("evaluation_role") != "continual_gate"
            or baseline_manifest.get("id2label")
            != {"0": "easy", "1": "medium", "2": "hard"}
            or baseline_manifest.get("model_version") != args.baseline_model_version
            or baseline_manifest.get("gate_status") != "approved"
        ):
            raise SystemExit("Approved baseline manifest identity mismatch")
        if sha256_file(args.baseline_metrics) != baseline_manifest.get("files", {}).get(
            metrics_filename
        ):
            raise SystemExit("Baseline metrics are not bound to the approved package")
        if candidate_manifest.get("length_contract_version") is not None:
            inference_length = candidate_manifest["inference_max_length"]
            if (
                baseline_manifest.get("length_contract_version")
                != candidate_manifest["length_contract_version"]
                or type(baseline_manifest.get("inference_max_length")) is not int
                or baseline_manifest["inference_max_length"] != inference_length
                or type(baseline_manifest.get("tokenizer_max_length")) is not int
                or baseline_manifest["tokenizer_max_length"] != inference_length
            ):
                raise SystemExit(
                    "Approved baseline inference length contract differs from candidate"
                )
            baseline_length = (
                _read_json(args.baseline_metrics)
                .get("model_info", {})
                .get("max_length")
            )
            if type(baseline_length) is not int or baseline_length != inference_length:
                raise SystemExit(
                    "Approved baseline metrics inference length differs from its package"
                )
        from .promotion_evidence import text_sha256

        baseline_evidence["baseline_manifest_sha256"] = text_sha256(baseline_text)
    result = evaluate_deploy_gate(
        candidate_metrics,
        _read_json(args.rules),
        candidate_model_version=args.candidate_model_version,
        package_validation=validation,
        baseline_metrics=(
            _read_json(args.baseline_metrics) if args.baseline_metrics else None
        ),
        baseline_model_version=args.baseline_model_version,
    )
    result.update(promotion_contract(candidate_manifest, result))
    result.update(baseline_evidence)
    if candidate_manifest.get("length_contract_version") is not None:
        result.update(
            length_contract_version=candidate_manifest["length_contract_version"],
            inference_max_length=candidate_manifest["inference_max_length"],
        )
    result.update(
        candidate_manifest_sha256=sha256_file(
            args.candidate_package / "model_manifest.json"
        ),
        candidate_metrics_sha256=candidate_sha,
        pipeline_job_resource_name=args.pipeline_run_id,
        batch_id=args.batch_id,
    )
    write_json(args.output, result)
    if args.observability_run_uri:
        if not args.batch_id:
            raise SystemExit("observability requires --batch-id")
        write_observability_json(
            f"{args.observability_run_uri.rstrip('/')}/gate/gate-result.json",
            {
                "batch_id": args.batch_id,
                "candidate_model_version": args.candidate_model_version,
                **result,
            },
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
