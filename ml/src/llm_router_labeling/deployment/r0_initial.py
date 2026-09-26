"""Qualify the first R0 baseline without inventing a previously approved model.

Local-only qualification. Absolute thresholds must be frozen on D_dev before
reading D_gate; ordinary retraining continues to require its approved baseline.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from ..training.routing_policy import LENGTH_CONTRACT_VERSION
from .model_package import validate_model_package
from .promotion_evidence import promotion_contract
from .r0_release import (
    _check_metrics_contract,
    _evaluation_contract,
    _read,
    _sha,
    r0_evaluation_profile,
)


def qualify_r0_initial(
    *,
    config_dir: Path,
    candidate_package: Path,
    gate_metrics: Path,
    development_metrics: Path,
    gate_dataset: Path,
    development_dataset: Path,
    rules_path: Path,
    pilot_report: Path,
    root_uri: str,
    expected_current_generation: int | None,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("Use a new immutable qualification output")
    if not root_uri.startswith("gs://") or not root_uri.rstrip("/").endswith(
        "/models/router-modernbert"
    ):
        raise ValueError("Explicit model registry root URI is required")
    if expected_current_generation is not None and (
        type(expected_current_generation) is not int or expected_current_generation <= 0
    ):
        raise ValueError(
            "CURRENT generation must be a positive integer or null for absent"
        )
    evaluation = r0_evaluation_profile(_read(config_dir / "continual_gate_v1.json"))
    development = r0_evaluation_profile(
        _read(config_dir / "initial_v1_dev_evaluation.json")
    )
    if (
        evaluation["evaluation_dataset_sha256"]
        == development["evaluation_dataset_sha256"]
    ):
        raise ValueError("D_dev and D_gate must be distinct")
    for dataset, config in (
        (gate_dataset, evaluation),
        (development_dataset, development),
    ):
        if _sha(dataset) != config["evaluation_dataset_sha256"]:
            raise ValueError("Frozen evaluation dataset SHA mismatch")
    validation = validate_model_package(candidate_package)
    manifest = _read(candidate_package / "model_manifest.json")
    expected = {
        "schema_version": "model-package-v1",
        "gate_status": "candidate",
        "input_contract": "structure-preserving-v2",
        "length_contract_version": LENGTH_CONTRACT_VERSION,
        "training_max_length": 512,
        "inference_max_length": 8192,
        "tokenizer_max_length": 8192,
        "evaluation_role": "continual_gate",
        "evaluation_version": evaluation["evaluation_version"],
        "dataset_manifest_sha256": evaluation["dataset_manifest_sha256"],
        **evaluation["dataset_contract"],
    }
    if manifest.get("objective", "ordinal") != "ordinal" or any(
        manifest.get(k) != v for k, v in expected.items()
    ):
        raise ValueError(
            "Initial candidate must be an R0 512-training/8192-inference package"
        )
    if validation["verified_files"].get("gate_metrics.json") != _sha(gate_metrics):
        raise ValueError("Gate metrics are not bound to candidate package")
    candidate = _read(gate_metrics)
    dev = _read(development_metrics)
    _check_metrics_contract(candidate, evaluation)
    _check_metrics_contract(dev, development)
    rules = _read(rules_path)
    version = rules.get("rules_version")
    if (
        not isinstance(version, str)
        or not version.strip()
        or any(w in version.lower() for w in ("smoke", "placeholder", "replace"))
    ):
        raise ValueError("Explicit non-smoke initial rules version required")
    if (
        rules.get("schema_version") != "r0-initial-rules-v1"
        or rules.get("minimum_delta")
        or rules.get("maximum_delta")
    ):
        raise ValueError(
            "Initial qualification requires absolute rules, not self-baseline deltas"
        )
    if rules.get("required_evaluation_contract") != _evaluation_contract(evaluation):
        raise ValueError("Initial rules must pin the entire frozen D_gate contract")
    required = {
        "macro_f1",
        "per_class.hard.recall",
        "per_source_dataset.routerbench.macro_f1",
        "per_source_dataset.xroutebench.macro_f1",
        *{
            f"per_domain.{d}.macro_f1"
            for d in evaluation["category_to_domain"].values()
        },
    }
    report_only = rules.get("report_only_domains", {})
    if not isinstance(report_only, dict):
        raise TypeError("report_only_domains must be an object")
    if report_only:
        import pandas as pd

        from ..evaluation.continual_gate import validate_continual_evaluation

        expected_policy = {
            "policy_version": "initial-small-domain-report-only-v1",
            "minimum_development_rows": 30,
            "frozen_before_gate": True,
            "scope": "initial_baseline_only",
        }
        if rules.get("small_domain_policy") != expected_policy:
            raise ValueError("Unsupported initial small-domain policy")
        validated_dev = validate_continual_evaluation(
            pd.read_parquet(development_dataset), development
        )
        counts = validated_dev.groupby("domain").size().to_dict()
        expected_reports = {
            str(domain): {
                "development_row_count": int(count),
                "reason": "insufficient_support",
                "quality_unverified": True,
            }
            for domain, count in counts.items()
            if count < 30
        }
        if report_only != expected_reports:
            raise ValueError(
                "Report-only domains must match actual frozen D_dev row counts"
            )
        for domain, details in report_only.items():
            path = f"per_domain.{domain}.macro_f1"
            required.discard(path)
            if path in rules.get("minimum_metrics", {}) or path in rules.get(
                "maximum_metrics", {}
            ):
                raise ValueError(
                    "Report-only domain cannot have a numeric pass threshold"
                )
            for metrics in (candidate, dev):
                summary = metrics.get("per_domain", {}).get(domain, {})
                value = summary.get("macro_f1")
                count = summary.get("row_count")
                if (
                    type(value) not in (float, int)
                    or not math.isfinite(value)
                    or not 0 <= value <= 1
                    or type(count) is not int
                    or count <= 0
                ):
                    raise ValueError(
                        "Report-only domain metrics must remain present and finite"
                    )
            if (
                dev["per_domain"][domain]["row_count"]
                != details["development_row_count"]
            ):
                raise ValueError("Report-only development metric support mismatch")
    minimum = rules.get("minimum_metrics", {})
    if not isinstance(minimum, dict) or not required.issubset(minimum):
        raise ValueError(
            "Initial absolute floors must cover F1, hard recall, sources and domains"
        )
    for path in required:
        threshold = minimum[path]
        if (
            type(threshold) not in (int, float)
            or not math.isfinite(threshold)
            or not 0 < threshold <= 1
        ):
            raise ValueError(
                "Required initial quality floors must be finite and in (0, 1]"
            )
    selection = rules.get("threshold_selection")
    expected_selection = {
        "evaluation_role": "development_calibration",
        "dataset_sha256": development["evaluation_dataset_sha256"],
        "frozen_before_gate": True,
        "report_sha256": _sha(pilot_report),
    }
    if selection != expected_selection:
        raise ValueError("Initial rules require frozen D_dev report provenance")
    pilot = _read(pilot_report)
    expected_pilot = {
        "schema_version": "r0-initial-pilot-v1",
        "objective": "ordinal",
        "evaluation_role": "development_calibration",
        "dataset_sha256": development["evaluation_dataset_sha256"],
        "pilot_metrics_sha256": _sha(development_metrics),
        "candidate_model_version": manifest["model_version"],
        "candidate_weights_sha256": manifest["files"]["model.safetensors"],
        "length_contract_version": LENGTH_CONTRACT_VERSION,
        "inference_max_length": 8192,
        "selected_rules": {
            k: v for k, v in rules.items() if k != "threshold_selection"
        },
    }
    if any(pilot.get(k) != v for k, v in expected_pilot.items()):
        raise ValueError(
            "Initial pilot must bind candidate weights, D_dev metrics and selected rules"
        )
    checks = []
    for group, operator in (("minimum_metrics", ">="), ("maximum_metrics", "<=")):
        thresholds = rules.get(group, {})
        if not isinstance(thresholds, dict):
            raise TypeError("Threshold group must be an object")
        for path, threshold in sorted(thresholds.items()):
            if type(threshold) not in (int, float) or not math.isfinite(threshold):
                raise ValueError("Threshold must be finite")
            values = []
            for metrics in (candidate, dev):
                value = metrics
                for part in path.split("."):
                    value = value.get(part) if isinstance(value, dict) else None
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f"Metric must be finite in gate and pilot: {path}")
                values.append(value)
            actual = values[0]
            checks.append(
                {
                    "name": path,
                    "actual": actual,
                    "threshold": threshold,
                    "operator": operator,
                    "passed": actual >= threshold
                    if operator == ">="
                    else actual <= threshold,
                }
            )
    result = {
        "schema_version": "deploy-gate-result-v1",
        "qualification_mode": "initial_baseline",
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "failed_checks": [c for c in checks if not c["passed"]],
        "candidate_model_version": manifest["model_version"],
        "baseline_model_version": None,
        "rules_version": version,
        "rules_sha256": _sha(rules_path),
        "pilot_report_sha256": _sha(pilot_report),
        "candidate_manifest_sha256": _sha(candidate_package / "model_manifest.json"),
        "candidate_metrics_sha256": _sha(gate_metrics),
        "length_contract_version": LENGTH_CONTRACT_VERSION,
        "inference_max_length": 8192,
        "expected_current_generation": expected_current_generation,
        "model_publish_root_uri": root_uri.rstrip("/"),
        "cloud_deployed": False,
        "report_only_domains": report_only,
        "quality_unverified_domains": sorted(report_only),
        "automatic_feedback_submission_authorized": False,
    }
    result.update(promotion_contract(manifest, result))
    import json

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("ml/config"))
    for key in (
        "candidate-package",
        "gate-metrics",
        "development-metrics",
        "gate-dataset",
        "development-dataset",
        "rules-path",
        "pilot-report",
        "output",
    ):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--root-uri", required=True)
    parser.add_argument(
        "--expected-current-generation",
        required=True,
        help="Exact observed generation; 'absent' only when CURRENT does not exist",
    )
    args = vars(parser.parse_args())
    raw = args["expected_current_generation"]
    args["expected_current_generation"] = None if raw == "absent" else int(raw)
    result = qualify_r0_initial(**args)
    print({"passed": result["passed"], "cloud_deployed": False})
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
