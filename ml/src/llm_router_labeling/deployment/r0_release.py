"""Prepare a local R0 release from explicit, frozen gate inputs; never publish."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from ..training.routing_policy import LENGTH_CONTRACT_VERSION
from .deploy_gate import evaluate_deploy_gate
from .model_package import validate_model_package


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected an object: {path.name}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def r0_evaluation_profile(config: dict[str, Any]) -> dict[str, Any]:
    """Derive a new inference profile without rewriting frozen 512-token history."""
    result = copy.deepcopy(config)
    result["evaluation_version"] = (
        "routerbench-xroutebench-label-evaluation-r0-inference8192-v1"
    )
    result["inference"].update(max_length=8192, batch_size=1)
    return result


def _evaluation_contract(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "evaluation_role": config["evaluation_role"],
        "evaluation_split": config["evaluation_split"],
        "evaluation_version": config["evaluation_version"],
        "dataset_sha256": config["evaluation_dataset_sha256"],
        "label_semantics": config["label_semantics"],
        "label_order": config["label_order"],
        "model_info.max_length": 8192,
        **{
            f"dataset_contract.{key}": value
            for key, value in config["dataset_contract"].items()
        },
    }


def _check_metrics_contract(metrics: dict[str, Any], config: dict[str, Any]) -> None:
    for key, expected in _evaluation_contract(config).items():
        actual: Any = metrics
        for part in key.split("."):
            actual = actual.get(part) if isinstance(actual, dict) else None
        if actual != expected or (
            key == "model_info.max_length" and type(actual) is not int
        ):
            raise ValueError(f"Evaluation metrics contract mismatch: {key}")


def validate_r0_gate_rules(
    rules: dict[str, Any], evaluation: dict[str, Any], development: dict[str, Any]
) -> None:
    """Require a D_dev-frozen rule set; do not invent/tune thresholds on D_gate."""
    version = rules.get("rules_version", "")
    if (
        not isinstance(version, str)
        or not version.strip()
        or any(word in version.lower() for word in ("smoke", "replace", "placeholder"))
    ):
        raise ValueError("An explicit non-smoke gate rules version is required")
    provenance = rules.get("threshold_selection", {})
    if (
        not isinstance(provenance, dict)
        or provenance.get("evaluation_role") != "development_calibration"
        or provenance.get("dataset_sha256") != development["evaluation_dataset_sha256"]
        or provenance.get("frozen_before_gate") is not True
        or not isinstance(provenance.get("report_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", provenance["report_sha256"]) is None
    ):
        raise ValueError("Gate thresholds require frozen D_dev pilot provenance")
    expected = _evaluation_contract(evaluation)
    contract = rules.get("required_evaluation_contract", {})
    if not isinstance(contract, dict) or any(
        contract.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Gate rules must pin the complete frozen D_gate contract")
    minimum_delta = rules.get("minimum_delta", {})
    required = {
        "macro_f1",
        "per_class.hard.recall",
        "per_source_dataset.routerbench.macro_f1",
        "per_source_dataset.xroutebench.macro_f1",
        *{
            f"per_domain.{name}.macro_f1"
            for name in evaluation["category_to_domain"].values()
        },
    }
    if not isinstance(minimum_delta, dict) or not required.issubset(minimum_delta):
        raise ValueError(
            "Gate must compare macro-F1, hard recall, sources and domains to baseline"
        )
    for group in (
        "minimum_metrics",
        "maximum_metrics",
        "minimum_delta",
        "maximum_delta",
    ):
        values = rules.get(group, {})
        if not isinstance(values, dict):
            raise TypeError(f"{group} must be an object")
        for value in values.values():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Gate thresholds must be finite numbers")
    if any(not -1 < minimum_delta[path] <= 0 for path in required):
        raise ValueError("Baseline regression tolerances must be in (-1, 0]")


def prepare_r0_release(
    *,
    config_dir: Path,
    output_dir: Path,
    gate_rules_path: Path,
    pilot_report_path: Path,
    pilot_metrics_path: Path,
    baseline_package_path: Path,
    development_dataset_path: Path,
    baseline_metrics_path: Path,
    baseline_model_version: str,
    base_dataset_path: Path,
    base_manifest_path: Path,
    gate_dataset_path: Path,
    code_commit: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("Release output already exists; use a new immutable directory")
    if re.fullmatch(r"[0-9a-f]{40}", code_commit) is None:
        raise ValueError("code_commit must be the full Git commit SHA")
    if not baseline_model_version.strip() or any(
        word in baseline_model_version.lower()
        for word in ("replace", "placeholder", "smoke")
    ):
        raise ValueError("An actual baseline model version is required")
    feedback = _read(config_dir / "feedback_retraining_continual_v1.json")
    evaluation = r0_evaluation_profile(_read(config_dir / "continual_gate_v1.json"))
    development = r0_evaluation_profile(
        _read(config_dir / "initial_v1_dev_evaluation.json")
    )
    rules = _read(gate_rules_path)
    validate_r0_gate_rules(rules, evaluation, development)
    if _sha(pilot_report_path) != rules["threshold_selection"]["report_sha256"]:
        raise ValueError("Pilot report SHA mismatch")
    pilot = _read(pilot_report_path)
    package_validation = validate_model_package(baseline_package_path)
    baseline_manifest_path = baseline_package_path / "model_manifest.json"
    baseline_manifest = _read(baseline_manifest_path)
    expected_baseline = {
        "schema_version": "model-package-v1",
        "model_version": baseline_model_version,
        "gate_status": "approved",
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
    if baseline_manifest.get("objective", "ordinal") != "ordinal" or any(
        baseline_manifest.get(key) != value for key, value in expected_baseline.items()
    ):
        raise ValueError(
            "Baseline must be an approved R0 512-training/8192-inference package"
        )
    if baseline_manifest["files"].get("gate_metrics.json") != _sha(
        baseline_metrics_path
    ):
        raise ValueError(
            "Baseline gate metrics SHA does not match the approved package"
        )
    pilot_metrics = _read(pilot_metrics_path)
    _check_metrics_contract(pilot_metrics, development)
    for metric_path in rules["minimum_delta"]:
        actual: Any = pilot_metrics
        for part in metric_path.split("."):
            actual = actual.get(part) if isinstance(actual, dict) else None
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(actual)
        ):
            raise ValueError(f"Pilot metric must be finite: {metric_path}")
    expected_pilot = {
        "schema_version": "r0-gate-pilot-v1",
        "objective": "ordinal",
        "evaluation_role": "development_calibration",
        "dataset_sha256": development["evaluation_dataset_sha256"],
        "baseline_model_version": baseline_model_version,
        "baseline_manifest_sha256": _sha(baseline_manifest_path),
        "pilot_metrics_sha256": _sha(pilot_metrics_path),
        "length_contract_version": LENGTH_CONTRACT_VERSION,
        "inference_max_length": 8192,
        "selected_rules": {
            k: v for k, v in rules.items() if k != "threshold_selection"
        },
    }
    if any(pilot.get(key) != value for key, value in expected_pilot.items()):
        raise ValueError(
            "Pilot report must bind D_dev, R0, baseline and the selected rules"
        )
    for path, expected in (
        (base_dataset_path, feedback["base_dataset_sha256"]),
        (base_manifest_path, feedback["base_manifest_sha256"]),
        (gate_dataset_path, evaluation["evaluation_dataset_sha256"]),
        (development_dataset_path, development["evaluation_dataset_sha256"]),
    ):
        if _sha(path) != expected:
            raise ValueError(f"Frozen artifact SHA mismatch: {path.name}")
    baseline = _read(baseline_metrics_path)
    _check_metrics_contract(baseline, evaluation)
    # Exercise every contract and metric path against the actual same-gate baseline.
    check = evaluate_deploy_gate(
        baseline,
        rules,
        candidate_model_version=baseline_model_version,
        baseline_model_version=baseline_model_version,
        baseline_metrics=baseline,
        package_validation=package_validation,
    )
    # Validation of an existing approved baseline does not approve a new candidate.
    if not check["passed"]:
        raise ValueError(
            "Baseline does not satisfy the frozen gate contract/absolute floors"
        )
    feedback.update(
        deployment_status="approved",
        training_objective="ordinal",
        feedback_target_contract="observed-escalation-final-label-proxy-v1",
        deployment_blockers=[],
        candidate_feedback_schema_note=(
            "R0 only: finalLabel is a legacy escalation proxy; chain completion is not "
            "objective answer correctness or observed per-candidate R1 success."
        ),
    )
    training = _read(config_dir / "modernbert_initial_v1_training.json")
    training["training_version"] = "modernbert-r0-feedback-v1"
    if (
        training["max_length"] != 512
        or training["train_sampling_strategy"] != "shuffle"
    ):
        raise ValueError(
            "R0 profile requires the frozen 512-token shuffle training contract"
        )
    contract = evaluation["dataset_contract"]
    metadata = {
        "schema_version": "model-package-v1",
        # The pipeline supplies the real per-run candidate model_version.
        "base_model": f"{training['model_id']}@{training['model_revision']}",
        "dataset_version": contract["dataset_version"],
        "preprocessing_version": contract["preprocessing_version"],
        "label_rule_version": contract["label_rule_version"],
        "split_version": contract["split_version"],
        "dataset_manifest_sha256": evaluation["dataset_manifest_sha256"],
        "training_version": training["training_version"],
        "evaluation_version": evaluation["evaluation_version"],
        "evaluation_role": "continual_gate",
        "tokenizer_max_length": 8192,
        "length_contract_version": LENGTH_CONTRACT_VERSION,
        "training_max_length": 512,
        "inference_max_length": 8192,
        "objective": "ordinal",
        "input_contract": "structure-preserving-v2",
        "code_commit": code_commit,
    }
    payloads = {
        "feedback.json": feedback,
        "training.json": training,
        "evaluation.json": evaluation,
        "development.json": development,
        "baseline-metrics.json": baseline,
        "gate-rules.json": rules,
        "package-metadata.json": metadata,
    }
    output_dir.mkdir(parents=True)
    files = {}
    for name, payload in payloads.items():
        path = output_dir / name
        if name == "baseline-metrics.json":
            # Gate CLI binds these exact bytes to the approved manifest, not a reserialization.
            path.write_bytes(baseline_metrics_path.read_bytes())
        else:
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
        files[name] = {"sha256": _sha(path), "size_bytes": path.stat().st_size}
    manifest = {
        "schema_version": "r0-release-config-v1",
        "objective": "ordinal",
        "code_commit": code_commit,
        "length_contract_version": LENGTH_CONTRACT_VERSION,
        "training_max_length": 512,
        "inference_max_length": 8192,
        "version_policy_mode": "successfulHttpRequest",
        "files": files,
        "baseline_model_version": baseline_model_version,
        "baseline_metrics_sha256": _sha(baseline_metrics_path),
        "baseline_manifest_sha256": _sha(baseline_manifest_path),
        "pilot_metrics_sha256": _sha(pilot_metrics_path),
        "development_dataset_sha256": _sha(development_dataset_path),
        "pilot_report_sha256": _sha(pilot_report_path),
        "base_dataset_sha256": _sha(base_dataset_path),
        "base_manifest_sha256": _sha(base_manifest_path),
        "gate_dataset_sha256": _sha(gate_dataset_path),
        "cloud_deployed": False,
        "model_approved": False,
        "remaining_release_checks": [
            "Verify active group archive in Drive by remote readback/SHA before switching.",
            "Build and pin all image digests; compile matching managed Vertex IR.",
            "After the single-group switch, upload immutable R0 artifacts and configure URIs.",
            "Verify a cloud run, CURRENT reconciliation, serving activation and rollback.",
        ],
    }
    (output_dir / "release.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("ml/config"))
    for name in (
        "output-dir",
        "gate-rules-path",
        "pilot-report-path",
        "pilot-metrics-path",
        "baseline-package-path",
        "development-dataset-path",
        "baseline-metrics-path",
        "base-dataset-path",
        "base-manifest-path",
        "gate-dataset-path",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("baseline-model-version", "code-commit"):
        parser.add_argument(f"--{name}", required=True)
    result = prepare_r0_release(**vars(parser.parse_args()))
    print(
        json.dumps(
            {"schema_version": result["schema_version"], "cloud_deployed": False}
        )
    )


if __name__ == "__main__":
    main()
