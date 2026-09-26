from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from llm_router_labeling.deployment.model_package import assemble_model_package
from llm_router_labeling.deployment.r0_release import (
    prepare_r0_release,
    r0_evaluation_profile,
)
from test_model_promotion import _checkpoint, _metadata


def put(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def release_inputs(tmp_path):
    source = Path(__file__).resolve().parents[1] / "config"
    config = tmp_path / "config"
    config.mkdir()
    for name in (
        "feedback_retraining_continual_v1.json",
        "continual_gate_v1.json",
        "initial_v1_dev_evaluation.json",
        "modernbert_initial_v1_training.json",
    ):
        (config / name).write_bytes((source / name).read_bytes())
    paths = {}
    for name in (
        "base_dataset",
        "base_manifest",
        "gate_dataset",
        "development_dataset",
        "pilot_report",
    ):
        path = tmp_path / name
        path.write_text("synthetic test fixture: " + name)
        paths[name + "_path"] = path
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    feedback = json.loads(
        (config / "feedback_retraining_continual_v1.json").read_text()
    )
    feedback.update(
        base_dataset_sha256=sha(paths["base_dataset_path"]),
        base_manifest_sha256=sha(paths["base_manifest_path"]),
    )
    put(config / "feedback_retraining_continual_v1.json", feedback)
    evaluation = json.loads((config / "continual_gate_v1.json").read_text())
    evaluation["evaluation_dataset_sha256"] = sha(paths["gate_dataset_path"])
    put(config / "continual_gate_v1.json", evaluation)
    development = json.loads((config / "initial_v1_dev_evaluation.json").read_text())
    development["evaluation_dataset_sha256"] = sha(paths["development_dataset_path"])
    put(config / "initial_v1_dev_evaluation.json", development)
    evaluation = r0_evaluation_profile(evaluation)
    development = r0_evaluation_profile(development)
    contract = {
        "evaluation_role": "continual_gate",
        "evaluation_split": "gate",
        "evaluation_version": evaluation["evaluation_version"],
        "dataset_sha256": evaluation["evaluation_dataset_sha256"],
        "label_semantics": evaluation["label_semantics"],
        "label_order": evaluation["label_order"],
        "model_info.max_length": 8192,
        **{
            f"dataset_contract.{k}": v
            for k, v in evaluation["dataset_contract"].items()
        },
    }
    metric_paths = [
        "macro_f1",
        "per_class.hard.recall",
        "per_source_dataset.routerbench.macro_f1",
        "per_source_dataset.xroutebench.macro_f1",
        *{
            f"per_domain.{v}.macro_f1"
            for v in evaluation["category_to_domain"].values()
        },
    ]
    baseline = {}
    for name, value in {**contract, **dict.fromkeys(metric_paths, 0.7)}.items():
        current = baseline
        pieces = name.split(".")
        for part in pieces[:-1]:
            current = current.setdefault(part, {})
        current[pieces[-1]] = value
    rules = {
        "rules_version": "synthetic-r0-pilot-v1",
        "threshold_selection": {
            "evaluation_role": "development_calibration",
            "frozen_before_gate": True,
            "dataset_sha256": development["evaluation_dataset_sha256"],
            "report_sha256": sha(paths["pilot_report_path"]),
        },
        "required_evaluation_contract": contract,
        "minimum_delta": dict.fromkeys(metric_paths, 0.0),
    }
    baseline_metrics = put(tmp_path / "baseline.json", baseline)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _checkpoint(checkpoint)
    model_config = json.loads((checkpoint / "config.json").read_text())
    model_config.update(max_position_embeddings=8192, router_objective="ordinal")
    put(checkpoint / "config.json", model_config)
    put(
        checkpoint / "training_config.json",
        {"max_length": 512, "objective_method": "ordinal"},
    )
    package = tmp_path / "baseline-package"
    manifest = assemble_model_package(
        checkpoint,
        package,
        {
            **_metadata(),
            "model_version": "fixture-r0-1.0",
            "input_contract": "structure-preserving-v2",
            "length_contract_version": "train-inference-lengths-v1",
            "training_max_length": 512,
            "inference_max_length": 8192,
            "tokenizer_max_length": 8192,
            "evaluation_role": "continual_gate",
            "evaluation_version": evaluation["evaluation_version"],
            "dataset_manifest_sha256": evaluation["dataset_manifest_sha256"],
            **evaluation["dataset_contract"],
        },
        baseline_metrics,
    )
    # Synthetic approved fixture; this test does not perform real GPU qualification.
    manifest["gate_status"] = "approved"
    put(package / "model_manifest.json", manifest)
    pilot_metrics = copy.deepcopy(baseline)
    pilot_metrics.update(
        evaluation_role=development["evaluation_role"],
        evaluation_split=development["evaluation_split"],
        dataset_sha256=development["evaluation_dataset_sha256"],
    )
    pilot_metrics_path = put(tmp_path / "pilot-metrics.json", pilot_metrics)
    pilot = {
        "schema_version": "r0-gate-pilot-v1",
        "objective": "ordinal",
        "evaluation_role": "development_calibration",
        "dataset_sha256": development["evaluation_dataset_sha256"],
        "baseline_model_version": "fixture-r0-1.0",
        "baseline_manifest_sha256": sha(package / "model_manifest.json"),
        "pilot_metrics_sha256": sha(pilot_metrics_path),
        "length_contract_version": "train-inference-lengths-v1",
        "inference_max_length": 8192,
        "selected_rules": {
            k: v for k, v in rules.items() if k != "threshold_selection"
        },
    }
    put(paths["pilot_report_path"], pilot)
    rules["threshold_selection"]["report_sha256"] = sha(paths["pilot_report_path"])
    return dict(
        config_dir=config,
        output_dir=tmp_path / "release",
        **paths,
        gate_rules_path=put(tmp_path / "rules.json", rules),
        baseline_metrics_path=baseline_metrics,
        baseline_package_path=package,
        pilot_metrics_path=pilot_metrics_path,
        baseline_model_version="fixture-r0-1.0",
        code_commit="a" * 40,
    )


def test_prepare_preserves_blocked_draft_and_emits_r0_only_files(release_inputs):
    args = release_inputs
    before = (args["config_dir"] / "feedback_retraining_continual_v1.json").read_bytes()
    result = prepare_r0_release(**args)
    assert (
        args["config_dir"] / "feedback_retraining_continual_v1.json"
    ).read_bytes() == before
    output = args["output_dir"]
    feedback = json.loads((output / "feedback.json").read_text())
    assert feedback["deployment_status"] == "approved"
    assert feedback["training_objective"] == "ordinal"
    assert "proxy" in feedback["feedback_target_contract"]
    metadata = json.loads((output / "package-metadata.json").read_text())
    assert metadata["schema_version"] == "model-package-v1"
    assert metadata["tokenizer_max_length"] == 8192
    assert metadata["training_max_length"] == 512
    assert metadata["inference_max_length"] == 8192
    assert "model_version" not in metadata
    assert json.loads((output / "training.json").read_text())["max_length"] == 512
    for filename in ("development.json", "evaluation.json"):
        assert (
            json.loads((output / filename).read_text())["inference"]["max_length"]
            == 8192
        )
    assert (output / "baseline-metrics.json").read_bytes() == args[
        "baseline_metrics_path"
    ].read_bytes()
    assert metadata["input_contract"] == "structure-preserving-v2"
    assert result["cloud_deployed"] is False and result["model_approved"] is False
    assert len(result["files"]) == 7
    with pytest.raises(ValueError, match="already exists"):
        prepare_r0_release(**args)


@pytest.mark.parametrize(
    "fault",
    [
        "smoke",
        "no_pilot",
        "bad_pilot",
        "base_hash",
        "gate_hash",
        "wrong_split",
        "missing_domain",
        "nan",
        "vacuous",
        "baseline",
        "baseline_512",
        "baseline_candidate",
        "baseline_dataset",
        "baseline_r1",
        "pilot_512",
        "pilot_metrics_sha",
        "pilot_metric_nan",
        "development_hash",
        "placeholder_version",
    ],
)
def test_invalid_release_inputs_never_create_output(release_inputs, fault):
    args = release_inputs
    rules = json.loads(args["gate_rules_path"].read_text())
    if fault == "smoke":
        rules["rules_version"] = "deploy-gate-smoke-v1"
    elif fault == "no_pilot":
        rules.pop("threshold_selection")
    elif fault == "bad_pilot":
        args["pilot_report_path"].write_text("wrong")
    elif fault in {"base_hash", "gate_hash"}:
        args[
            "base_dataset_path" if fault == "base_hash" else "gate_dataset_path"
        ].write_text("wrong")
    elif fault == "wrong_split":
        rules["required_evaluation_contract"]["evaluation_split"] = "test"
    elif fault == "missing_domain":
        rules["minimum_delta"].pop("per_domain.math.macro_f1")
    elif fault in {"nan", "vacuous"}:
        rules["minimum_delta"]["macro_f1"] = float("nan") if fault == "nan" else -1.0
    elif fault == "development_hash":
        args["development_dataset_path"].write_text("wrong")
    elif fault == "placeholder_version":
        args["baseline_model_version"] = "replace-approved-version"
    elif fault in {
        "baseline_512",
        "baseline_candidate",
        "baseline_dataset",
        "baseline_r1",
    }:
        path = args["baseline_package_path"] / "model_manifest.json"
        manifest = json.loads(path.read_text())
        if fault == "baseline_512":
            manifest.update(inference_max_length=512, tokenizer_max_length=512)
        elif fault == "baseline_dataset":
            manifest["dataset_version"] = "different-dataset"
        elif fault == "baseline_r1":
            manifest["objective"] = "pointwise"
        else:
            manifest["gate_status"] = "candidate"
        put(path, manifest)
    elif fault in {"pilot_512", "pilot_metrics_sha", "pilot_metric_nan"}:
        path = args["pilot_metrics_path"]
        metrics = json.loads(path.read_text())
        if fault == "pilot_512":
            metrics["model_info"]["max_length"] = 512
        elif fault == "pilot_metric_nan":
            metrics["macro_f1"] = float("nan")
        else:
            metrics["macro_f1"] = 0.8
        put(path, metrics)
    elif fault == "baseline":
        put(
            args["baseline_metrics_path"],
            {"evaluation_role": "development_calibration"},
        )
    put(args["gate_rules_path"], rules)
    with pytest.raises((ValueError, RuntimeError)):
        prepare_r0_release(**args)
    assert not args["output_dir"].exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("objective", "pointwise"),
        ("evaluation_role", "continual_gate"),
        ("baseline_model_version", "other"),
        ("selected_rules", {}),
    ],
)
def test_pilot_contents_are_checked_even_when_checksum_matches(
    release_inputs, field, value
):
    args = release_inputs
    pilot = json.loads(args["pilot_report_path"].read_text())
    pilot[field] = value
    put(args["pilot_report_path"], pilot)
    rules = json.loads(args["gate_rules_path"].read_text())
    rules["threshold_selection"]["report_sha256"] = hashlib.sha256(
        args["pilot_report_path"].read_bytes()
    ).hexdigest()
    put(args["gate_rules_path"], rules)
    with pytest.raises(ValueError, match="Pilot report must bind"):
        prepare_r0_release(**args)
    assert not args["output_dir"].exists()


def test_checked_in_8192_profiles_are_derived_without_changing_frozen_history():
    config = Path(__file__).resolve().parents[1] / "config"
    for source, destination in (
        ("continual_gate_v1.json", "r0_continual_gate_8192.json"),
        ("initial_v1_dev_evaluation.json", "r0_dev_evaluation_8192.json"),
    ):
        original = json.loads((config / source).read_text())
        assert original["inference"]["max_length"] == 512
        assert r0_evaluation_profile(original) == json.loads(
            (config / destination).read_text()
        )
