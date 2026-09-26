from __future__ import annotations

import json

import pytest
from llm_router_labeling.deployment.model_publish import (
    ModelPublishError,
    approve_candidate,
    publish_candidate,
)
from llm_router_labeling.deployment.r0_initial import qualify_r0_initial
from llm_router_labeling.deployment.r0_release import _read, _sha
from llm_router_labeling.shared.object_store import LocalObjectStore
from test_r0_release import release_inputs  # noqa: F401


@pytest.fixture
def initial_inputs(release_inputs):  # noqa: F811
    old = release_inputs
    package = old["baseline_package_path"]
    manifest_path = package / "model_manifest.json"
    manifest = _read(manifest_path)
    manifest["gate_status"] = "candidate"
    manifest_path.write_text(json.dumps(manifest))
    rules = _read(old["gate_rules_path"])
    rules["schema_version"] = "r0-initial-rules-v1"
    rules["minimum_metrics"] = dict.fromkeys(rules.pop("minimum_delta"), 0.5)
    pilot = {
        "schema_version": "r0-initial-pilot-v1",
        "objective": "ordinal",
        "evaluation_role": "development_calibration",
        "dataset_sha256": _sha(old["development_dataset_path"]),
        "pilot_metrics_sha256": _sha(old["pilot_metrics_path"]),
        "candidate_model_version": manifest["model_version"],
        "candidate_weights_sha256": manifest["files"]["model.safetensors"],
        "length_contract_version": "train-inference-lengths-v1",
        "inference_max_length": 8192,
        "selected_rules": {
            k: v for k, v in rules.items() if k != "threshold_selection"
        },
    }
    old["pilot_report_path"].write_text(json.dumps(pilot))
    rules["threshold_selection"]["report_sha256"] = _sha(old["pilot_report_path"])
    old["gate_rules_path"].write_text(json.dumps(rules))
    return {
        "config_dir": old["config_dir"],
        "candidate_package": package,
        "gate_metrics": old["baseline_metrics_path"],
        "development_metrics": old["pilot_metrics_path"],
        "gate_dataset": old["gate_dataset_path"],
        "development_dataset": old["development_dataset_path"],
        "rules_path": old["gate_rules_path"],
        "pilot_report": old["pilot_report_path"],
        "root_uri": "gs://fixture/releases/r0/models/router-modernbert",
        "expected_current_generation": None,
        "output": package.parent / "initial-gate.json",
    }


def test_initial_qualification_and_absent_current_approval(initial_inputs, tmp_path):
    result = qualify_r0_initial(**initial_inputs)
    assert result["passed"] is True
    assert result["baseline_model_version"] is None
    assert result["qualification_mode"] == "initial_baseline"
    store = LocalObjectStore(tmp_path / "store")
    version = result["candidate_model_version"]
    root = initial_inputs["root_uri"]
    publish_candidate(
        initial_inputs["candidate_package"],
        root_uri=root,
        model_version=version,
        object_store=store,
        gate_result=result,
    )
    approved = approve_candidate(
        root_uri=root,
        model_version=version,
        object_store=store,
        approved_by="test",
        work_dir=tmp_path / "approval",
        training_mode="initial_training",
        initial_qualification=result,
    )
    assert approved["model_version"] == version
    assert approved["previous_pointer_generation"] is None
    with pytest.raises(ValueError, match="immutable"):
        qualify_r0_initial(**initial_inputs)


@pytest.mark.parametrize(
    "failure",
    [
        "changed_current",
        "wrong_mode",
        "wrong_root",
        "missing_generation",
        "boolean_generation",
        "wrong_manifest",
    ],
)
def test_initial_approval_rejects_stale_or_unbound(initial_inputs, tmp_path, failure):
    result = qualify_r0_initial(**initial_inputs)
    store = LocalObjectStore(tmp_path / "store")
    root = initial_inputs["root_uri"]
    version = result["candidate_model_version"]
    publish_candidate(
        initial_inputs["candidate_package"],
        root_uri=root,
        model_version=version,
        object_store=store,
        gate_result=result,
    )
    if failure == "changed_current":
        store.write_text(
            root + "/CURRENT.json",
            json.dumps(
                {"schema_version": "router-current-model-v1", "model_version": "other"}
            ),
        )
    elif failure == "wrong_root":
        result["model_publish_root_uri"] += "-other"
    elif failure == "missing_generation":
        result.pop("expected_current_generation")
    elif failure == "boolean_generation":
        result["expected_current_generation"] = True
    elif failure == "wrong_manifest":
        result["candidate_manifest_sha256"] = "0" * 64
    with pytest.raises(ModelPublishError):
        approve_candidate(
            root_uri=root,
            model_version=version,
            object_store=store,
            approved_by="test",
            work_dir=tmp_path / "approval",
            training_mode="minor_fine_tune"
            if failure == "wrong_mode"
            else "initial_training",
            initial_qualification=result,
        )


@pytest.mark.parametrize(
    "failure",
    [
        "512",
        "unbound_metrics",
        "unbound_pilot",
        "self_delta",
        "no_floor",
        "zero_floor",
        "nan_metric",
        "changed_data",
    ],
)
def test_initial_qualification_rejects_invalid_evidence(initial_inputs, failure):
    args = initial_inputs
    if failure == "changed_data":
        args["gate_dataset"].write_text("changed")
    elif failure in ("self_delta", "no_floor", "zero_floor"):
        rules = _read(args["rules_path"])
        if failure == "self_delta":
            rules["minimum_delta"] = {"macro_f1": 0}
        if failure == "no_floor":
            rules["minimum_metrics"].pop("macro_f1")
        if failure == "zero_floor":
            rules["minimum_metrics"]["macro_f1"] = 0
        args["rules_path"].write_text(json.dumps(rules))
    elif failure in ("512", "unbound_pilot", "nan_metric"):
        metrics = _read(args["development_metrics"])
        if failure == "512":
            metrics["model_info"]["max_length"] = 512
        if failure == "unbound_pilot":
            metrics["macro_f1"] = 0.8
        if failure == "nan_metric":
            metrics["macro_f1"] = float("nan")
        args["development_metrics"].write_text(json.dumps(metrics))
    else:
        args["gate_metrics"].write_text("{}")
    with pytest.raises(ValueError):
        qualify_r0_initial(**args)
    assert not args["output"].exists()


def test_initial_cli_preserves_generation_guard(initial_inputs, tmp_path, monkeypatch):
    from llm_router_labeling.deployment.model_publish_cli import main

    result = qualify_r0_initial(**initial_inputs)
    version = result["candidate_model_version"]
    root = initial_inputs["root_uri"]
    store_path = tmp_path / "store"
    published_path = tmp_path / "published.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "publish",
            "--local-store-root",
            str(store_path),
            "publish-candidate",
            "--package-dir",
            str(initial_inputs["candidate_package"]),
            "--root-uri",
            root,
            "--model-version",
            version,
            "--gate-result",
            str(initial_inputs["output"]),
            "--output",
            str(published_path),
        ],
    )
    main()
    monkeypatch.setattr(
        "sys.argv",
        [
            "publish",
            "--local-store-root",
            str(store_path),
            "approve",
            "--root-uri",
            root,
            "--model-version",
            version,
            "--approved-by",
            "test",
            "--publish-result",
            str(published_path),
            "--training-mode",
            "minor_fine_tune",
        ],
    )
    with pytest.raises(SystemExit, match="cannot authorize feedback"):
        main()
    monkeypatch.setattr(
        "sys.argv",
        [
            "publish",
            "--local-store-root",
            str(store_path),
            "approve",
            "--root-uri",
            root,
            "--model-version",
            version,
            "--approved-by",
            "test",
            "--publish-result",
            str(published_path),
            "--training-mode",
            "initial_training",
        ],
    )
    main()
    pointer = _read(
        store_path / "fixture" / "releases/r0/models/router-modernbert/CURRENT.json"
    )
    assert pointer["model_version"] == version


def test_below_floor_returns_failed_gate_without_approval(initial_inputs):
    # Consistently bind a measured-but-insufficient metric to the package.
    path = initial_inputs["gate_metrics"]
    metrics = _read(path)
    metrics["macro_f1"] = 0.1
    path.write_text(json.dumps(metrics))
    package = initial_inputs["candidate_package"]
    (package / "gate_metrics.json").write_bytes(path.read_bytes())
    manifest = _read(package / "model_manifest.json")
    manifest["files"]["gate_metrics.json"] = _sha(path)
    (package / "model_manifest.json").write_text(json.dumps(manifest))
    result = qualify_r0_initial(**initial_inputs)
    assert result["passed"] is False
    assert result["failed_checks"][0]["name"] == "macro_f1"
    assert result["cloud_deployed"] is False


@pytest.fixture
def small_domain_inputs(initial_inputs):
    import pandas as pd

    args = initial_inputs
    config_path = args["config_dir"] / "initial_v1_dev_evaluation.json"
    config = _read(config_path)
    categories = {}
    for key, domain in config["category_to_domain"].items():
        categories.setdefault(domain, key)
    rows = []
    for domain, key in categories.items():
        source, category = key.split(":")
        for i in range(8 if domain == "conversation" else 31):
            rows.append(
                {
                    "sample_id": f"{domain}-{i}",
                    "model_input_text": f"Question {domain}-{i}",
                    "difficulty_label": ["easy", "medium", "hard"][i % 3],
                    "duplicate_group_id": f"{domain}-{i}",
                    "source_dataset": source,
                    "category": category,
                    "split": "validation",
                    **config["dataset_contract"],
                }
            )
    pd.DataFrame(rows).to_parquet(args["development_dataset"], index=False)
    dev_sha = _sha(args["development_dataset"])
    config["evaluation_dataset_sha256"] = dev_sha
    config_path.write_text(json.dumps(config))
    for key in ("development_metrics", "gate_metrics"):
        metrics = _read(args[key])
        metrics["per_domain"]["conversation"]["row_count"] = 8
        if key == "development_metrics":
            metrics["dataset_sha256"] = dev_sha
        args[key].write_text(json.dumps(metrics))
    package = args["candidate_package"]
    (package / "gate_metrics.json").write_bytes(args["gate_metrics"].read_bytes())
    manifest = _read(package / "model_manifest.json")
    manifest["files"]["gate_metrics.json"] = _sha(args["gate_metrics"])
    (package / "model_manifest.json").write_text(json.dumps(manifest))
    rules = _read(args["rules_path"])
    rules["minimum_metrics"].pop("per_domain.conversation.macro_f1")
    rules["small_domain_policy"] = {
        "policy_version": "initial-small-domain-report-only-v1",
        "minimum_development_rows": 30,
        "frozen_before_gate": True,
        "scope": "initial_baseline_only",
    }
    rules["report_only_domains"] = {
        "conversation": {
            "development_row_count": 8,
            "reason": "insufficient_support",
            "quality_unverified": True,
        }
    }
    pilot = _read(args["pilot_report"])
    pilot["dataset_sha256"] = dev_sha
    pilot["pilot_metrics_sha256"] = _sha(args["development_metrics"])
    pilot["selected_rules"] = {
        k: v for k, v in rules.items() if k != "threshold_selection"
    }
    args["pilot_report"].write_text(json.dumps(pilot))
    rules["threshold_selection"]["dataset_sha256"] = dev_sha
    rules["threshold_selection"]["report_sha256"] = _sha(args["pilot_report"])
    args["rules_path"].write_text(json.dumps(rules))
    return args


def test_small_domain_is_explicitly_unqualified(small_domain_inputs):
    gate = qualify_r0_initial(**small_domain_inputs)
    assert gate["passed"] is True
    assert gate["quality_unverified_domains"] == ["conversation"]
    assert gate["automatic_feedback_submission_authorized"] is False
    assert all(c["name"] != "per_domain.conversation.macro_f1" for c in gate["checks"])


@pytest.mark.parametrize(
    "failure", ["wrong_count", "large_domain", "numeric_floor", "changed_policy"]
)
def test_small_domain_exception_cannot_bypass_evidence(small_domain_inputs, failure):
    args = small_domain_inputs
    rules = _read(args["rules_path"])
    if failure == "wrong_count":
        rules["report_only_domains"]["conversation"]["development_row_count"] = 7
    elif failure == "large_domain":
        rules["report_only_domains"]["code"] = {
            "development_row_count": 31,
            "reason": "insufficient_support",
            "quality_unverified": True,
        }
    elif failure == "numeric_floor":
        rules["minimum_metrics"]["per_domain.conversation.macro_f1"] = 0
    else:
        rules["small_domain_policy"]["minimum_development_rows"] = 10000
    args["rules_path"].write_text(json.dumps(rules))
    with pytest.raises(ValueError):
        qualify_r0_initial(**args)
