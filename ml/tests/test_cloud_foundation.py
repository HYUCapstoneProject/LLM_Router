from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from llm_router_labeling.deployment.cloud_foundation import (
    audit,
    cloud_commands,
    namespace_sha256,
    validate_config,
)


@pytest.fixture
def config():
    value = json.loads(
        (Path(__file__).parents[1] / "config/cloud-foundation-v1.json").read_text()
    )
    # Synthetic inventory tests start from a planned layout; actual receipts are
    # verified separately against the checked-in repository root.
    for profile in value["profiles"].values():
        profile["namespace_status"] = "planned"
        profile.pop("namespace_evidence", None)
    return value


def inventory(config):
    return {
        "apis": [{"config": {"name": name}} for name in config["required_apis"]],
        "vm": {
            "name": config["serving_instance"],
            "status": "RUNNING",
            "serviceAccounts": [
                {
                    "email": config["service_accounts"].get("serving")
                    or "observed-serving@example.iam.gserviceaccount.com"
                }
            ],
            "metadata": {"items": [{"key": "token", "value": "SHOULD_NOT_APPEAR"}]},
        },
        "bucket": {"name": config["bucket"], "location": "US-WEST1"},
        "repository": {"name": config["artifact_repository"], "format": "DOCKER"},
        **{
            f"{role}_account": {
                "email": config["service_accounts"][role],
                "disabled": False,
            }
            for role in ("pipeline", "training")
        },
        **{
            name: {"bindings": [], "version": 3}
            for name in (
                "bucket_iam",
                "project_iam",
                "repository_iam",
                "pipeline_iam",
                "training_iam",
            )
        },
    }


def runner_for(config, values):
    by_command = {tuple(command): name for name, command in cloud_commands(config)}
    return lambda command: values[by_command[tuple(command)]]


def test_proposed_config_is_locally_valid_but_not_release_ready(config):
    report = audit(config)
    assert report["status"] == "local_verified"
    assert report["local_errors"] == []
    assert len(report["release_inputs_pending"]) == 8
    assert report["deployment_ready"] is False
    assert report["cloud_inventory_verified"] is False
    assert report["mutations_performed"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("registry_root_uri", "gs://llm-router-mlserver/models/router-modernbert"),
        ("feedback_root_uri", "gs://llm-router-mlserver/releases/r1/datasets/feedback"),
        ("control_root_uri", "gs://other-bucket/control"),
        ("config_root_uri", "gs://llm-router-mlserver/releases/r0/../config"),
        (
            "current_pointer_uri",
            "gs://llm-router-mlserver/models/router-modernbert/CURRENT.json",
        ),
        ("sqlite_path", "/app/var/r1/retraining.sqlite3"),
        (
            "gate_rules_uri",
            "gs://llm-router-mlserver/releases/r0/config/deploy_gate_rules_smoke.json",
        ),
        (
            "baseline_metrics_uri",
            "gs://llm-router-mlserver/releases/r0/config/baseline_metrics_empty.json",
        ),
        (
            "approved_baseline_uri",
            "gs://llm-router-mlserver/releases/r1/models/router-modernbert/model/approved",
        ),
    ],
)
def test_cross_profile_legacy_or_invalid_paths_block_before_cloud(config, field, value):
    config["profiles"]["r0"][field] = value

    def forbidden(command):
        pytest.fail("Invalid configuration must not even contact cloud APIs")

    report = audit(config, cloud=True, runner=forbidden)
    assert report["status"] == "planned"
    assert report["local_errors"]


def test_mutable_image_or_wrong_registry_is_rejected(config):
    config["images"]["training"] = "us-west1-docker.pkg.dev/project/repo/train:latest"
    assert any("immutable digest" in error for error in validate_config(config)[0])


def test_provisioned_marker_requires_hash_bound_checks_and_rejects_layout_changes(
    config, tmp_path
):
    profile = config["profiles"]["r0"]
    profile["namespace_status"] = "provisioned"
    assert validate_config(config, evidence_root=tmp_path)[0]
    record = {
        "schema_version": "router-namespace-provisioning-v1",
        "profile": "r0",
        "namespace_sha256": namespace_sha256(profile, config["service_accounts"]),
        "observed_at": "2026-09-26T01:00:00+09:00",
        "iam_configuration_verified": True,
        "prefix_access_probe_verified": True,
        "production_pointer_unchanged": True,
    }
    path = tmp_path / "provisioning.json"
    path.write_text(json.dumps(record))
    profile["namespace_evidence"] = {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert validate_config(config, evidence_root=tmp_path)[0] == []
    profile["feedback_root_uri"] += "-changed"
    assert validate_config(config, evidence_root=tmp_path)[0]


def test_provisioning_record_cannot_replace_access_probe_with_inventory(
    config, tmp_path
):
    profile = config["profiles"]["r0"]
    profile["namespace_status"] = "provisioned"
    record = {"status": "cloud_verified", "cloud_inventory_verified": True}
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(record))
    profile["namespace_evidence"] = {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert validate_config(config, evidence_root=tmp_path)[0]


def test_inventory_success_does_not_claim_iam_or_deployment_success(config):
    report = audit(config, cloud=True, runner=runner_for(config, inventory(config)))
    assert report["status"] == "cloud_verified"
    assert report["cloud_inventory_verified"] is True
    assert report["iam_effective_access_verified"] is False
    assert report["deployment_ready"] is False
    assert "SHOULD_NOT_APPEAR" not in json.dumps(report)
    assert report["predeploy_checks_pending"]


@pytest.mark.parametrize(
    "change", ["api", "identity", "disabled", "empty_bucket", "wrong_resource"]
)
def test_partial_or_conflicting_cloud_evidence_never_passes(config, change):
    values = inventory(config)
    if change == "api":
        values["apis"] = []
    elif change == "identity":
        config["service_accounts"]["serving"] = (
            f"pinned-serving@{config['project']}.iam.gserviceaccount.com"
        )
    elif change == "disabled":
        values["training_account"]["disabled"] = True
    elif change == "empty_bucket":
        values["bucket"] = {}
    else:
        values["repository"]["name"] = (
            "projects/another/locations/us-west1/repositories/wrong"
        )
    report = audit(config, cloud=True, runner=runner_for(config, values))
    assert report["status"] == "local_verified"
    assert report["cloud_errors"]
    assert report["cloud_inventory_verified"] is False


def test_command_errors_do_not_leak_stderr_or_partial_secrets(config):
    def fail(command):
        raise ValueError("authorization token SECRET_WOULD_LEAK")

    report = audit(config, cloud=True, runner=fail)
    assert report["cloud_errors"]
    assert "SECRET_WOULD_LEAK" not in json.dumps(report)
    assert report["status"] == "local_verified"


def test_only_fixed_read_only_gcloud_operations_are_constructed(config):
    allowed = {
        ("services", "list"),
        ("compute", "instances", "describe"),
        ("storage", "buckets", "describe"),
        ("storage", "buckets", "get-iam-policy"),
        ("projects", "get-iam-policy"),
        ("artifacts", "repositories", "describe"),
        ("artifacts", "repositories", "get-iam-policy"),
        ("iam", "service-accounts", "describe"),
        ("iam", "service-accounts", "get-iam-policy"),
    }
    for _, command in cloud_commands(config):
        assert command[0] == "gcloud"
        assert any(tuple(command[1 : 1 + len(prefix)]) == prefix for prefix in allowed)
        assert any(arg.startswith("--format=json") for arg in command)
        if command[1:3] == ["compute", "instances"]:
            assert "--format=json(name,status,machineType,serviceAccounts)" in command
        assert not any(
            word in command
            for word in ("create", "update", "delete", "submit", "deploy", "access")
        )


def test_checked_in_provisioning_evidence_remains_valid():
    root = Path(__file__).parents[2]
    value = json.loads((root / "ml/config/cloud-foundation-v1.json").read_text())
    assert validate_config(value, evidence_root=root)[0] == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("service_mode", "parallel"),
        ("http_contract", "v2-required"),
        ("gcs_model_groups", "all-objectives"),
        ("switch_requires_remote_archive_readback", False),
        ("cross_objective_gcs_overlap_allowed", True),
        ("target_profile", "r3"),
    ],
)
def test_single_active_unchanged_api_policy_is_required(config, key, value):
    config["deployment_policy"][key] = value
    assert validate_config(config)[0]


@pytest.mark.parametrize("active,inactive", [("r0", "r1"), ("r1", "r0")])
def test_only_active_profile_release_inputs_are_required(config, active, inactive):
    config["deployment_policy"]["target_profile"] = active
    pending = audit(config)["release_inputs_pending"]
    assert any(item.startswith(f"{active}:") for item in pending)
    assert not any(item.startswith(f"{inactive}:") for item in pending)
