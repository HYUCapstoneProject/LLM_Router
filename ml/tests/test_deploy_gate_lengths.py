"""A package qualified for 8192 cannot be promoted using 512-token scores."""

import pytest
from llm_router_labeling.deployment.deploy_gate import (
    DeployGateError,
    evaluate_deploy_gate,
)
from llm_router_labeling.deployment.model_package import validate_model_package
from llm_router_labeling.training.routing_policy import LENGTH_CONTRACT_VERSION
from test_model_length_contract import assemble


def run_gate(candidate, baseline, package_validation):
    return evaluate_deploy_gate(
        {"macro_f1": 0.99, **candidate},
        {"minimum_metrics": {"macro_f1": 0.8}},
        candidate_model_version=package_validation["model_version"],
        package_validation=package_validation,
        baseline_metrics=baseline,
        baseline_model_version="approved-baseline",
    )


@pytest.mark.parametrize(
    ("candidate_length", "baseline_length", "failed_roles"),
    [
        (512, 512, ["candidate", "baseline"]),
        (512, 8192, ["candidate"]),
        (8192, 512, ["baseline"]),
        (8192, 8192, []),
    ],
)
def test_real_validated_8192_package_requires_both_evaluations_at_8192(
    tmp_path, candidate_length, baseline_length, failed_roles
):
    _, package, _ = assemble(tmp_path, schema="v1")
    validation = validate_model_package(package)
    assert validation["length_contract_version"] == LENGTH_CONTRACT_VERSION
    result = run_gate(
        {"model_info": {"max_length": candidate_length}},
        {"model_info": {"max_length": baseline_length}},
        validation,
    )
    assert result["passed"] is (not failed_roles)
    assert result["length_contract_version"] == LENGTH_CONTRACT_VERSION
    assert result["inference_max_length"] == 8192
    assert [check["name"] for check in result["failed_checks"]] == [
        f"{role}.inference_max_length" for role in failed_roles
    ]
    assert all(check["threshold"] == 8192 for check in result["failed_checks"])


@pytest.mark.parametrize("role", ["candidate", "baseline"])
@pytest.mark.parametrize("length", [None, "8192", 8192.0, True, -1, 0])
def test_evaluation_length_must_be_a_strict_positive_integer(tmp_path, role, length):
    _, package, _ = assemble(tmp_path, schema="v1")
    metrics = {
        "candidate": {"model_info": {"max_length": 8192}},
        "baseline": {"model_info": {"max_length": 8192}},
    }
    metrics[role]["model_info"]["max_length"] = length
    with pytest.raises(DeployGateError, match=f"{role} model_info.max_length"):
        run_gate(metrics["candidate"], metrics["baseline"], validate_model_package(package))


@pytest.mark.parametrize("missing", ["baseline", "candidate_length", "baseline_length"])
def test_new_length_contract_cannot_qualify_without_length_evidence(tmp_path, missing):
    _, package, _ = assemble(tmp_path, schema="v1")
    candidate = {} if missing == "candidate_length" else {"model_info": {"max_length": 8192}}
    baseline = (
        None if missing == "baseline" else
        {} if missing == "baseline_length" else
        {"model_info": {"max_length": 8192}}
    )
    with pytest.raises(DeployGateError):
        run_gate(candidate, baseline, validate_model_package(package))


def test_explicit_512_rollback_package_uses_512_evaluations(tmp_path):
    _, package, _ = assemble(tmp_path, schema="v1", inference_length=512)
    result = run_gate(
        {"model_info": {"max_length": 512}},
        {"model_info": {"max_length": 512}},
        validate_model_package(package),
    )
    assert result["passed"] is True


def test_legacy_package_gate_keeps_existing_metric_contract():
    assert run_gate({}, None, {"valid": True, "model_version": "legacy"})["passed"] is True
