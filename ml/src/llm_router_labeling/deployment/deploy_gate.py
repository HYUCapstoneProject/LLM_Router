from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..training.routing_policy import LENGTH_CONTRACT_VERSION

SCHEMA_VERSION = "deploy-gate-result-v1"


class DeployGateError(RuntimeError):
    """Raised when gate inputs are malformed or a metric cannot be resolved."""


def _contract_value(metrics: Mapping[str, Any], path: str) -> Any:
    current: Any = metrics
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise DeployGateError(f"Evaluation contract path not found: {path}")
        current = current[part]
    return current


def _metric(metrics: Mapping[str, Any], path: str) -> float:
    current: Any = metrics
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise DeployGateError(f"Metric path not found: {path}")
        current = current[part]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise DeployGateError(f"Metric must be numeric: {path}")
    return float(current)


def evaluate_deploy_gate(
    candidate_metrics: Mapping[str, Any],
    rules: Mapping[str, Any],
    *,
    candidate_model_version: str,
    package_validation: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any] | None = None,
    baseline_model_version: str | None = None,
) -> dict[str, Any]:
    if not package_validation.get("valid"):
        raise DeployGateError("Candidate model package did not pass validation")

    packaged_version = str(package_validation.get("model_version", ""))
    if packaged_version != candidate_model_version:
        raise DeployGateError(
            "Package model_version does not match the candidate model version: "
            f"{packaged_version!r} != {candidate_model_version!r}"
        )

    checks: list[dict[str, Any]] = [
        {
            "name": "package_integrity",
            "operator": "valid",
            "actual": True,
            "threshold": True,
            "passed": True,
        },
        {
            "name": "package_model_version",
            "operator": "==",
            "actual": packaged_version,
            "threshold": candidate_model_version,
            "passed": True,
        },
    ]
    if "length_contract_version" in package_validation:
        if package_validation["length_contract_version"] != LENGTH_CONTRACT_VERSION:
            raise DeployGateError("Unsupported package length_contract_version")
        inference_length = package_validation.get("inference_max_length")
        if type(inference_length) is not int or inference_length not in {512, 8192}:
            raise DeployGateError("Package inference_max_length must be an explicit 512 or 8192")
        if baseline_metrics is None:
            raise DeployGateError("Explicit inference length qualification requires baseline metrics")
        for role, metrics in (("candidate", candidate_metrics), ("baseline", baseline_metrics)):
            actual = _contract_value(metrics, "model_info.max_length")
            if type(actual) is not int or actual <= 0:
                raise DeployGateError(f"{role} model_info.max_length must be a positive integer")
            checks.append(
                {
                    "name": f"{role}.inference_max_length",
                    "operator": "==",
                    "actual": actual,
                    "threshold": inference_length,
                    "passed": actual == inference_length,
                }
            )
    expected_contract = rules.get("required_evaluation_contract", {})
    if not isinstance(expected_contract, Mapping):
        raise DeployGateError("required_evaluation_contract must be an object")
    for path, expected in sorted(expected_contract.items()):
        for role, metrics in (
            ("candidate", candidate_metrics),
            ("baseline", baseline_metrics),
        ):
            if metrics is None:
                continue
            actual = _contract_value(metrics, str(path))
            checks.append(
                {
                    "name": f"{role}.evaluation_contract.{path}",
                    "operator": "==",
                    "actual": actual,
                    "threshold": expected,
                    "passed": actual == expected,
                }
            )

    rule_groups = (
        ("minimum_metrics", ">=", lambda actual, threshold: actual >= threshold),
        ("maximum_metrics", "<=", lambda actual, threshold: actual <= threshold),
    )
    for group_name, operator, predicate in rule_groups:
        values = rules.get(group_name, {})
        if not isinstance(values, Mapping):
            raise DeployGateError(f"{group_name} must be an object")
        for path, raw_threshold in sorted(values.items()):
            actual = _metric(candidate_metrics, str(path))
            threshold = float(raw_threshold)
            checks.append(
                {
                    "name": str(path),
                    "operator": operator,
                    "actual": actual,
                    "threshold": threshold,
                    "passed": bool(predicate(actual, threshold)),
                }
            )

    delta_groups = (
        ("minimum_delta", ">=", lambda delta, threshold: delta >= threshold),
        ("maximum_delta", "<=", lambda delta, threshold: delta <= threshold),
    )
    for group_name, operator, predicate in delta_groups:
        values = rules.get(group_name, {})
        if not isinstance(values, Mapping):
            raise DeployGateError(f"{group_name} must be an object")
        if values and baseline_metrics is None:
            raise DeployGateError(f"{group_name} requires baseline metrics")
        for path, raw_threshold in sorted(values.items()):
            candidate = _metric(candidate_metrics, str(path))
            baseline = _metric(baseline_metrics or {}, str(path))
            delta = candidate - baseline
            threshold = float(raw_threshold)
            checks.append(
                {
                    "name": f"{path}.delta",
                    "operator": operator,
                    "actual": delta,
                    "threshold": threshold,
                    "candidate": candidate,
                    "baseline": baseline,
                    "passed": bool(predicate(delta, threshold)),
                }
            )

    failed_checks = [check for check in checks if not check["passed"]]
    return {
        "schema_version": SCHEMA_VERSION,
        **(
            {
                "length_contract_version": package_validation["length_contract_version"],
                "inference_max_length": package_validation["inference_max_length"],
            }
            if "length_contract_version" in package_validation else {}
        ),
        "passed": not failed_checks,
        "candidate_model_version": candidate_model_version,
        "baseline_model_version": baseline_model_version,
        "rules_version": str(rules.get("rules_version", "unspecified")),
        "checks": checks,
        "failed_checks": failed_checks,
    }
