"""Frozen service-pool qualification for R1; never publish or change CURRENT.

Training may use partial observations. This comparison instead requires all
three candidate outcomes for each gate question, so changing actions cannot
silently change the evaluated question set.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from ..training.routing_policy import ACTION_ORDER, RoutingPolicy, route_scores

SCHEMA = "r1-routing-metrics-v1"


def _sha(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("A pinned SHA-256 is required")
    return value


def _number(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Gate values must be finite numbers")
    return float(value)


def summarize_r1_predictions(
    policy: RoutingPolicy,
    logits: Sequence[Sequence[float]],
    rows: Sequence[Mapping[str, Any]],
    *,
    model_version: str,
    model_weights_sha256: str,
    dataset_sha256: str,
) -> dict[str, Any]:
    if policy.objective != "pointwise" or policy.max_length != 8192:
        raise ValueError("R1 gate requires the frozen pointwise 8192 service policy")
    if not model_version or len(logits) != len(rows) or not rows:
        raise ValueError("Gate predictions must cover every nonempty dataset row")
    _sha(model_weights_sha256)
    _sha(dataset_sha256)
    values: dict[str, list[tuple[float, float]]] = {}
    seen: set[str] = set()
    expected_ids = [candidate.model_id for candidate in policy.candidates]
    for scores, row in zip(logits, rows, strict=True):
        if (
            row.get("evaluation_role") != "continual_gate"
            or row.get("split") != "gate"
            or row.get("candidate_pool_version") != policy.candidate_pool_version
            or row.get("candidate_model_ids") != expected_ids
        ):
            raise ValueError("Gate rows must use the same protected service pool")
        sample = row.get("sample_id")
        domain = row.get("domain")
        if (
            not isinstance(sample, str)
            or not sample
            or sample in seen
            or not isinstance(domain, str)
            or not domain
        ):
            raise ValueError("Gate questions must be unique and have a domain")
        seen.add(sample)
        qualities = row.get("quality")
        if not isinstance(qualities, list) or len(qualities) != 3:
            raise ValueError("All three observed gate outcomes are required")
        quality = [_number(value) for value in qualities]
        if any(not 0 <= value <= 1 for value in quality):
            raise ValueError("Observed gate quality must be in [0,1]")
        decision = route_scores(policy, scores)
        index = ACTION_ORDER.index(decision.action)
        values.setdefault(domain, []).append(
            (quality[index], policy.candidates[index].decision_cost)
        )

    def aggregate(items):
        return {
            "row_count": len(items),
            "mean_quality": sum(q for q, _ in items) / len(items),
            "mean_decision_cost": sum(c for _, c in items) / len(items),
        }

    return {
        "schema_version": SCHEMA,
        "evaluation_role": "continual_gate",
        "split": "gate",
        "dataset_sha256": dataset_sha256,
        "policy_sha256": policy.policy_sha256,
        "candidate_pool_version": policy.candidate_pool_version,
        "candidate_model_ids": expected_ids,
        "cost_unit": policy.cost_unit,
        "model_info": {
            "model_version": model_version,
            "weights_sha256": model_weights_sha256,
            "objective": "pointwise",
            "max_length": 8192,
        },
        "complete_candidate_outcomes": True,
        **aggregate([item for group in values.values() for item in group]),
        "per_domain": {
            domain: aggregate(group) for domain, group in sorted(values.items())
        },
    }


def evaluate_r1_gate(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any], rules: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate predeclared quality/cost limits without tuning against D_gate."""
    if rules.get("schema_version") != "r1-gate-rules-v1":
        raise ValueError("An explicit R1 rules contract is required")
    source = rules.get("threshold_selection", {})
    if (
        not isinstance(source, Mapping)
        or source.get("evaluation_role") != "development_calibration"
        or source.get("frozen_before_gate") is not True
    ):
        raise ValueError(
            "Thresholds must be frozen on development data before gate evaluation"
        )
    _sha(source.get("dataset_sha256"))
    _sha(source.get("report_sha256"))
    pins = rules.get("required_contract", {})
    if not isinstance(pins, Mapping):
        raise TypeError("Rules must pin the complete service gate contract")
    for field in ("dataset_sha256", "policy_sha256"):
        _sha(pins.get(field))
    if source["dataset_sha256"] == pins["dataset_sha256"]:
        raise ValueError("Gate data cannot select its own thresholds")
    required_fields = (
        "dataset_sha256",
        "policy_sha256",
        "candidate_pool_version",
        "candidate_model_ids",
        "cost_unit",
    )
    if set(pins) != set(required_fields):
        raise ValueError("Rules must pin the complete service gate contract")
    domains = rules.get("domains")
    if (
        not isinstance(domains, list)
        or not domains
        or any(not isinstance(domain, str) or not domain for domain in domains)
        or len(set(domains)) != len(domains)
    ):
        raise ValueError("Freeze the protected domains before gate evaluation")
    for metrics in (candidate, baseline):
        if (
            metrics.get("schema_version") != SCHEMA
            or metrics.get("evaluation_role") != "continual_gate"
            or metrics.get("split") != "gate"
            or metrics.get("complete_candidate_outcomes") is not True
        ):
            raise ValueError(
                "Difficulty accuracy/partial feedback cannot be an R1 policy gate"
            )
        if any(metrics.get(field) != pins[field] for field in required_fields):
            raise ValueError("Candidate/baseline service gate contract mismatch")
        model = metrics.get("model_info", {})
        if (
            model.get("objective") != "pointwise"
            or type(model.get("max_length")) is not int
            or model["max_length"] != 8192
        ):
            raise ValueError("Both R1 models must be evaluated at 8192")
        _sha(model.get("weights_sha256"))
        if (
            not isinstance(model.get("model_version"), str)
            or not model["model_version"]
        ):
            raise ValueError("Model identity is required")
        if set(metrics.get("per_domain", {})) != set(domains):
            raise ValueError("Gate domain coverage differs from frozen rules")
        groups = [metrics, *metrics["per_domain"].values()]
        for group in groups:
            if type(group.get("row_count")) is not int or group["row_count"] < 1:
                raise ValueError("Gate row counts must be positive integers")
            if (
                not 0 <= _number(group.get("mean_quality")) <= 1
                or _number(group.get("mean_decision_cost")) < 0
            ):
                raise ValueError("Invalid gate metrics")
        if sum(group["row_count"] for group in groups[1:]) != metrics["row_count"]:
            raise ValueError("Gate domain counts do not sum to total rows")
        for field in ("mean_quality", "mean_decision_cost"):
            weighted = (
                sum(group[field] * group["row_count"] for group in groups[1:])
                / metrics["row_count"]
            )
            if not math.isclose(weighted, metrics[field], rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(
                    "Gate domain metrics do not aggregate to total metrics"
                )
    checks = []
    floor = _number(rules.get("minimum_mean_quality"))
    delta = _number(rules.get("minimum_quality_delta"))
    budget = _number(rules.get("maximum_mean_decision_cost"))
    domain_delta = _number(rules.get("minimum_domain_quality_delta"))
    domain_budgets = rules.get("maximum_domain_decision_cost")
    if not isinstance(domain_budgets, Mapping) or set(domain_budgets) != set(domains):
        raise ValueError("Freeze a decision-cost limit for every protected domain")
    if any(_number(value) < 0 for value in domain_budgets.values()):
        raise ValueError("Domain decision-cost limits must be nonnegative")
    if (
        not 0 <= floor <= 1
        or not -1 < delta <= 1
        or not -1 < domain_delta <= 1
        or budget < 0
    ):
        raise ValueError("Invalid quality/cost tolerances")

    def check(name, actual, limit, predicate):
        value = _number(actual)
        checks.append(
            {
                "name": name,
                "actual": value,
                "threshold": limit,
                "passed": predicate(value, limit),
            }
        )

    for name, cand, base in [
        ("all", candidate, baseline),
        *[
            (domain, candidate["per_domain"][domain], baseline["per_domain"][domain])
            for domain in domains
        ],
    ]:
        if (
            type(cand.get("row_count")) is not int
            or cand["row_count"] < 1
            or cand["row_count"] != base.get("row_count")
        ):
            raise ValueError("Candidate/baseline gate row counts differ or are empty")
        for group in (cand, base):
            if (
                not 0 <= _number(group["mean_quality"]) <= 1
                or _number(group["mean_decision_cost"]) < 0
            ):
                raise ValueError("Invalid gate metrics")
        check(
            name + ".quality_delta",
            cand["mean_quality"] - base["mean_quality"],
            delta if name == "all" else domain_delta,
            lambda a, b: a >= b,
        )
        if name != "all":
            check(
                name + ".mean_decision_cost",
                cand["mean_decision_cost"],
                _number(domain_budgets[name]),
                lambda a, b: a <= b,
            )
    check("mean_quality", candidate["mean_quality"], floor, lambda a, b: a >= b)
    check(
        "mean_decision_cost",
        candidate["mean_decision_cost"],
        budget,
        lambda a, b: a <= b,
    )
    return {
        "schema_version": "r1-qualification-result-v1",
        "objective": "pointwise",
        "candidate_model_version": candidate["model_info"]["model_version"],
        "baseline_model_version": baseline["model_info"]["model_version"],
        "inference_max_length": 8192,
        "policy_sha256": pins["policy_sha256"],
        "dataset_sha256": pins["dataset_sha256"],
        "candidate_weights_sha256": candidate["model_info"]["weights_sha256"],
        "baseline_weights_sha256": baseline["model_info"]["weights_sha256"],
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "automatic_promotion_supported": False,
        "promotion_note": "Qualification only. Release binding, publisher integration and server E2E remain required.",
    }
