from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from llm_router_labeling.evaluation.r1_gate import (
    evaluate_r1_gate,
    summarize_r1_predictions,
)
from llm_router_labeling.training.routing_policy import RoutingPolicy


def comparison():
    payload = json.loads(
        (
            Path(__file__).parent / "fixtures/r1-observations/routing_policy.json"
        ).read_text()
    )
    policy = RoutingPolicy.from_mapping(payload)
    rows = [
        {
            "sample_id": str(index),
            "domain": domain,
            "split": "gate",
            "evaluation_role": "continual_gate",
            "candidate_pool_version": policy.candidate_pool_version,
            "candidate_model_ids": [c.model_id for c in policy.candidates],
            "quality": [0.8, 0.9, 1.0],
        }
        for index, domain in enumerate(("coding", "knowledge"))
    ]
    common = {"policy": policy, "rows": rows, "dataset_sha256": "d" * 64}
    baseline = summarize_r1_predictions(
        **common,
        logits=[[9, -9, -9]] * 2,
        model_version="r1-1.0",
        model_weights_sha256="b" * 64,
    )
    candidate = summarize_r1_predictions(
        **common,
        logits=[[-9, 9, -9]] * 2,
        model_version="r1-1.1",
        model_weights_sha256="c" * 64,
    )
    rules = {
        "schema_version": "r1-gate-rules-v1",
        "threshold_selection": {
            "evaluation_role": "development_calibration",
            "frozen_before_gate": True,
            "dataset_sha256": "e" * 64,
            "report_sha256": "f" * 64,
        },
        "required_contract": {
            field: candidate[field]
            for field in (
                "dataset_sha256",
                "policy_sha256",
                "candidate_pool_version",
                "candidate_model_ids",
                "cost_unit",
            )
        },
        "domains": ["coding", "knowledge"],
        "minimum_mean_quality": 0.75,
        "minimum_quality_delta": -0.01,
        "minimum_domain_quality_delta": -0.01,
        "maximum_mean_decision_cost": 0.4,
        "maximum_domain_decision_cost": {"coding": 0.4, "knowledge": 0.4},
    }
    return policy, rows, candidate, baseline, rules


def test_same_frozen_policy_quality_and_cost_qualification_does_not_authorize_promotion():
    _, _, candidate, baseline, rules = comparison()
    result = evaluate_r1_gate(candidate, baseline, rules)
    assert result["passed"] is True
    assert result["automatic_promotion_supported"] is False
    assert result["schema_version"] == "r1-qualification-result-v1"
    assert candidate["mean_quality"] == 0.9
    assert candidate["mean_decision_cost"] == 0.3


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_mean_decision_cost", 0.2),
        ("minimum_mean_quality", 0.95),
        ("minimum_quality_delta", 0.15),
        ("minimum_domain_quality_delta", 0.15),
    ],
)
def test_quality_or_cost_failures_cannot_qualify(field, value):
    _, _, candidate, baseline, rules = comparison()
    rules[field] = value
    result = evaluate_r1_gate(candidate, baseline, rules)
    assert result["passed"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.update(dataset_sha256="0" * 64),
        lambda m: m.update(policy_sha256="0" * 64),
        lambda m: m.update(complete_candidate_outcomes=False),
        lambda m: m["model_info"].update(max_length=512),
        lambda m: m["model_info"].update(objective="ordinal"),
        lambda m: m["per_domain"]["coding"].update(row_count=True),
        lambda m: m["per_domain"]["coding"].update(mean_quality=0.5),
        lambda m: m["per_domain"].pop("coding"),
    ],
)
def test_mismatched_or_inconsistent_baseline_is_rejected(mutation):
    _, _, candidate, baseline, rules = comparison()
    mutation(baseline)
    with pytest.raises(ValueError):
        evaluate_r1_gate(candidate, baseline, rules)


@pytest.mark.parametrize(
    "change",
    [
        {"frozen_before_gate": False},
        {"evaluation_role": "continual_gate"},
        {"dataset_sha256": "d" * 64},
    ],
)
def test_thresholds_cannot_be_tuned_using_gate(change):
    _, _, candidate, baseline, rules = comparison()
    rules["threshold_selection"].update(change)
    with pytest.raises(ValueError):
        evaluate_r1_gate(candidate, baseline, rules)


@pytest.mark.parametrize(
    "change",
    [
        {"quality": [1, None, 0]},
        {"quality": [1, True, 0]},
        {"candidate_model_ids": ["benchmark-low", "benchmark-mid", "benchmark-high"]},
        {"split": "train"},
        {"evaluation_role": "development_calibration"},
    ],
)
def test_partial_or_wrong_pool_training_rows_cannot_be_gate_data(change):
    policy, rows, *_ = comparison()
    rows = copy.deepcopy(rows)
    rows[0].update(change)
    with pytest.raises(ValueError):
        summarize_r1_predictions(
            policy,
            [[9, -9, -9]] * 2,
            rows,
            model_version="r1-1.1",
            model_weights_sha256="c" * 64,
            dataset_sha256="d" * 64,
        )


def test_global_budget_pass_cannot_hide_domain_budget_failure():
    _, _, candidate, baseline, rules = comparison()
    rules["maximum_domain_decision_cost"]["coding"] = 0.2
    result = evaluate_r1_gate(candidate, baseline, rules)
    assert result["passed"] is False
    assert any(
        check["name"] == "coding.mean_decision_cost" and not check["passed"]
        for check in result["checks"]
    )
