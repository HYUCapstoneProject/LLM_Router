from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from llm_router_labeling.objective_comparison.evaluation import route_actions
from llm_router_labeling.training.routing_policy import (
    ACTION_ORDER,
    OBJECTIVE_SCORE_SEMANTICS,
    RoutingPolicy,
    RoutingPolicyError,
    compute_policy_sha256,
    load_routing_policy,
    route_scores,
)


def fixture_policy_payload(
    objective: str = "pointwise", lambda_cost: float = 0.25
) -> dict:
    """Synthetic unit-test values; these are not a chosen service configuration."""
    payload = {
        "schema_version": "routing-policy-v2",
        "scope": "service-explicit",
        "policy_version": "fixture-policy-v2",
        "objective": objective,
        "input_contract": "structure-preserving-v2",
        "max_length": 512,
        "action_order": ["low", "medium", "high"],
        "score_semantics": OBJECTIVE_SCORE_SEMANTICS[objective],
        "lambda_cost": lambda_cost,
        "candidate_pool": {
            "version": "fixture-service-pool-v1",
            "price_version": "fixture-prices-v1",
            "decision_cost_version": "fixture-pre-request-costs-v1",
            "decision_cost_basis": "pre-request-fixed",
            "cost_unit": "normalized-0-1",
            "candidates": [
                {
                    "action": "low",
                    "tier_projection": "easy",
                    "model": "haiku",
                    "model_id": "fixture-provider/small-model-v1",
                    "decision_cost": 0.0,
                },
                {
                    "action": "medium",
                    "tier_projection": "medium",
                    "model": "sonnet",
                    "model_id": "fixture-provider/mid-model-v1",
                    "decision_cost": 0.3,
                },
                {
                    "action": "high",
                    "tier_projection": "hard",
                    "model": "opus",
                    "model_id": "fixture-provider/large-model-v1",
                    "decision_cost": 1.0,
                },
            ],
        },
    }
    payload["policy_sha256"] = compute_policy_sha256(payload)
    return payload


class RoutingPolicyTest(unittest.TestCase):
    def test_online_actions_match_frozen_offline_rule_for_all_objectives(self) -> None:
        random = np.random.default_rng(42)
        logits = np.vstack(
            [
                random.normal(size=(300, 3)) * 8,
                [0.0, 0.0, 0.0],
                [1000.0, -1000.0, 0.0],
                [-1000.0, 1000.0, 0.0],
                [-1000.0, 0.0, 1000.0],
                [0.0, 0.3, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        costs = np.tile([0.0, 0.3, 1.0], (len(logits), 1))
        for objective in OBJECTIVE_SCORE_SEMANTICS:
            for penalty in (0.0, 0.25, 1.0, 3.0):
                with self.subTest(objective=objective, penalty=penalty):
                    policy = RoutingPolicy.from_mapping(
                        fixture_policy_payload(objective, penalty)
                    )
                    expected = route_actions(objective, logits, costs, penalty)
                    actual = [
                        ACTION_ORDER.index(route_scores(policy, row.tolist()).action)
                        for row in logits
                    ]
                    np.testing.assert_array_equal(actual, expected)

    def test_tier_projection_and_model_mapping_remain_three_way(self) -> None:
        for objective in OBJECTIVE_SCORE_SEMANTICS:
            policy = RoutingPolicy.from_mapping(fixture_policy_payload(objective, 0.0))
            for index, (tier, model) in enumerate(
                zip(
                    ("easy", "medium", "hard"), ("haiku", "sonnet", "opus"), strict=True
                )
            ):
                scores = [0.0, 0.0, 0.0]
                scores[index] = 2.0
                decision = route_scores(policy, scores)
                self.assertEqual(decision.action, ACTION_ORDER[index])
                self.assertEqual(decision.tier_projection, tier)
                self.assertEqual(decision.model, model)
                self.assertEqual(decision.model_id, policy.candidates[index].model_id)
                self.assertEqual(
                    decision.candidate_pool_version, "fixture-service-pool-v1"
                )
            expected_labels = (
                ("easy", "medium", "hard") if objective == "ordinal" else ACTION_ORDER
            )
            self.assertEqual(policy.output_label_order, expected_labels)

    def test_confidence_uses_selected_action_score_and_pairwise_has_no_probability(
        self,
    ) -> None:
        for objective in OBJECTIVE_SCORE_SEMANTICS:
            policy = RoutingPolicy.from_mapping(fixture_policy_payload(objective, 4.0))
            decision = route_scores(policy, [0.0, 0.0, 4.0])
            self.assertEqual(decision.action, "low")
            self.assertLess(
                decision.action_scores["low"], decision.action_scores["high"]
            )
            self.assertEqual(
                decision.score_semantics, OBJECTIVE_SCORE_SEMANTICS[objective]
            )
            if objective == "pairwise":
                self.assertIsNone(decision.confidence)
                self.assertEqual(decision.action_scores["high"], 4.0)
            else:
                self.assertEqual(decision.confidence, decision.action_scores["low"])
            if objective == "ordinal":
                self.assertAlmostEqual(sum(decision.action_scores.values()), 1.0)
            elif objective == "pointwise":
                self.assertGreater(sum(decision.action_scores.values()), 1.0)

    def test_exact_ties_use_fixed_action_order_even_after_cost_adjustment(self) -> None:
        policy = RoutingPolicy.from_mapping(fixture_policy_payload("pairwise", 1.0))
        decision = route_scores(policy, [0.0, 0.3, 1.0])
        self.assertEqual(set(decision.utilities.values()), {0.0})
        self.assertEqual(decision.action, "low")
        decision = route_scores(policy, [-1.0, 0.3, 1.0])
        self.assertEqual(decision.action, "medium")

    def test_source_local_policy_cannot_be_implicitly_mapped_to_service(self) -> None:
        for mutation in ("scope", "source_penalties", "decision_cost_basis"):
            payload = fixture_policy_payload()
            if mutation == "scope":
                payload["scope"] = "source-local"
            elif mutation == "source_penalties":
                payload["source_penalties"] = {"routerbench": 0.1, "xroutebench": 3.0}
            else:
                payload["candidate_pool"]["decision_cost_basis"] = (
                    "observed-output-tokens"
                )
            with self.subTest(mutation=mutation), self.assertRaises(RoutingPolicyError):
                RoutingPolicy.from_mapping(payload)

    def test_missing_explicit_policy_pool_cost_and_model_fields_are_rejected(
        self,
    ) -> None:
        original = fixture_policy_payload()
        cases = (
            [((), key) for key in original]
            + [(("candidate_pool",), key) for key in original["candidate_pool"]]
            + [
                (("candidate_pool", "candidates", 0), key)
                for key in original["candidate_pool"]["candidates"][0]
            ]
        )
        for path, key in cases:
            payload = copy.deepcopy(original)
            target = payload
            for part in path:
                target = target[part]
            del target[key]
            with (
                self.subTest(path=path, missing=key),
                self.assertRaises(RoutingPolicyError),
            ):
                RoutingPolicy.from_mapping(payload)

    def test_wrong_shape_meaning_model_mapping_and_values_are_rejected(self) -> None:
        changes = [
            (("objective",), "regret"),
            (("max_length",), 8192),
            (("max_length",), 512.0),
            (("input_contract",), "legacy-word-only-v1"),
            (("action_order",), ["high", "medium", "low"]),
            (("score_semantics",), "difficulty_class_probabilities"),
            (("lambda_cost",), -0.1),
            (("lambda_cost",), float("nan")),
            (("lambda_cost",), float("inf")),
            (("lambda_cost",), 10**1000),
            (("lambda_cost",), True),
            (("policy_version",), ""),
            (("candidate_pool", "price_version"), ""),
            (("candidate_pool", "decision_cost_version"), ""),
            (("candidate_pool", "cost_unit"), "unknown"),
            (("candidate_pool", "cost_unit"), []),
            (("candidate_pool", "candidates", 0, "model_id"), "haiku"),
            (
                ("candidate_pool", "candidates", 0, "model_id"),
                "fixture-provider/mid-model-v1",
            ),
            (("candidate_pool", "candidates", 0, "model"), "opus"),
            (("candidate_pool", "candidates", 0, "tier_projection"), "hard"),
            (("candidate_pool", "candidates", 0, "decision_cost"), 0.1),
            (("candidate_pool", "candidates", 1, "decision_cost"), 2.0),
            (("candidate_pool", "candidates", 1, "decision_cost"), -1.0),
            (("candidate_pool", "candidates", 1, "decision_cost"), float("inf")),
        ]
        for path, value in changes:
            payload = fixture_policy_payload()
            target = payload
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = value
            with (
                self.subTest(path=path, value=value),
                self.assertRaises(RoutingPolicyError),
            ):
                RoutingPolicy.from_mapping(payload)

    def test_internal_policy_hash_and_package_file_hash_are_separate(self) -> None:
        payload = fixture_policy_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routing_policy.json"
            path.write_text(json.dumps(payload, indent=2) + "\n")
            file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertNotEqual(file_sha, payload["policy_sha256"])
            policy = load_routing_policy(path, expected_file_sha256=file_sha)
            self.assertEqual(policy.policy_sha256, payload["policy_sha256"])
            self.assertEqual(
                RoutingPolicy.from_mapping(
                    payload, expected_policy_sha256=policy.policy_sha256
                ),
                policy,
            )
            with self.assertRaisesRegex(RoutingPolicyError, "file SHA-256"):
                load_routing_policy(path, expected_file_sha256=policy.policy_sha256)
            with self.assertRaisesRegex(RoutingPolicyError, "canonical policy digest"):
                RoutingPolicy.from_mapping(payload, expected_policy_sha256=file_sha)
            payload["lambda_cost"] = 0.5
            with self.assertRaisesRegex(RoutingPolicyError, "content does not match"):
                RoutingPolicy.from_mapping(payload)

    def test_hash_is_order_independent_but_covers_service_identity_and_prices(
        self,
    ) -> None:
        payload = fixture_policy_payload()
        original = payload["policy_sha256"]
        self.assertEqual(
            compute_policy_sha256(dict(reversed(list(payload.items())))), original
        )
        payload["candidate_pool"]["price_version"] = "fixture-prices-v2"
        self.assertNotEqual(compute_policy_sha256(payload), original)
        payload = fixture_policy_payload()
        payload["candidate_pool"]["candidates"][0]["model_id"] = (
            "fixture-provider/other-model-v1"
        )
        self.assertNotEqual(compute_policy_sha256(payload), original)

    def test_invalid_score_vector_and_overflow_are_rejected(self) -> None:
        policy = RoutingPolicy.from_mapping(fixture_policy_payload())
        for scores in (
            None,
            1,
            "123",
            [1, 2],
            [0, 1, 2, 3],
            [0, True, 1],
            [0, float("nan"), 1],
            [0, float("inf"), 1],
        ):
            with self.subTest(scores=scores), self.assertRaises(RoutingPolicyError):
                route_scores(policy, scores)
        payload = fixture_policy_payload("pairwise", 1e308)
        payload["candidate_pool"]["cost_unit"] = "usd-per-request"
        payload["candidate_pool"]["candidates"][2]["decision_cost"] = 1e308
        payload["policy_sha256"] = compute_policy_sha256(payload)
        with self.assertRaisesRegex(RoutingPolicyError, "overflowed"):
            route_scores(RoutingPolicy.from_mapping(payload), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
