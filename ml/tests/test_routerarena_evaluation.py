from __future__ import annotations

import unittest

import pandas as pd

from llm_router_labeling.evaluation.routerarena_preprocessing import RouterArenaError
from llm_router_labeling.evaluation.routerarena_routing_evaluator import (
    evaluate_routerarena_predictions,
    resolve_model_label_order,
    run_routerarena_routing_evaluation,
)


LABELS = ("easy", "medium", "hard")
ROUTES = {"easy": "haiku", "medium": "sonnet", "hard": "opus"}


def evaluation_frame() -> pd.DataFrame:
    actual = ("easy", "medium", "hard", "hard", "medium", "easy")
    return pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(len(actual))],
            "model_input_text": [
                f"RouterArena question number {index}" for index in range(len(actual))
            ],
            "difficulty_label": actual,
            "evaluation_split": "full",
            "source_sample_id": [f"source-{index}" for index in range(len(actual))],
        }
    )


class FakePredictor:
    def predict(self, texts: list[str]) -> pd.DataFrame:
        labels = ("easy", "easy", "hard", "easy", "hard", "medium")
        return pd.DataFrame(
            {
                "predicted_label": labels[: len(texts)],
                "prediction_confidence": [0.8] * len(texts),
                "prob_easy": [0.6] * len(texts),
                "prob_medium": [0.2] * len(texts),
                "prob_hard": [0.2] * len(texts),
                "input_token_count": [20] * len(texts),
                "was_truncated": [False] * len(texts),
                "inference_latency_ms": [2.0] * len(texts),
            }
        )

    def info(self) -> dict[str, object]:
        return {"model_path": "synthetic-checkpoint", "device": "cpu"}


class RouterArenaRoutingEvaluationTest(unittest.TestCase):
    def test_resolves_explicit_or_generic_checkpoint_labels(self) -> None:
        self.assertEqual(
            resolve_model_label_order(
                {0: "easy", 1: "medium", 2: "hard"}, LABELS, num_labels=3
            ),
            LABELS,
        )
        self.assertEqual(
            resolve_model_label_order(
                {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
                LABELS,
                num_labels=3,
            ),
            LABELS,
        )

    def test_rejects_checkpoint_label_mismatch(self) -> None:
        with self.assertRaises(RouterArenaError):
            resolve_model_label_order(
                {0: "negative", 1: "neutral", 2: "positive"},
                LABELS,
                num_labels=3,
            )

    def test_compares_predictions_and_reports_routing_direction(self) -> None:
        metadata = pd.DataFrame(
            {
                "sample_id": [f"sample-{index}" for index in range(6)],
                "domain": ["science"] * 3 + ["history"] * 3,
                "source_dataset": ["dataset-a"] * 6,
            }
        )
        result = run_routerarena_routing_evaluation(
            evaluation_frame(),
            FakePredictor(),
            label_order=LABELS,
            route_mapping=ROUTES,
            evaluation_version="test-v2",
            label_semantics="routerarena-three-class-alignment",
            metadata=metadata,
        )
        self.assertEqual(result.metrics["row_count"], 6)
        self.assertEqual(
            result.metrics["label_semantics"],
            "routerarena-three-class-alignment",
        )
        self.assertAlmostEqual(result.metrics["accuracy"], 2 / 6)
        self.assertEqual(result.metrics["under_routing_count"], 2)
        self.assertEqual(result.metrics["over_routing_count"], 2)
        self.assertEqual(result.metrics["severe_under_routing_count"], 1)
        self.assertEqual(result.metrics["truncated_count"], 0)
        self.assertEqual(
            result.metrics["confusion_matrix"]["hard"],
            {"easy": 1, "medium": 0, "hard": 1},
        )
        self.assertEqual(
            result.predictions["routing_result"].value_counts().to_dict(),
            {"aligned": 2, "under_routing": 2, "over_routing": 2},
        )
        self.assertIn("routerarena_label", result.predictions)
        self.assertIn("is_label_match", result.predictions)
        self.assertEqual(
            result.metrics["routerarena_label_distribution"],
            {"easy": 2, "medium": 2, "hard": 2},
        )
        self.assertIn("science", result.metrics["per_domain"])
        self.assertEqual(result.metrics["model"]["model_path"], "synthetic-checkpoint")

    def test_rejects_invalid_prediction_label(self) -> None:
        output = pd.DataFrame(
            {"predicted_label": ["unknown"] * len(evaluation_frame())}
        )
        with self.assertRaises(RouterArenaError):
            evaluate_routerarena_predictions(
                evaluation_frame(),
                output,
                label_order=LABELS,
                route_mapping=ROUTES,
                evaluation_version="test-v2",
                label_semantics="routerarena-three-class-alignment",
            )


if __name__ == "__main__":
    unittest.main()
