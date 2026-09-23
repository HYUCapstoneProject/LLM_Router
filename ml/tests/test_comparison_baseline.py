from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from llm_router_labeling.comparison.evaluation import (
    ComparisonEvaluationError,
    evaluate_classification_predictions,
)


class ComparisonEvaluationTest(unittest.TestCase):
    def _validation(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sample_id": ["a", "b", "c", "d"],
                "source_dataset": ["routerbench", "routerbench", "xroutebench", "xroutebench"],
                "category": ["knowledge", "math", "mmlu", "mmlu"],
                "difficulty_label": ["easy", "medium", "hard", "hard"],
            }
        )

    def test_evaluation_reports_safety_and_source_metrics(self) -> None:
        probabilities = np.array(
            [
                [0.90, 0.05, 0.05],
                [0.10, 0.80, 0.10],
                [0.80, 0.10, 0.10],
                [0.05, 0.10, 0.85],
            ]
        )
        predictions, metrics = evaluate_classification_predictions(
            self._validation(), probabilities, latency_ms=[1.0, 2.0, 3.0]
        )
        self.assertEqual(predictions["is_hard_to_easy"].sum(), 1)
        self.assertEqual(metrics["overall"]["hard_to_easy_rate"], 0.5)
        self.assertEqual(set(metrics["per_source"]), {"routerbench", "xroutebench"})
        self.assertGreater(metrics["overall"]["q_model"], 0.0)
        self.assertEqual(metrics["batch_1_latency_ms"]["p50"], 2.0)

    def test_probability_contract_is_enforced(self) -> None:
        with self.assertRaises(ComparisonEvaluationError):
            evaluate_classification_predictions(
                self._validation(), np.ones((4, 3), dtype=float)
            )


if __name__ == "__main__":
    unittest.main()
