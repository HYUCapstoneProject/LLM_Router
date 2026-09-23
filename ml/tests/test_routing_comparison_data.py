from __future__ import annotations

import unittest

import pandas as pd

from llm_router_labeling.routing_data.preprocessing import (
    RoutingDataError,
    _assign_xroute_validation,
    _labels_from_rates,
    _pivot_xroute,
    _validate_xroute_rows,
)


class RoutingComparisonDataTest(unittest.TestCase):
    def _rows(self) -> pd.DataFrame:
        rows = []
        for embedding_id, metric, performances in (
            (10, "em_mc", (1.0, 0.0, 1.0)),
            (11, "f1_score", (0.80, 0.69, None)),
        ):
            for model_name, performance in zip(("a", "b", "c"), performances, strict=True):
                rows.append(
                    {
                        "task_name": "task",
                        "query": f"[System Instruction]\nignore\n\n[User Query]\nquestion {embedding_id}",
                        "metric": metric,
                        "task_id": None,
                        "embedding_id": embedding_id,
                        "model_name": model_name,
                        "performance": performance,
                    }
                )
        return pd.DataFrame(rows)

    def test_xroute_pivot_uses_embedding_fallback_and_metric_contract(self) -> None:
        validated = _validate_xroute_rows(
            self._rows(),
            split="train",
            candidates=("a", "b", "c"),
            exact_metrics={"em_mc"},
            continuous_metrics={"f1_score"},
        )
        pivoted, accepted, metrics = _pivot_xroute(
            validated,
            source_split="train",
            candidates=("a", "b", "c"),
            minimum_valid=2,
        )
        self.assertEqual(metrics["query_rows"], 2)
        self.assertEqual(metrics["eligible_query_rows"], 2)
        self.assertEqual(pivoted["accepted_model_count"].tolist(), [2, 1])
        self.assertEqual(pivoted["valid_model_count"].tolist(), [3, 2])
        self.assertEqual(pivoted["model_input_text"].tolist(), ["question 10", "question 11"])
        self.assertEqual(list(accepted.columns), ["a", "b", "c"])

    def test_duplicate_query_model_run_is_rejected(self) -> None:
        frame = pd.concat([self._rows(), self._rows().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(RoutingDataError, "duplicate query/model"):
            _validate_xroute_rows(
                frame,
                split="train",
                candidates=("a", "b", "c"),
                exact_metrics={"em_mc"},
                continuous_metrics={"f1_score"},
            )

    def test_threshold_labels_are_ordinal(self) -> None:
        labels = _labels_from_rates(
            pd.Series([0.8, 0.4, 0.1]).to_numpy(), hard=0.27, easy=0.64
        )
        self.assertEqual(labels.tolist(), ["easy", "medium", "hard"])

    def test_validation_split_is_stratified_at_every_threshold_boundary(self) -> None:
        rates = [0.10] * 300 + [0.30] * 33 + [0.80] * 300
        frame = pd.DataFrame(
            {
                "sample_id": [f"sample-{index}" for index in range(len(rates))],
                "duplicate_group_id": [f"group-{index}" for index in range(len(rates))],
                "category": "task",
                "source_split": "train",
                "overall_solve_rate": rates,
                "split": "train",
            }
        )
        _assign_xroute_validation(
            frame,
            fraction=0.1,
            seed=42,
            score_boundaries=[0.18, 0.27, 0.36, 0.55, 0.64, 0.73],
        )
        train_labels = _labels_from_rates(
            frame.loc[frame["split"].eq("train"), "overall_solve_rate"].to_numpy(),
            hard=0.27,
            easy=0.55,
        )
        validation_labels = _labels_from_rates(
            frame.loc[frame["split"].eq("validation"), "overall_solve_rate"].to_numpy(),
            hard=0.27,
            easy=0.55,
        )
        self.assertEqual(pd.Series(train_labels).value_counts()["medium"], 30)
        self.assertEqual(pd.Series(validation_labels).value_counts()["medium"], 3)


if __name__ == "__main__":
    unittest.main()
