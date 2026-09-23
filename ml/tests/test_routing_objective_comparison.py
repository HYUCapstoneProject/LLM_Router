from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from llm_router_labeling.objective_comparison.data import (
    pareto_frontier,
    select_three_arms,
)
from llm_router_labeling.objective_comparison.evaluation import (
    evaluate_penalty_curve,
    route_actions,
    summarize_curve,
)
from llm_router_labeling.objective_comparison.objectives import objective_loss
from llm_router_labeling.objective_comparison.selection import build_validation_lock


class ObjectiveDataTest(unittest.TestCase):
    def test_pareto_knee_selects_three_ordered_arms(self) -> None:
        models = ["cheap", "dominated", "knee", "strong"]
        quality = pd.DataFrame(
            {
                "cheap": [0.2, 0.4, 0.3],
                "dominated": [0.1, 0.3, 0.2],
                "knee": [0.7, 0.8, 0.9],
                "strong": [1.0, 0.9, 1.0],
            },
            index=["a", "b", "c"],
        )
        cost = pd.DataFrame(
            {
                "cheap": [1.0, 1.0, 1.0],
                "dominated": [2.0, 2.0, 2.0],
                "knee": [3.0, 3.0, 3.0],
                "strong": [10.0, 10.0, 10.0],
            },
            index=quality.index,
        )
        selected, statistics = select_three_arms(
            quality, cost, quality.index, minimum_frontier_size=3
        )
        self.assertEqual(
            selected, {"low": "cheap", "medium": "knee", "high": "strong"}
        )
        self.assertFalse(
            bool(statistics.set_index("model_name").at["dominated", "on_pareto_frontier"])
        )

    def test_pareto_frontier_removes_more_expensive_weaker_model(self) -> None:
        statistics = pd.DataFrame(
            {
                "model_name": ["a", "b", "c"],
                "mean_quality": [0.5, 0.4, 0.8],
                "mean_cost": [1.0, 2.0, 3.0],
            }
        )
        self.assertEqual(pareto_frontier(statistics)["model_name"].tolist(), ["a", "c"])


class ObjectiveLossTest(unittest.TestCase):
    def test_all_losses_are_finite_and_differentiable(self) -> None:
        import torch

        for method in ("ordinal", "pointwise", "pairwise", "regret"):
            logits = torch.tensor(
                [[0.3, 0.2, 0.1], [0.0, 0.5, 1.0]],
                dtype=torch.float32,
                requires_grad=True,
            )
            loss = objective_loss(
                method,
                logits,
                difficulty_ids=torch.tensor([0, 2]),
                success_targets=torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
                quality_targets=torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.4, 1.0]]),
                decision_costs=torch.tensor([[0.0, 0.3, 1.0], [0.0, 0.3, 1.0]]),
                sample_weights=torch.ones(2),
                regret_lambdas=[0.0, 0.5, 1.0],
                pairwise_tie_weight=0.25,
            )
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())


class ObjectiveEvaluationTest(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "source_dataset": ["routerbench", "routerbench", "xroutebench"],
                "category": ["x", "x", "y"],
                "difficulty_label": ["easy", "medium", "hard"],
                "quality_low": [1.0, 0.0, 0.0],
                "quality_medium": [1.0, 1.0, 0.5],
                "quality_high": [1.0, 1.0, 1.0],
                "actual_cost_low": [1.0, 1.0, 1.0],
                "actual_cost_medium": [3.0, 3.0, 3.0],
                "actual_cost_high": [10.0, 10.0, 10.0],
                "decision_cost_low": [0.0, 0.0, 0.0],
                "decision_cost_medium": [0.3, 0.3, 0.3],
                "decision_cost_high": [1.0, 1.0, 1.0],
            }
        )

    def test_penalty_moves_routes_toward_low_cost(self) -> None:
        scores = np.array([[0.0, 1.0, 2.0]] * 3)
        costs = self._frame()[
            ["decision_cost_low", "decision_cost_medium", "decision_cost_high"]
        ].to_numpy()
        self.assertTrue(np.all(route_actions("pairwise", scores, costs, 0.0) == 2))
        self.assertTrue(np.all(route_actions("pairwise", scores, costs, 100.0) == 0))

    def test_curve_summary_returns_locked_budget_penalties(self) -> None:
        frame = self._frame()
        scores = np.array([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 1.0, 3.0]])
        curve = evaluate_penalty_curve(
            frame, scores, method="pairwise", penalties=[0.0, 1.0, 100.0]
        )
        summary, penalties = summarize_curve(
            frame,
            curve,
            budgets=[0.0, 0.5, 1.0],
            retention_targets=[0.95],
        )
        self.assertIn("quality_cost_nauc", summary)
        self.assertEqual(set(penalties), {"0.0000", "0.5000", "1.0000"})


class ObjectiveSelectionTest(unittest.TestCase):
    def _write_run(self, root: Path, method: str, seed: int, score: float) -> None:
        run = root / method / f"seed-{seed}"
        run.mkdir(parents=True)
        manifest = {
            "schema_version": "routing-objective-run-v1",
            "method": method,
            "seed": seed,
            "maximum_train_rows": 0,
            "maximum_validation_rows": 0,
            "maximum_train_steps": 0,
        }
        metrics = {
            "summary": {"quality_cost_nauc": score},
            "validation_budget_penalties": {"0.5000": 1.0},
        }
        (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        pd.DataFrame(
            {"quality_low": [0.2], "quality_medium": [0.6], "quality_high": [0.9]}
        ).to_parquet(run / "validation-scores.parquet", index=False)
        (run / "_SUCCESS").write_text("ok\n", encoding="utf-8")

    def test_validation_lock_uses_extra_seeds_only_for_top_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "runs"
            config = {
                "experiment_version": "test",
                "seed": 42,
                "finalist_seeds": [17, 73],
                "evaluation": {
                    "selection_metric": "quality_cost_nauc",
                    "default_relative_cost_budget": 0.5,
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            for method, score in zip(
                ("ordinal", "pointwise", "pairwise", "regret"),
                (0.1, 0.2, 0.3, 0.4),
                strict=True,
            ):
                self._write_run(runs, method, 42, score)
            provisional = build_validation_lock(
                run_roots=[runs],
                experiment_config_path=config_path,
                output_path=root / "provisional.json",
                stage="provisional",
            )
            self.assertEqual(provisional["finalists"], ["regret", "pairwise"])
            for method in provisional["finalists"]:
                self._write_run(runs, method, 17, 0.5 if method == "regret" else 0.3)
                self._write_run(runs, method, 73, 0.6 if method == "regret" else 0.3)
            resumed_provisional = build_validation_lock(
                run_roots=[runs],
                experiment_config_path=config_path,
                output_path=root / "provisional.json",
                stage="provisional",
            )
            self.assertEqual(resumed_provisional, provisional)
            final = build_validation_lock(
                run_roots=[runs],
                experiment_config_path=config_path,
                output_path=root / "final.json",
                stage="final",
            )
            self.assertEqual(final["winner_method"], "regret")


if __name__ == "__main__":
    unittest.main()
