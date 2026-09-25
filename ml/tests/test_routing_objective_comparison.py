from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
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
    summarize_source_curves,
)
from llm_router_labeling.objective_comparison.objectives import (
    objective_loss,
    primary_row_losses,
)
from llm_router_labeling.objective_comparison.selection import (
    ObjectiveSelectionError,
    _locked_budget_metrics,
    build_validation_lock,
)
from llm_router_labeling.objective_comparison.shared_training import _verify_imported_r0
from llm_router_labeling.shared.dataset_io import sha256_file
from llm_router_labeling.training.modernbert_checkpointing import (
    ARTIFACT_MANIFEST_NAME,
    seal_artifact_directory,
    training_fingerprint,
)
from llm_router_labeling.training.modernbert_runtime import load_runtime_contract
from llm_router_labeling.training.modernbert_training import load_training_config


class ObjectiveDataTest(unittest.TestCase):
    def test_pareto_knee_selects_three_ordered_arms(self) -> None:
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
        self.assertEqual(selected, {"low": "cheap", "medium": "knee", "high": "strong"})
        self.assertFalse(
            bool(
                statistics.set_index("model_name").at["dominated", "on_pareto_frontier"]
            )
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
    def test_shared_row_losses_match_existing_objectives(self) -> None:
        import torch

        for method in ("ordinal", "pointwise", "pairwise"):
            kwargs = {
                "difficulty_ids": torch.tensor([0, 2]),
                "success_targets": torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
                "quality_targets": torch.tensor([[1.0, 0.2, 1.0], [0.0, 0.4, 1.0]]),
                "pairwise_tie_weight": 0.25,
            }
            logits = torch.tensor(
                [[0.3, 0.2, 0.1], [0.0, 0.5, 1.0]], requires_grad=True
            )
            weights = torch.tensor([1.0, 2.0])
            rows = primary_row_losses(method, logits, **kwargs)
            old = objective_loss(
                method,
                logits,
                **kwargs,
                decision_costs=torch.zeros(2, 3),
                sample_weights=weights,
                regret_lambdas=[0.0],
            )
            self.assertTrue(torch.allclose(old, (rows * weights).sum() / weights.sum()))

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
    def test_locked_source_policies_are_not_reselected_on_test_outcomes(self) -> None:
        frame = self._frame()
        for tier, value in (("low", 0.0), ("medium", 1.0), ("high", 2.0)):
            frame[f"score_{tier}"] = value
        policies = {
            "routerbench": {"0.5000": 100.0},
            "xroutebench": {"0.5000": 0.0},
        }
        metrics, predictions = _locked_budget_metrics(
            frame, method="pairwise", penalties=policies
        )
        self.assertEqual(predictions["selected_tier"].tolist(), ["low", "low", "high"])
        self.assertAlmostEqual(metrics["0.5000"]["source_macro_quality"], 0.75)
        frame["quality_low"], frame["quality_high"] = 1.0, 0.0
        _, changed = _locked_budget_metrics(
            frame, method="pairwise", penalties=policies
        )
        self.assertEqual(
            predictions["selected_tier"].tolist(), changed["selected_tier"].tolist()
        )

    def test_degenerate_cost_span_preserves_raw_curve_but_cannot_select_policy(
        self,
    ) -> None:
        frame = self._frame()
        frame["actual_cost_high"] = frame["actual_cost_low"]
        curve = evaluate_penalty_curve(
            frame, np.zeros((3, 3)), method="pairwise", penalties=[0.0, 100.0]
        )
        summary, penalties = summarize_curve(
            frame, curve, budgets=[0.0, 1.0], retention_targets=[0.95]
        )
        self.assertIsNone(summary["quality_cost_nauc"])
        self.assertEqual(penalties, {})
        self.assertFalse(summary["normalization_valid"])
        self.assertEqual(len(curve), 2)
        self.assertTrue(curve["mean_actual_cost"].notna().all())

    def test_source_macro_does_not_change_when_larger_source_is_repeated(self) -> None:
        frame = self._frame()
        scores = np.array([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 1.0, 3.0]])
        kwargs = {
            "method": "pairwise",
            "penalties": [0.0, 1.0, 100.0],
            "budgets": [0.0, 0.5, 1.0],
            "retention_targets": [0.95],
        }
        summary, penalties, _ = summarize_source_curves(frame, scores, **kwargs)
        expanded = pd.concat([frame, *[frame.iloc[:2]] * 4], ignore_index=True)
        expanded_scores = np.concatenate([scores, *[scores[:2]] * 4])
        repeated, _, _ = summarize_source_curves(expanded, expanded_scores, **kwargs)
        self.assertAlmostEqual(
            summary["source_macro_quality_cost_nauc"],
            repeated["source_macro_quality_cost_nauc"],
        )
        self.assertEqual(set(penalties), {"routerbench", "xroutebench"})

    def test_nauc_is_unavailable_when_quality_anchor_span_is_not_positive(self) -> None:
        frame = self._frame()
        frame["quality_high"] = frame["quality_low"]
        scores = np.zeros((3, 3))
        curve = evaluate_penalty_curve(
            frame, scores, method="pairwise", penalties=[0.0, 100.0]
        )
        summary, _ = summarize_curve(
            frame, curve, budgets=[0.0, 1.0], retention_targets=[0.95]
        )
        self.assertIsNone(summary["quality_cost_nauc"])

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
    def test_strict_lock_checks_frozen_contract_sealed_access_and_model_integrity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "experiment_version": "strict-test",
                "seed": 42,
                "finalist_seeds": [],
                "methods": ["ordinal", "pointwise", "pairwise"],
                "base_training_sha256": "base-sha",
                "selection": {"canonical_only": True, "require_contract_match": True},
                "evaluation": {
                    "selection_metric": "source_macro_quality_cost_nauc",
                    "default_relative_cost_budget": 0.5,
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            for method, value in (
                ("ordinal", 0.3),
                ("pointwise", 0.8),
                ("pairwise", 0.7),
            ):
                self._write_run(root / "runs", method, 42, value)
                run = root / "runs" / method / "seed-42"
                metrics = json.loads((run / "metrics.json").read_text())
                metrics["summary"] = {"source_macro_quality_cost_nauc": value}
                metrics.update(
                    optimizer_steps=2007,
                    training_metrics={"optimization": {"planned_steps": 2007}},
                )
                (run / "metrics.json").write_text(json.dumps(metrics))
                (run / "validation-curve.parquet").write_bytes(b"sealed-curve")
                model = run / "model"
                model.mkdir()
                (model / "model.safetensors").write_bytes(b"sealed-weights")
                manifest = json.loads((run / "manifest.json").read_text())
                manifest.update(
                    experiment_version=config["experiment_version"],
                    experiment_config_sha256=sha256_file(config_path),
                    base_training_sha256="base-sha",
                    outcome_manifest_sha256="outcome-sha",
                    training_data_sha256="combined-data-sha",
                    validation_data_sha256="validation-sha",
                    common_training_config={"seed": 42},
                    runtime_version="pinned-runtime",
                    runtime_contract_sha256="runtime-sha",
                    test_data_accessed=False,
                    routerarena_accessed=False,
                    model_dir=str(model),
                    model_artifacts={
                        "model.safetensors": {
                            "size_bytes": 14,
                            "sha256": sha256_file(model / "model.safetensors"),
                        }
                    },
                    artifacts={
                        name: {"sha256": sha256_file(run / name)}
                        for name in [
                            "metrics.json",
                            "validation-scores.parquet",
                            "validation-curve.parquet",
                        ]
                    },
                )
                (run / "manifest.json").write_text(json.dumps(manifest))
                (run / "_SUCCESS").write_text(sha256_file(run / "manifest.json"))
            kwargs = {
                "run_roots": [root / "runs"],
                "experiment_config_path": config_path,
                "output_path": root / "lock.json",
                "stage": "final",
            }
            self.assertEqual(
                build_validation_lock(**kwargs)["winner_method"], "pointwise"
            )
            run = root / "runs" / "pointwise" / "seed-42"
            manifest = json.loads((run / "manifest.json").read_text())
            manifest["test_data_accessed"] = True
            (run / "manifest.json").write_text(json.dumps(manifest))
            (run / "_SUCCESS").write_text(sha256_file(run / "manifest.json"))
            with self.assertRaisesRegex(ObjectiveSelectionError, "sealed evaluation"):
                build_validation_lock(**kwargs)
            manifest["test_data_accessed"] = False
            (run / "manifest.json").write_text(json.dumps(manifest))
            (run / "_SUCCESS").write_text(sha256_file(run / "manifest.json"))
            (run / "model/model.safetensors").write_bytes(b"changed-weights")
            with self.assertRaisesRegex(
                ObjectiveSelectionError, "Model artifact checksum"
            ):
                build_validation_lock(**kwargs)

    def test_primary_three_methods_can_lock_without_optional_regret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "experiment_version": "test",
                "seed": 42,
                "finalist_seeds": [],
                "methods": ["ordinal", "pointwise", "pairwise"],
                "evaluation": {
                    "selection_metric": "quality_cost_nauc",
                    "default_relative_cost_budget": 0.5,
                },
            }
            path = root / "config.json"
            path.write_text(json.dumps(config))
            for method, value in (("ordinal", 0.3), ("pointwise", 0.8)):
                self._write_run(root / "runs", method, 42, value)
            with self.assertRaisesRegex(ObjectiveSelectionError, "Missing canonical"):
                build_validation_lock(
                    run_roots=[root / "runs"],
                    experiment_config_path=path,
                    output_path=root / "lock.json",
                    stage="final",
                )
            self._write_run(root / "runs", "pairwise", 42, 0.7)
            lock = build_validation_lock(
                run_roots=[root / "runs"],
                experiment_config_path=path,
                output_path=root / "lock.json",
                stage="final",
            )
            self.assertEqual(lock["winner_method"], "pointwise")
            self.assertEqual(set(lock["methods"]), {"ordinal", "pointwise", "pairwise"})
            self.assertFalse(lock["production_activation_authorized"])

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


class ImportedR0ContractTest(unittest.TestCase):
    def test_import_requires_exact_full_budget_and_runtime_fingerprint(self) -> None:
        config_root = Path(__file__).resolve().parents[1] / "config"
        config = load_training_config(
            config_root / "modernbert_initial_v1_training.json"
        )
        runtime = load_runtime_contract(
            config_root / "modernbert_runtime_cu124_v1.json"
        )
        fingerprint = training_fingerprint(
            training_data_sha256="frozen-data-sha",
            config=asdict(config),
            maximum_steps=None,
            runtime_version=runtime.runtime_version,
        )
        for mutation in (
            None,
            "short_budget",
            "fingerprint",
            "runtime",
            "wrong_targets",
        ):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                metrics = {
                    "data": {
                        "training_data_sha256": "frozen-data-sha",
                        "train_row_count": 21406,
                        "validation_row_count": 3581,
                    },
                    "optimization": {
                        "optimizer_steps": 2007,
                        "planned_steps": 2007,
                        "max_train_steps": None,
                        "training_fingerprint": fingerprint,
                    },
                    "runtime": {
                        "runtime_version": runtime.runtime_version,
                        "python_minor": runtime.allowed_python_minors[0],
                        "torch_base_version": runtime.torch_version,
                        "transformers_version": runtime.transformers_version,
                        "numpy_version": runtime.numpy_version,
                        "pandas_version": runtime.pandas_version,
                        "pyarrow_version": runtime.pyarrow_version,
                        "cuda_version": runtime.cuda_version,
                        "cuda_available": True,
                    },
                }
                model_config = {
                    "id2label": {"0": "easy", "1": "medium", "2": "hard"},
                    "label2id": {"easy": 0, "medium": 1, "hard": 2},
                }
                if mutation == "short_budget":
                    metrics["optimization"].update(optimizer_steps=40, planned_steps=40)
                elif mutation == "fingerprint":
                    metrics["optimization"]["training_fingerprint"] = "another-run"
                elif mutation == "runtime":
                    metrics["runtime"]["torch_base_version"] = "different-version"
                elif mutation == "wrong_targets":
                    model_config["router_objective"] = "pointwise"
                (root / "config.json").write_text(json.dumps(model_config))
                (root / "training_config.json").write_text(json.dumps(asdict(config)))
                (root / "metrics.json").write_text(json.dumps(metrics))
                (root / "tokenizer_config.json").write_text("{}")
                (root / "model.safetensors").write_bytes(b"verified-placeholder")
                seal_artifact_directory(
                    root,
                    manifest_name=ARTIFACT_MANIFEST_NAME,
                    metadata={"training_fingerprint": fingerprint},
                )
                kwargs = {
                    "training_config": config,
                    "runtime_contract": runtime,
                    "base_training_sha256": "frozen-data-sha",
                    "expected_train_rows": 21406,
                    "expected_validation_rows": 3581,
                }
                if mutation is None:
                    self.assertEqual(
                        _verify_imported_r0(root, **kwargs)["optimization"][
                            "optimizer_steps"
                        ],
                        2007,
                    )
                else:
                    with self.assertRaises((ValueError, RuntimeError)):
                        _verify_imported_r0(root, **kwargs)


if __name__ == "__main__":
    unittest.main()
