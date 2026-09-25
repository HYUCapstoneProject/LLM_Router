from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from llm_router_labeling.objective_comparison.sealed_protocol import (
    METHODS,
    TEST_PARTITIONS,
    SealedProtocolError,
    _actions_frame,
    _join_scores,
    _matched_counts,
    _matched_random,
    _oracle_actions,
    _policy_predictions,
    _summary,
    create_sealed_protocol,
    evaluate_sealed_scores,
    infer_sealed_scores,
    paired_group_bootstrap,
)
from llm_router_labeling.shared.dataset_io import (
    DatasetIOError,
    sha256_file,
    write_json,
)

TIERS = ("low", "medium", "high")


def record(path):
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


class SealedRoutingProtocolTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.train = self.root / "train-outcomes"
        self.projection = self.root / "projection"
        self.train.mkdir()
        self.projection.mkdir()
        self.protocol_path = self.root / "protocol.json"
        self.config_path = self.root / "config.json"
        self.lock_path = self.root / "validation-lock.json"
        self.arms = {
            "selection_split": "train_only",
            "sources": {
                source: {
                    "training_row_count": 3,
                    "selected_models": {tier: f"{source}-{tier}" for tier in TIERS},
                }
                for source in ("routerbench", "xroutebench")
            },
        }
        self.config = {
            "methods": list(METHODS),
            "seed": 42,
            "base_dataset_manifest_sha256": "base-pin",
            "evaluation": {
                "selection_metric": "source_macro_quality_cost_nauc",
                "relative_cost_budgets": [0.0, 0.5],
                "default_relative_cost_budget": 0.5,
            },
        }
        write_json(self.config_path, self.config)
        self.validation = pd.concat(
            [self._rows(source, 3, "validation") for source in self.arms["sources"]],
            ignore_index=True,
        )
        self.validation.to_parquet(self.train / "validation.parquet", index=False)
        pd.concat(
            [self._rows(source, 3, "train") for source in self.arms["sources"]],
            ignore_index=True,
        ).to_parquet(self.train / "train.parquet", index=False)
        write_json(self.train / "arms.json", self.arms)
        pd.DataFrame(
            [
                {
                    "source_dataset": source,
                    "model_name": f"{source}-{tier}",
                    "mean_quality": quality,
                    "mean_cost": cost,
                    "selected_tier": tier,
                    "valid_rows": 3,
                }
                for source in self.arms["sources"]
                for tier, quality, cost in zip(
                    TIERS, [0.3, 0.9, 0.8], [0.1, 0.3, 0.9], strict=True
                )
            ]
        ).to_parquet(self.train / "candidate-statistics.parquet", index=False)
        train_manifest = {
            "base_dataset_manifest_sha256": "base-pin",
            "arms": self.arms,
            "artifacts": {
                name: record(self.train / name)
                for name in (
                    "train.parquet",
                    "validation.parquet",
                    "arms.json",
                    "candidate-statistics.parquet",
                )
            },
        }
        write_json(self.train / "manifest.json", train_manifest)
        (self.train / "_SUCCESS").write_text(sha256_file(self.train / "manifest.json"))
        self.test_rows = (
            pd.concat(
                [
                    self._rows("routerbench", 4, "test_routerbench"),
                    self._rows("xroutebench", 2, "test_xroutebench"),
                ],
                ignore_index=True,
            )
            .sort_values("sample_id")
            .reset_index(drop=True)
        )
        for partition in TEST_PARTITIONS:
            self.test_rows.loc[
                self.test_rows["objective_split"].eq(partition)
            ].to_parquet(self.projection / f"{partition}.parquet", index=False)
        projection_manifest = {
            "train_outcome_manifest_sha256": sha256_file(self.train / "manifest.json"),
            "objective_config_sha256": sha256_file(self.config_path),
            "base_dataset_manifest_sha256": "base-pin",
            "arms_sha256": record(self.train / "arms.json")["sha256"],
            "candidate_statistics_sha256": record(
                self.train / "candidate-statistics.parquet"
            )["sha256"],
            "coverage_status": "complete",
            "dropped_row_count": 0,
            "row_counts": {
                partition: int(self.test_rows["objective_split"].eq(partition).sum())
                for partition in TEST_PARTITIONS
            },
            "artifacts": {
                f"{partition}.parquet": record(self.projection / f"{partition}.parquet")
                for partition in TEST_PARTITIONS
            },
        }
        write_json(self.projection / "manifest.json", projection_manifest)
        (self.projection / "_SUCCESS").write_text(
            sha256_file(self.projection / "manifest.json")
        )
        self.policies = {
            source: {"0.0000": 100.0, "0.5000": 0.0} for source in self.arms["sources"]
        }
        self.run_dirs = {}
        methods = {}
        for index, method in enumerate(METHODS):
            directory = self.root / "runs" / method
            directory.mkdir(parents=True)
            self.run_dirs[method] = directory
            model = directory / "model"
            model.mkdir()
            write_json(model / "config.json", {"router_objective": method})
            write_json(model / "tokenizer.json", {})
            write_json(model / "tokenizer_config.json", {})
            (model / "model.safetensors").write_bytes(b"synthetic; inference is mocked")
            scored = self.validation.copy()
            for tier_index, tier in enumerate(TIERS):
                scored[f"score_{tier}"] = [
                    5.0
                    if (i % 3 == 2 and tier_index == 2)
                    or (i % 3 != 2 and tier_index == index)
                    else 0.0
                    for i in range(len(scored))
                ]
            scored.to_parquet(directory / "validation-scores.parquet", index=False)
            metrics = {
                "summary": {"source_macro_quality_cost_nauc": 0.2 + index / 10},
                "validation_budget_penalties": self.policies,
            }
            write_json(directory / "metrics.json", metrics)
            manifest = {
                "method": method,
                "seed": 42,
                "model_dir": str(model),
                "experiment_config_sha256": sha256_file(self.config_path),
                "outcome_manifest_sha256": sha256_file(self.train / "manifest.json"),
                "validation_data_sha256": record(self.train / "validation.parquet")[
                    "sha256"
                ],
                "test_data_accessed": False,
                "routerarena_accessed": False,
                "artifacts": {
                    name: record(directory / name)
                    for name in ("metrics.json", "validation-scores.parquet")
                },
                "model_artifacts": {
                    path.name: record(path) for path in model.iterdir()
                },
            }
            write_json(directory / "manifest.json", manifest)
            (directory / "_SUCCESS").write_text(
                sha256_file(directory / "manifest.json")
            )
            methods[method] = {
                "canonical_run_dir": str(directory),
                "canonical_quality_cost_nauc": metrics["summary"][
                    "source_macro_quality_cost_nauc"
                ],
                "locked_budget_penalties": self.policies,
            }
        write_json(
            self.lock_path,
            {
                "stage": "final",
                "selection_data": "validation_only",
                "experiment_config_sha256": sha256_file(self.config_path),
                "methods": methods,
                "winner_method": "pairwise",
                "canonical_seed": 42,
                "seed_scope": "canonical_only",
            },
        )

    @staticmethod
    def _rows(source, count, partition):
        records = []
        for index in range(count):
            row = {
                "sample_id": f"{source}:{partition}:{index}",
                "source_dataset": source,
                "model_input_text": f"Question {source} {partition} {index}",
                "difficulty_label": "medium",
                "duplicate_group_id": f"{source}:{partition}:group-{index // 3}",
                "objective_split": partition,
                "split": partition,
                "category": "math",
            }
            for tier, quality, cost, decision in zip(
                TIERS, [0.2, 0.3, 0.95], [0.1, 0.3, 0.9], [0.0, 0.25, 1.0], strict=True
            ):
                row.update(
                    {
                        f"model_{tier}": f"{source}-{tier}",
                        f"quality_{tier}": quality,
                        f"success_{tier}": float(quality > 0.7),
                        f"actual_cost_{tier}": cost,
                        f"decision_cost_{tier}": decision,
                    }
                )
            records.append(row)
        return pd.DataFrame(records)

    def _lock(self):
        return create_sealed_protocol(
            validation_lock_path=self.lock_path,
            config_path=self.config_path,
            train_outcome_dir=self.train,
            projection_dir=self.projection,
            output_path=self.protocol_path,
        )

    def _fake_infer(self, *, model_dir, data_path, output_path, **kwargs):
        del kwargs
        frame = pd.read_parquet(data_path)
        method = model_dir.parent.name
        selected = METHODS.index(method)
        for index, tier in enumerate(TIERS):
            frame[f"score_{tier}"] = 5.0 if index == selected else 0.0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)
        return {"row_count": len(frame)}

    def _infer(self):
        with patch(
            "llm_router_labeling.objective_comparison.training.infer_scores",
            side_effect=self._fake_infer,
        ):
            return infer_sealed_scores(
                protocol_path=self.protocol_path, output_dir=self.root / "inference"
            )

    def test_lock_reads_no_heldout_rows_and_uses_train_best_and_source_usage(self):
        real_read = pd.read_parquet

        def guarded_read(path, *args, **kwargs):
            self.assertFalse(Path(path).resolve().is_relative_to(self.projection))
            return real_read(path, *args, **kwargs)

        with patch(
            "llm_router_labeling.objective_comparison.sealed_protocol.pd.read_parquet",
            side_effect=guarded_read,
        ):
            protocol = self._lock()
        self.assertTrue(
            all(
                item["tier"] == "medium"
                for item in protocol["train_best_single"].values()
            )
        )
        self.assertAlmostEqual(
            protocol["methods"]["ordinal"]["validation_usage_ratios"]["routerbench"][
                "0.5000"
            ]["low"],
            2 / 3,
        )
        self.assertEqual(protocol["bootstrap"]["iterations"], 10000)
        self.assertEqual(len(protocol["matched_random"]["seeds"]), 100)
        self.assertFalse(protocol["production_activation_authorized"])

    def test_end_to_end_synthetic_scores_keep_policy_budget_and_uncertainty_roles(self):
        self._lock()
        self._infer()
        with patch(
            "llm_router_labeling.objective_comparison.training.infer_scores",
            side_effect=AssertionError("Evaluation must not run inference"),
        ):
            result = evaluate_sealed_scores(
                protocol_path=self.protocol_path,
                inference_dir=self.root / "inference",
                output_dir=self.root / "evaluation",
            )
        self.assertEqual(result["winner_method_unchanged"], "pairwise")
        self.assertEqual(result["row_count"], 6)
        self.assertTrue(
            result["methods"]["pairwise"]["0.5000"]["per_source"]["routerbench"][
                "budget_exceeded"
            ]
        )
        self.assertEqual(
            result["baselines"]["train_best_single"]["source_macro_mean_quality"], 0.3
        )
        self.assertEqual(
            result["paired_group_bootstrap_vs_ordinal"]["iterations"], 10000
        )
        self.assertFalse(
            result["matched_random"]["distribution_is_confidence_interval"]
        )
        self.assertTrue(
            all(
                item["seed_count"] == 100
                for item in result["matched_random"]["summaries"]
            )
        )
        self.assertEqual(
            result["test_policy"], "raw_frozen_policy_results_no_test_frontier_or_nauc"
        )
        predictions = pd.read_parquet(self.root / "evaluation/predictions.parquet")
        self.assertIn("duplicate_group_id", predictions)
        self.assertIn("objective_split", predictions)

    def test_model_or_outcome_tampering_stops_before_inference(self):
        self._lock()
        for path in (
            self.run_dirs["pairwise"] / "model/model.safetensors",
            self.projection / "test_routerbench.parquet",
        ):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with patch(
                    "llm_router_labeling.objective_comparison.training.infer_scores"
                ) as infer:
                    with self.assertRaises(DatasetIOError):
                        infer_sealed_scores(
                            protocol_path=self.protocol_path,
                            output_dir=self.root / "inference",
                        )
                    infer.assert_not_called()
                self.assertFalse((self.root / "inference").exists())
                path.write_bytes(original)

    def test_heldout_quality_does_not_change_frozen_actions(self):
        rows = self.test_rows
        scored = rows.copy()
        for index, tier in enumerate(TIERS):
            scored[f"score_{tier}"] = float(index)
        first = _policy_predictions(rows, scored, "pairwise", self.policies, "0.5000")
        changed = rows.copy()
        changed["quality_low"], changed["quality_high"] = 1.0, 0.0
        changed["actual_cost_low"], changed["actual_cost_high"] = 50.0, 0.001
        second = _policy_predictions(
            changed, scored, "pairwise", self.policies, "0.5000"
        )
        self.assertEqual(
            first["selected_model_id"].tolist(), second["selected_model_id"].tolist()
        )
        self.assertEqual(first["penalty"].tolist(), second["penalty"].tolist())

    def test_missing_duplicate_and_misbound_score_ids_are_rejected(self):
        scored = self.test_rows.copy()
        for tier in TIERS:
            scored[f"score_{tier}"] = 0.0
        with self.assertRaisesRegex(SealedProtocolError, "different sample IDs"):
            _join_scores(self.test_rows, scored.iloc[:-1])
        with self.assertRaisesRegex(SealedProtocolError, "unique"):
            _join_scores(self.test_rows, pd.concat([scored, scored.iloc[:1]]))
        changed = scored.copy()
        changed.loc[0, "duplicate_group_id"] = "incorrect-group"
        with self.assertRaisesRegex(SealedProtocolError, "metadata differs"):
            _join_scores(self.test_rows, changed)
        shuffled = _join_scores(self.test_rows, scored.iloc[::-1])
        self.assertEqual(
            shuffled["sample_id"].tolist(), self.test_rows["sample_id"].tolist()
        )

    def test_protocol_edits_and_incomplete_inference_cannot_be_evaluated(self):
        self._lock()
        original = self.protocol_path.read_bytes()
        self.protocol_path.write_bytes(original + b" ")
        with patch(
            "llm_router_labeling.objective_comparison.training.infer_scores"
        ) as infer:
            with self.assertRaisesRegex(SealedProtocolError, "protocol checksum"):
                infer_sealed_scores(
                    protocol_path=self.protocol_path, output_dir=self.root / "inference"
                )
            infer.assert_not_called()
        self.protocol_path.write_bytes(original)
        with (
            patch(
                "llm_router_labeling.objective_comparison.training.infer_scores",
                return_value={"row_count": 0},
            ),
            self.assertRaisesRegex(SealedProtocolError, "row count"),
        ):
            infer_sealed_scores(
                protocol_path=self.protocol_path, output_dir=self.root / "inference"
            )
        self.assertFalse((self.root / "inference/_SUCCESS").exists())

    def test_degenerate_normalization_and_oracle_cost_id_ties(self):
        rows = self.test_rows.iloc[:2].copy().reset_index(drop=True)
        for tier in TIERS:
            rows[f"quality_{tier}"] = 1.0
            rows[f"actual_cost_{tier}"] = 1.0
        rows["model_low"], rows["model_medium"], rows["model_high"] = "z", "b", "a"
        rows.loc[1, "actual_cost_low"] = 0.5
        self.assertEqual(_oracle_actions(rows).tolist(), [2, 0])
        constant = rows.iloc[:1].copy()
        summary = _summary(
            constant, _actions_frame(constant, np.array([2])), budget=0.5
        )["per_source"]["routerbench"]
        self.assertIsNone(summary["relative_cost"])
        self.assertIsNone(summary["normalized_quality_gain_unclipped"])
        self.assertIsNone(summary["budget_exceeded"])
        self.assertEqual(summary["mean_quality"], 1.0)
        rows["actual_cost_high"] = 0.0
        summary = _summary(rows, _actions_frame(rows, np.array([2, 2])), budget=0.5)[
            "per_source"
        ]["routerbench"]
        self.assertIsNone(summary["relative_cost"])

    def test_matched_counts_preserve_ratio_and_reproducible_rounding(self):
        self.assertEqual(
            _matched_counts([0.6, 0.3, 0.1], 10, ["a", "b", "c"]).tolist(), [6, 3, 1]
        )
        self.assertEqual(
            _matched_counts([0.5, 0.5, 0], 3, ["z", "a", "b"]).tolist(), [1, 2, 0]
        )
        with self.assertRaisesRegex(SealedProtocolError, "sum to one"):
            _matched_counts([0.4, 0.4, 0.4], 10, ["a", "b", "c"])

    def test_matched_random_macro_uses_source_relative_cost_under_imbalance(self):
        protocol = self._lock()
        rows = self.test_rows.copy()
        self.assertEqual(rows.groupby("source_dataset").size().tolist(), [4, 2])
        scaled = rows.copy()
        fields = [f"actual_cost_{tier}" for tier in TIERS]
        scaled.loc[scaled.source_dataset.eq("xroutebench"), fields] *= 1000
        original, runs = _matched_random(rows, protocol)
        changed, scaled_runs = _matched_random(scaled, protocol)
        for before, after in zip(
            original["summaries"], changed["summaries"], strict=True
        ):
            self.assertEqual(before["mean_quality"], after["mean_quality"])
            for statistic in ("mean", "std", "p025", "p975"):
                self.assertAlmostEqual(
                    before["relative_cost"][statistic],
                    after["relative_cost"][statistic],
                )
            if before["source_dataset"] == "source_macro":
                self.assertEqual(
                    before["mean_actual_cost"],
                    {"available": False, "reason": "incomparable_source_cost_units"},
                )
            elif before["source_dataset"] == "xroutebench":
                self.assertAlmostEqual(
                    before["mean_actual_cost"]["mean"] * 1000,
                    after["mean_actual_cost"]["mean"],
                )
        self.assertTrue(
            runs.loc[runs.source_dataset.eq("source_macro"), "mean_actual_cost"]
            .isna()
            .all()
        )
        np.testing.assert_allclose(runs.relative_cost, scaled_runs.relative_cost)
        summary = _summary(rows, _actions_frame(rows, np.ones(len(rows), dtype=int)))
        self.assertEqual(
            summary["row_weighted_actual_cost_interpretation"],
            "pooled_source_cost_not_comparable",
        )
        rows.loc[rows.source_dataset.eq("xroutebench"), "actual_cost_high"] = 0.1
        degenerate, _ = _matched_random(rows, protocol)
        for result in degenerate["summaries"]:
            if result["source_dataset"] in ("source_macro", "xroutebench"):
                self.assertEqual(
                    result["relative_cost"],
                    {
                        "available": False,
                        "reason": "nonpositive_source_high_low_cost_span",
                    },
                )


class GroupBootstrapTest(unittest.TestCase):
    def test_cost_macro_is_scale_invariant_and_resamples_source_denominators(self):
        rows = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(6)],
                "source_dataset": ["a"] * 4 + ["b"] * 2,
                "duplicate_group_id": ["a0"] * 2 + ["a1"] * 2 + ["b0", "b1"],
                "difficulty_label": "medium",
                "actual_cost_low": 0.1,
                "actual_cost_high": [0.9, 0.9, 1.7, 1.7, 0.9, 0.9],
            }
        )
        delta = np.column_stack(
            ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.2, 0.4, 0.2, 0.4, 0.8, 0.4])
        )
        original = paired_group_bootstrap(
            rows, delta, iterations=1000, cost_column_indices=(1,)
        )
        scaled = rows.copy()
        scaled.loc[
            scaled.source_dataset.eq("b"), ["actual_cost_low", "actual_cost_high"]
        ] *= 1000
        scaled_delta = delta.copy()
        scaled_delta[4:, 1] *= 1000
        changed = paired_group_bootstrap(
            scaled, scaled_delta, iterations=1000, cost_column_indices=(1,)
        )
        for statistic in ("estimate", "ci95_lower", "ci95_upper"):
            np.testing.assert_allclose(
                original["source_macro"][statistic], changed["source_macro"][statistic]
            )
            self.assertAlmostEqual(
                original["per_source"]["b"][statistic][1] * 1000,
                changed["per_source"]["b"][statistic][1],
            )
        self.assertAlmostEqual(original["source_macro"]["estimate"][1], 0.5)
        # The lowest paired draw uses A's larger span and B's cheaper delta.
        # Reusing the original A span would incorrectly make this CI 0.375.
        self.assertAlmostEqual(original["source_macro"]["ci95_lower"][1], 0.34375)
        self.assertNotEqual(
            original["row_weighted"]["estimate"][1],
            changed["row_weighted"]["estimate"][1],
        )
        self.assertEqual(
            changed["row_weighted"]["cost_columns_interpretation"],
            "pooled_source_cost_not_comparable",
        )
        self.assertEqual(
            changed["source_macro"]["columns"][1]["metric_scale"],
            "source_relative_cost_delta",
        )

    def test_cost_normalization_failure_is_explicit_without_dropping_draws(self):
        rows = pd.DataFrame(
            {
                "sample_id": ["a", "b"],
                "source_dataset": "s",
                "duplicate_group_id": ["a", "b"],
                "difficulty_label": "medium",
                "actual_cost_low": 0.1,
                "actual_cost_high": [0.1, 0.9],
            }
        )
        delta = np.array([[0.1], [0.2]])
        result = paired_group_bootstrap(
            rows, delta, iterations=1000, cost_column_indices=(0,)
        )
        self.assertAlmostEqual(result["source_macro"]["estimate"][0], 0.375)
        self.assertEqual(result["source_macro"]["ci95_lower"], [None])
        self.assertEqual(result["source_macro"]["ci95_upper"], [None])
        detail = result["source_macro"]["columns"][0]
        self.assertGreater(detail["invalid_draw_count"], 0)
        self.assertEqual(
            detail["valid_draw_count"] + detail["invalid_draw_count"], 1000
        )
        self.assertEqual(
            detail["unavailable_reason"],
            "nonpositive_source_high_low_cost_span_in_bootstrap_draw",
        )
        rows["actual_cost_high"] = 0.1
        result = paired_group_bootstrap(
            rows, delta, iterations=1000, cost_column_indices=(0,)
        )
        self.assertEqual(result["source_macro"]["estimate"], [None])
        self.assertEqual(
            result["source_macro"]["columns"][0]["unavailable_reason"],
            "nonpositive_source_high_low_cost_span",
        )
        self.assertAlmostEqual(result["per_source"]["s"]["estimate"][0], 0.15)

    def test_duplicate_group_is_atomic_even_when_its_size_differs_hundredfold(self):
        rows = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(101)],
                "source_dataset": "source-a",
                "duplicate_group_id": ["large"] * 100 + ["small"],
                "difficulty_label": "medium",
            }
        )
        delta = np.array([1.0] * 100 + [-1.0])[:, None]
        result = paired_group_bootstrap(rows, delta, iterations=1000)
        self.assertEqual(result["group_count"], 2)
        # Row resampling would almost never produce all-negative draws here.
        self.assertEqual(result["source_macro"]["ci95_lower"], [-1.0])
        self.assertEqual(result["source_macro"]["ci95_upper"], [1.0])
        self.assertAlmostEqual(result["source_macro"]["estimate"][0], 99 / 101)

    def test_source_macro_pairing_and_row_order_are_preserved(self):
        rows = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(8)],
                "source_dataset": ["a"] * 6 + ["b"] * 2,
                "duplicate_group_id": ["a0"] * 3 + ["a1"] + ["a2"] * 2 + ["b0", "b1"],
                "difficulty_label": "medium",
            }
        )
        first = np.array([1, 1, 1, -1, 0, 0, 0.5, -0.5])
        differences = np.stack([first, -first], axis=1)
        result = paired_group_bootstrap(rows, differences, iterations=400)
        order = np.array([7, 2, 5, 0, 1, 6, 4, 3])
        reordered = paired_group_bootstrap(
            rows.iloc[order], differences[order], iterations=400
        )
        self.assertEqual(result, reordered)
        self.assertAlmostEqual(result["source_macro"]["estimate"][0], 1 / 6)
        self.assertAlmostEqual(result["row_weighted"]["estimate"][0], 1 / 4)
        self.assertAlmostEqual(
            result["source_macro"]["ci95_lower"][0],
            -result["source_macro"]["ci95_upper"][1],
        )

    def test_mixed_label_groups_stay_intact_and_cross_source_groups_fail(self):
        rows = pd.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "source_dataset": ["s", "s", "s"],
                "duplicate_group_id": ["g", "g", "h"],
                "difficulty_label": ["easy", "hard", "medium"],
            }
        )
        report = paired_group_bootstrap(rows, np.ones((3, 1)), iterations=20)
        self.assertIn(
            ["easy", "hard"],
            [item["label_signature"] for item in report["stratum_support"]],
        )
        rows.loc[1, "source_dataset"] = "other"
        with self.assertRaisesRegex(SealedProtocolError, "Source-spanning"):
            paired_group_bootstrap(rows, np.ones((3, 1)), iterations=20)


if __name__ == "__main__":
    unittest.main()
