from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

from llm_router_labeling.training.routerbench_labeling import (
    ModelTiers,
    RouterBenchError,
    calibrate_model_tiers,
    canonicalize_sources,
    detect_model_columns,
    extract_query_text,
    filter_eval_scope,
    label_questions,
    stratified_smoke_sample,
    validate_smoke_output,
)
from llm_router_labeling.training.routerbench_cli import (
    build_parser,
    full_label_output_dir,
    full_preprocess_output_dir,
    resolve_preprocess_paths,
    validate_labeled_versions,
)


MODELS = [f"model-{index}" for index in range(1, 7)]


def source_frame(shot: str) -> pd.DataFrame:
    rows = []
    for index in range(12):
        row = {
            "sample_id": f"mmlu-test.{index}",
            "prompt": f"['question {index} {shot}']",
            "eval_name": "mmlu.test",
            "oracle_model_to_route_to": "model-6",
        }
        for model_index, model in enumerate(MODELS, start=1):
            row[model] = float(model_index > index % 6)
            row[f"{model}|model_response"] = "answer"
            row[f"{model}|total_cost"] = model_index / 1000
        rows.append(row)
    return pd.DataFrame(rows)


class RouterBenchLabelingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.canonical_config = {
            "included_eval_patterns": ["mmlu", "mbpp"],
            "five_shot_patterns": ["mmlu"],
            "zero_shot_patterns": ["mbpp"],
            "default_source": "0shot",
        }
        self.acceptance_config = {
            "exact_match_patterns": ["mmlu"],
            "continuous_patterns": ["mbpp"],
            "exact_match_threshold": 1.0,
            "continuous_threshold": 0.7,
            "default_threshold": 1.0,
        }
        self.labeling_config = {
            "low_consensus_threshold": 0.67,
            "medium_consensus_threshold": 0.67,
            "high_consensus_threshold": 0.5,
            "minimum_valid_models_for_confidence": 2,
        }

    def test_detects_only_performance_columns(self) -> None:
        self.assertEqual(detect_model_columns(source_frame("0shot")), MODELS)

    def test_detects_object_dtype_performance_columns(self) -> None:
        dataframe = source_frame("0shot")
        for model in MODELS:
            dataframe[model] = dataframe[model].astype(object)
        self.assertEqual(detect_model_columns(dataframe), MODELS)

    def test_canonicalizer_prefers_five_shot_for_mmlu(self) -> None:
        canonical, models = canonicalize_sources(
            source_frame("0shot"), source_frame("5shot"), self.canonical_config
        )
        self.assertEqual(models, MODELS)
        self.assertEqual(len(canonical), 12)
        self.assertTrue(canonical["shot_source"].eq("5shot").all())
        self.assertTrue(canonical["query_text"].str.endswith("5shot").all())

    def test_extracts_target_question_from_five_shot_prompt(self) -> None:
        prompt = str(
            [
                "Example one\nAnswer:\nA\n\n"
                "Example two\nAnswer:\nB\n\n"
                "Target question\nA) one\nB) two\nAnswer:"
            ]
        )
        self.assertEqual(
            extract_query_text(prompt), "Target question\nA) one\nB) two"
        )

    def test_extracts_target_from_separate_five_shot_list_item(self) -> None:
        prompt = str(
            [
                "TASK: Solve the problem.\nQuestion: Example one\nAnswer: 1\n\n"
                "Question: Example two\nAnswer: 2",
                "A train travels 60 kilometers in one hour. How far in two hours?",
            ]
        )
        self.assertEqual(
            extract_query_text(prompt),
            "A train travels 60 kilometers in one hour. How far in two hours?",
        )

    def test_preserves_multiturn_context(self) -> None:
        prompt = str(["Initial question", "Why is the above question interesting?"])
        self.assertEqual(
            extract_query_text(prompt),
            "Initial question\n\nWhy is the above question interesting?",
        )

    def test_filters_supplemental_eval_scope(self) -> None:
        dataframe = source_frame("0shot")
        supplemental = dataframe.iloc[[0]].copy()
        supplemental["sample_id"] = "supplemental.dev.0"
        supplemental["eval_name"] = "supplemental"
        scoped, metrics = filter_eval_scope(
            pd.concat([dataframe, supplemental], ignore_index=True),
            self.canonical_config,
        )
        self.assertEqual(len(scoped), len(dataframe))
        self.assertEqual(metrics["scope_excluded_row_count"], 1)
        self.assertEqual(metrics["scope_excluded_eval_names"], ["supplemental"])

    def test_calibration_partitions_every_model_once(self) -> None:
        canonical, models = canonicalize_sources(
            source_frame("0shot"), source_frame("5shot"), self.canonical_config
        )
        tiers = calibrate_model_tiers(canonical, models)
        self.assertEqual(set(tiers.all_models), set(MODELS))
        self.assertEqual(len(tiers.all_models), len(set(tiers.all_models)))
        self.assertEqual(len(tiers.low), 2)
        self.assertEqual(len(tiers.medium), 2)
        self.assertEqual(len(tiers.high), 2)

    def test_label_rules_cover_easy_medium_and_hard(self) -> None:
        rows = []
        performance_rows = (
            ("easy", [1, 1, 1, 1, 1, 1]),
            ("medium", [0, 0, 1, 1, 1, 1]),
            ("hard-strong", [0, 0, 0, 0, 1, 1]),
            ("hard-unsolved", [0, 0, 0, 0, 0, 0]),
        )
        for sample_id, scores in performance_rows:
            row = {
                "sample_id": sample_id,
                "prompt": "question",
                "query_text": "question",
                "eval_name": "mmlu.test",
                "domain": "knowledge",
                "shot_source": "5shot",
                "needs_retrieval": False,
            }
            for model, score in zip(MODELS, scores, strict=True):
                row[model] = score
                row[f"{model}|total_cost"] = int(model.split("-")[1]) / 1000
            rows.append(row)
        dataframe = pd.DataFrame(rows)
        tiers = ModelTiers(
            low=("model-1", "model-2"),
            medium=("model-3", "model-4"),
            high=("model-5", "model-6"),
            statistics={},
        )
        labeled = label_questions(
            dataframe, tiers, self.acceptance_config, self.labeling_config
        )
        self.assertEqual(
            labeled["difficulty_label"].tolist(),
            ["easy", "medium", "hard", "hard"],
        )
        self.assertEqual(
            labeled["label_reason"].tolist(),
            [
                "low_tier_consensus",
                "medium_tier_consensus",
                "strong_only",
                "unsolved",
            ],
        )

    def test_smoke_sample_is_reproducible_and_valid(self) -> None:
        canonical, _ = canonicalize_sources(
            source_frame("0shot"), source_frame("5shot"), self.canonical_config
        )
        first = stratified_smoke_sample(canonical, 8, 42)
        second = stratified_smoke_sample(canonical, 8, 42)
        self.assertEqual(first["sample_id"].tolist(), second["sample_id"].tolist())
        tiers = ModelTiers(
            low=("model-1", "model-2"),
            medium=("model-3", "model-4"),
            high=("model-5", "model-6"),
            statistics={},
        )
        labeled = label_questions(
            first, tiers, self.acceptance_config, self.labeling_config
        )
        validate_smoke_output(labeled, 8)

    def test_cli_exposes_full_labeling_profile(self) -> None:
        args = build_parser().parse_args(["label", "--trust-pickle"])
        self.assertEqual(args.command, "label")
        self.assertTrue(args.trust_pickle)

    def test_full_profile_uses_versioned_default_paths(self) -> None:
        config = {
            "dataset_version": "dataset-v1",
            "tier_config_version": "tiers-v2",
            "label_rule_version": "labels-v3",
            "preprocessing_version": "preprocess-v4",
        }
        label_dir = full_label_output_dir(config)
        training_dir = full_preprocess_output_dir(config)
        self.assertEqual(
            label_dir,
            Path("ml/artifacts/routerbench-labeled/dataset-v1__tiers-v2__labels-v3"),
        )
        self.assertEqual(
            training_dir,
            Path("ml/artifacts/routerbench-training/dataset-v1__labels-v3__preprocess-v4"),
        )
        input_path, output_dir = resolve_preprocess_paths(
            Namespace(profile="full", input_path=None, output_dir=None), config
        )
        self.assertEqual(input_path, label_dir / "labeled.parquet")
        self.assertEqual(output_dir, training_dir)

    def test_preprocessing_rejects_mismatched_label_versions(self) -> None:
        config = {
            "dataset_version": "dataset-v1",
            "tier_config_version": "tiers-v2",
            "label_rule_version": "labels-v3",
        }
        labeled = pd.DataFrame(
            {
                "dataset_version": ["dataset-v1"],
                "tier_config_version": ["tiers-v2"],
                "label_rule_version": ["outdated-labels"],
            }
        )
        with self.assertRaisesRegex(RouterBenchError, "label_rule_version"):
            validate_labeled_versions(labeled, config)


if __name__ == "__main__":
    unittest.main()
