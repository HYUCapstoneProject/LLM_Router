from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from llm_router_labeling.shared.dataset_io import load_config
from llm_router_labeling.shared.object_store import LocalObjectStore
from llm_router_labeling.training.feedback_retraining_cli import _reserve_unique_batch
from llm_router_labeling.training.feedback_retraining_preprocessing import (
    FeedbackDataError,
    canonical_training_digest,
    load_feedback_events,
    merge_feedback_with_base_training,
    prepare_feedback_training_dataset,
    validate_feedback_manifest,
)
from llm_router_labeling.training.routerbench_preprocessing import TRAINING_COLUMNS


class FeedbackRetrainingPreprocessingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ml_root = Path(__file__).resolve().parents[1]
        cls.fixture_root = cls.ml_root / "tests" / "fixtures" / "feedback_v0_1"
        cls.events_path = cls.fixture_root / "events.jsonl"
        cls.manifest = load_config(cls.fixture_root / "manifest.json")
        cls.config = load_config(
            cls.ml_root / "config" / "feedback_retraining_preprocessing.json"
        )

    def _prepare(self):
        validate_feedback_manifest(
            self.manifest,
            self.events_path,
            expected_schema_version=self.config["schema_version"],
        )
        events = load_feedback_events(self.events_path, self.manifest["data_format"])
        return prepare_feedback_training_dataset(events, self.manifest, self.config)

    def test_manifest_checksum_and_version_are_enforced(self) -> None:
        invalid = json.loads(json.dumps(self.manifest))
        invalid["sha256"] = "0" * 64
        with self.assertRaisesRegex(FeedbackDataError, "SHA-256 mismatch"):
            validate_feedback_manifest(
                invalid,
                self.events_path,
                expected_schema_version=self.config["schema_version"],
            )

    def test_prepares_only_consistent_escalation_labels(self) -> None:
        result = self._prepare()
        self.assertEqual(tuple(result.training.columns), TRAINING_COLUMNS)
        self.assertEqual(len(result.training), 2)
        labels = dict(
            zip(
                result.training["model_input_text"],
                result.training["difficulty_label"],
                strict=True,
            )
        )
        self.assertEqual(labels["Explain why the sky appears blue."], "medium")
        self.assertEqual(
            labels["Prove that there are infinitely many prime numbers."], "hard"
        )
        repeated = result.training[
            result.training["model_input_text"].str.contains("sky")
        ].iloc[0]
        self.assertEqual(repeated["sample_weight"], 1.25)
        self.assertEqual(result.metrics["accepted_event_count"], 3)
        self.assertEqual(result.metrics["training_row_count"], 2)
        self.assertEqual(
            result.metrics["exclusion_distribution"]["conflicting_final_labels"],
            2,
        )
        self.assertEqual(len(result.metrics["canonical_batch_digest"]), 64)
        self.assertEqual(
            result.metrics["canonical_batch_digest"],
            canonical_training_digest(result.training.iloc[::-1].reset_index(drop=True)),
        )

    def test_merge_replaces_train_conflict_and_protects_holdout(self) -> None:
        feedback = self._prepare().training
        base = pd.DataFrame(
            [
                {
                    "sample_id": "base-sky",
                    "model_input_text": "Explain why the sky appears blue.",
                    "difficulty_label": "easy",
                    "sample_weight": 1.0,
                    "split": "train",
                    "duplicate_group_id": "base-group-sky",
                },
                {
                    "sample_id": "base-primes",
                    "model_input_text": "Prove that there are infinitely many prime numbers.",
                    "difficulty_label": "medium",
                    "sample_weight": 1.0,
                    "split": "validation",
                    "duplicate_group_id": "base-group-primes",
                },
                {
                    "sample_id": "base-other",
                    "model_input_text": "Describe a balanced binary search tree.",
                    "difficulty_label": "medium",
                    "sample_weight": 1.0,
                    "split": "train",
                    "duplicate_group_id": "base-group-other",
                },
            ],
            columns=TRAINING_COLUMNS,
        )
        result = merge_feedback_with_base_training(base, feedback, self.config)
        self.assertEqual(len(result.training), 3)
        self.assertNotIn("base-sky", set(result.training["sample_id"]))
        sky = result.training[
            result.training["model_input_text"].str.contains("sky")
        ].iloc[0]
        self.assertEqual(sky["difficulty_label"], "medium")
        self.assertEqual(sky["duplicate_group_id"], "base-group-sky")
        validation = result.training[result.training["split"].eq("validation")]
        self.assertEqual(validation.iloc[0]["sample_id"], "base-primes")
        self.assertEqual(result.metrics["accepted_feedback_row_count"], 1)
        self.assertEqual(result.metrics["excluded_feedback_row_count"], 1)
        self.assertEqual(result.metrics["replaced_base_row_count"], 1)

    def test_compact_rehearsal_is_small_balanced_and_deterministic(self) -> None:
        feedback = self._prepare().training
        rows = []
        for split, count in (("train", 40), ("validation", 20)):
            for label in ("easy", "medium", "hard"):
                for index in range(count):
                    sample_id = f"base-{split}-{label}-{index}"
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "model_input_text": f"Unique {split} {label} question {index}",
                            "difficulty_label": label,
                            "sample_weight": 1.0,
                            "split": split,
                            "duplicate_group_id": sample_id,
                        }
                    )
        base = pd.DataFrame(rows, columns=TRAINING_COLUMNS)

        first = merge_feedback_with_base_training(
            base,
            feedback,
            self.config,
            training_data_strategy="compact_rehearsal",
        )
        second = merge_feedback_with_base_training(
            base,
            feedback,
            self.config,
            training_data_strategy="compact_rehearsal",
        )

        rehearsal = first.training[
            ~first.training["sample_id"].str.startswith("feedback-")
        ]
        counts = rehearsal.groupby(["split", "difficulty_label"]).size().to_dict()
        self.assertEqual(set(counts.values()), {16, 32})
        self.assertEqual(len(rehearsal), 144)
        self.assertEqual(
            first.training["sample_id"].tolist(), second.training["sample_id"].tolist()
        )
        self.assertEqual(first.metrics["training_data_strategy"], "compact_rehearsal")

    def test_validated_unique_policy_rejects_a_committed_content_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalObjectStore(Path(directory))
            root = "gs://llm-router/retraining-control"
            args = argparse.Namespace(
                version_policy_mode="validatedUniqueBatch",
                retraining_control_root_uri=root,
                batch_id="batch-1",
                candidate_model_version="router-candidate-1",
            )
            metrics = {"canonical_batch_digest": "a" * 64}
            self.assertTrue(_reserve_unique_batch(args, metrics, object_store=store))
            store.write_text(
                f"{root}/validated-batches/committed/{'a' * 64}.json",
                "{}",
                if_generation_match=None,
            )
            duplicate_metrics = {"canonical_batch_digest": "a" * 64}
            self.assertFalse(
                _reserve_unique_batch(args, duplicate_metrics, object_store=store)
            )
            self.assertEqual(
                duplicate_metrics["failure_code"], "NO_UNIQUE_FEEDBACK"
            )


if __name__ == "__main__":
    unittest.main()
