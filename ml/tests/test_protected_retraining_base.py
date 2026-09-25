from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from llm_router_labeling.shared.dataset_io import (
    DatasetIOError,
    sha256_file,
    write_json,
)
from llm_router_labeling.training.feedback_retraining_cli import (
    command_merge,
    command_prepare,
)
from llm_router_labeling.training.feedback_retraining_preprocessing import (
    FeedbackDataError,
    merge_feedback_with_base_training,
)
from llm_router_labeling.training.protected_retraining_base import (
    PARTITIONS,
    build_protected_retraining_base,
)


class ProtectedRetrainingBaseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.contract = {
            "dataset_version": "fixture-v1",
            "label_rule_version": "overall-solve-rate-grid-v1",
            "preprocessing_version": "structure-v2",
            "split_version": "frozen-split",
        }
        self.manifest = {
            **self.contract,
            "artifacts": {},
            "input_contract": "structure-preserving-v2",
            "effective_sample_weight": 1.0,
            "selected_thresholds": {
                "selection_data": "train_only_group_oof",
                "routerarena_used_for_selection": False,
                "t1": {
                    "candidate_id": "h0.27-e0.64",
                    "hard_threshold": 0.27,
                    "easy_threshold": 0.64,
                },
            },
            "row_counts": {key: 1 for key in PARTITIONS},
        }
        for split, relative in PARTITIONS.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    **self.contract,
                    "label_rule_version": "overall-solve-rate-grid-v1:h0.27-e0.64",
                    "sample_id": split,
                    "duplicate_group_id": f"group-{split}",
                    "model_input_text": f"Question from {split}",
                    "difficulty_label": "hard",
                    "sample_weight": 1.0,
                    "source_dataset": "routerbench",
                    "split": split,
                }
            ]
            if split in {"train", "validation"}:
                rows.extend(
                    {
                        **rows[0],
                        "sample_id": f"{split}-{label}",
                        "duplicate_group_id": f"group-{split}-{label}",
                        "model_input_text": f"Question from {split} {label}",
                        "difficulty_label": label,
                    }
                    for label in ("easy", "medium")
                )
            self.manifest["row_counts"][split] = len(rows)
            pd.DataFrame(rows).to_parquet(path, index=False)
        (self.source / "audit.json").write_text('{"status":"frozen"}')
        self._refresh_manifest()
        self.feedback_config = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "config/feedback_retraining_preprocessing.json"
            ).read_text()
        )
        self.feedback_config["input_contract"] = "structure-preserving-v2"

    def _refresh_manifest(self):
        for relative in (*PARTITIONS.values(), "audit.json"):
            path = self.source / relative
            self.manifest["artifacts"][relative] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        write_json(self.source / "manifest.json", self.manifest)
        self.manifest_sha = sha256_file(self.source / "manifest.json")

    def _build(self, output="base"):
        return build_protected_retraining_base(
            self.source, self.root / output, expected_manifest_sha256=self.manifest_sha
        )

    def test_stream_is_excluded_and_sealed_rows_only_block_feedback_overlap(self):
        source_hashes = {
            path: sha256_file(path) for path in self.source.rglob("*") if path.is_file()
        }
        report = self._build()
        base = pd.read_parquet(self.root / "base/training.parquet")
        self.assertEqual(report["row_count"], 10)
        self.assertEqual(report["excluded_stream_row_count"], 1)
        self.assertEqual(
            report["split_row_counts"], {"test": 4, "train": 3, "validation": 3}
        )
        self.assertNotIn("stream", set(base["sample_id"]))
        self.assertEqual(
            set(base.loc[base["split"].eq("test"), "original_partition"]),
            {"gate", "retention_test", "test", "external_test"},
        )
        self.assertTrue(
            all(sha256_file(path) == digest for path, digest in source_hashes.items())
        )
        feedback = pd.DataFrame(
            [
                {
                    "sample_id": f"feedback-{partition}",
                    "model_input_text": f"Question from {partition}",
                    "difficulty_label": "hard",
                    "sample_weight": 1.0,
                    "split": "train",
                    "duplicate_group_id": f"feedback-group-{partition}",
                }
                for partition in (
                    "gate",
                    "retention_test",
                    "test",
                    "external_test",
                    "stream",
                )
            ]
        )
        merged = merge_feedback_with_base_training(
            base,
            feedback,
            self.feedback_config,
            training_data_strategy="compact_rehearsal",
        )
        self.assertEqual(merged.metrics["excluded_feedback_row_count"], 4)
        self.assertIn("feedback-stream", set(merged.training["sample_id"]))
        self.assertEqual(set(merged.training["split"]), {"train", "validation"})
        self.assertFalse(
            set(merged.training["sample_id"])
            & {"gate", "retention_test", "test", "external_test"}
        )

    def test_every_manifest_artifact_and_manifest_itself_are_verified(self):
        (self.source / "audit.json").write_text('{"status":"changed"}')
        with self.assertRaises(DatasetIOError):
            self._build()
        self.assertFalse((self.root / "base").exists())
        self._refresh_manifest()
        self.manifest["dataset_version"] = "tampered"
        write_json(self.source / "manifest.json", self.manifest)
        with self.assertRaisesRegex(FeedbackDataError, "manifest SHA-256"):
            self._build()

    def test_group_overlap_is_rejected_including_stream_and_protected_partitions(self):
        for partition in ("gate", "stream"):
            with self.subTest(partition=partition):
                path = self.source / PARTITIONS[partition]
                frame = pd.read_parquet(path)
                frame["duplicate_group_id"] = "group-train"
                frame.to_parquet(path, index=False)
                self._refresh_manifest()
                with self.assertRaisesRegex(FeedbackDataError, "groups overlap"):
                    self._build()
                frame["duplicate_group_id"] = f"group-{partition}"
                frame.to_parquet(path, index=False)
                self._refresh_manifest()

    def test_source_and_existing_outputs_cannot_be_overwritten(self):
        with self.assertRaisesRegex(FeedbackDataError, "outside"):
            build_protected_retraining_base(
                self.source,
                self.source / "derived",
                expected_manifest_sha256=self.manifest_sha,
            )
        self._build()
        with self.assertRaisesRegex(FeedbackDataError, "cannot be overwritten"):
            self._build()

    def test_merge_checks_optional_base_hash_before_reading_or_writing(self):
        report = self._build()
        base_path = self.root / "base/training.parquet"
        feedback_path = self.root / "feedback.parquet"
        pd.DataFrame(
            [
                {
                    "sample_id": "new-feedback",
                    "model_input_text": "A new question",
                    "difficulty_label": "hard",
                    "sample_weight": 1.0,
                    "split": "train",
                    "duplicate_group_id": "new-feedback-group",
                }
            ]
        ).to_parquet(feedback_path, index=False)
        args = argparse.Namespace(
            base_training_path=base_path,
            feedback_training_path=feedback_path,
            output_dir=self.root / "merged",
            training_data_strategy="full_base",
        )
        config = {**self.feedback_config, "base_dataset_sha256": "0" * 64}
        with patch(
            "llm_router_labeling.training.feedback_retraining_cli.pd.read_parquet"
        ) as read:
            with self.assertRaisesRegex(FeedbackDataError, "SHA-256 mismatch"):
                command_merge(args, config)
            read.assert_not_called()
        self.assertFalse(args.output_dir.exists())
        config["base_dataset_sha256"] = report["artifacts"]["training.parquet"][
            "sha256"
        ]
        self.assertEqual(command_merge(args, config)["accepted_feedback_row_count"], 1)
        args.output_dir = self.root / "legacy-merge"
        self.assertEqual(
            command_merge(args, self.feedback_config)["accepted_feedback_row_count"], 1
        )

    def test_draft_config_cannot_prepare_or_merge_before_objective_selection(self):
        config = {"deployment_status": "blocked_pending_objective_selection"}
        for command in (command_prepare, command_merge):
            with (
                self.subTest(command=command.__name__),
                self.assertRaisesRegex(FeedbackDataError, "not approved"),
            ):
                command(argparse.Namespace(), config)


if __name__ == "__main__":
    unittest.main()
