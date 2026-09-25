from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from llm_router_labeling.objective_comparison.data import ObjectiveDataError
from llm_router_labeling.shared.dataset_io import (
    DatasetIOError,
    load_config,
    sha256_file,
    write_json,
)
from llm_router_labeling.training.frozen_outcome_projection import (
    OUTPUT_ROLES,
    PARTITIONS,
    SELECTED_MODELS,
    build_frozen_evaluation_outcomes,
)


class FrozenOutcomeProjectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.outcome = self.root / "outcome"
        self.base.mkdir()
        self.outcome.mkdir()
        self.frames = {}
        self.base_manifest = {"artifacts": {}, "source_artifacts": {}, "row_counts": {}}
        for logical, relative in PARTITIONS.items():
            split = {
                "test_routerbench": "test",
                "test_xroutebench": "external_test",
            }.get(logical, logical)
            sources = (
                [logical.removeprefix("test_")]
                if logical.startswith("test_")
                else list(SELECTED_MODELS)
            )
            rows = [
                {
                    "sample_id": f"{source}:{logical}",
                    "source_dataset": source,
                    "model_input_text": f"  if {source}:\n    return '{logical}'",
                    "difficulty_label": "medium",
                    "duplicate_group_id": f"group-{source}-{logical}",
                    "split": split,
                    "sample_weight": 1.0,
                }
                for source in sources
            ]
            frame = pd.DataFrame(rows)
            path = self.base / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            self.frames[logical] = frame
            self.base_manifest["row_counts"][split] = len(frame)
        self.config = {
            "base_dataset_dir": "base",
            "input_contract": "structure-preserving-v2",
            "routerbench": {
                "labeled_path": "raw/rb.parquet",
                "model_tiers_path": "raw/tiers.json",
            },
            "xroutebench": {
                "data_dir": "raw/xrb",
                "train_file": "train.parquet",
                "test_file": "test.parquet",
                "candidates_file": "candidates.parquet",
            },
        }
        for relative in (
            "raw/rb.parquet",
            "raw/tiers.json",
            "raw/xrb/train.parquet",
            "raw/xrb/test.parquet",
            "raw/xrb/candidates.parquet",
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"immutable source placeholder; matrix helper is mocked")
            self.base_manifest["source_artifacts"][relative] = self._record(path)
        self.arms = {
            "selection_split": "train_only",
            "sources": {
                source: {"selected_models": models, "training_row_count": 1}
                for source, models in SELECTED_MODELS.items()
            },
        }
        write_json(self.outcome / "arms.json", self.arms)
        statistics = pd.DataFrame(
            [
                {
                    "source_dataset": source,
                    "model_name": model,
                    "selected_tier": tier,
                    "valid_rows": 1,
                    "mean_cost": cost,
                }
                for source, models in SELECTED_MODELS.items()
                for (tier, model), cost in zip(
                    models.items(), [0.1, 0.3, 0.9], strict=True
                )
            ]
        )
        statistics.to_parquet(
            self.outcome / "candidate-statistics.parquet", index=False
        )
        for split in ("train", "validation"):
            self.frames[split].to_parquet(
                self.outcome / f"{split}.parquet", index=False
            )
        self._refresh_pins()
        self.matrices = {}
        for source, models in SELECTED_MODELS.items():
            ids = [
                sample_id
                for split, frame in self.frames.items()
                if split in OUTPUT_ROLES
                for sample_id in frame.loc[
                    frame["source_dataset"].eq(source), "sample_id"
                ]
            ]
            quality = pd.DataFrame(
                [[0.2, 0.7, 0.9]] * len(ids), columns=list(models.values()), index=ids
            )
            success = pd.DataFrame(
                [[0.0, 1.0, 1.0]] * len(ids), columns=list(models.values()), index=ids
            )
            cost = pd.DataFrame(
                [[0.001, 0.004, 0.015]] * len(ids),
                columns=list(models.values()),
                index=ids,
            )
            self.matrices[source] = (quality, success, cost)

    @staticmethod
    def _record(path):
        return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}

    def _refresh_pins(self):
        self.base_manifest["artifacts"] = {
            relative: self._record(self.base / relative)
            for relative in PARTITIONS.values()
        }
        write_json(self.base / "manifest.json", self.base_manifest)
        self.base_sha = sha256_file(self.base / "manifest.json")
        self.config["base_dataset_manifest_sha256"] = self.base_sha
        write_json(self.root / "config.json", self.config)
        self.config_sha = sha256_file(self.root / "config.json")
        outcome_manifest = {
            "base_dataset_manifest_sha256": self.base_sha,
            "arms": self.arms,
            "routerarena_used_for_selection": False,
            "artifacts": {
                name: self._record(self.outcome / name)
                for name in (
                    "arms.json",
                    "candidate-statistics.parquet",
                    "train.parquet",
                    "validation.parquet",
                )
            },
        }
        write_json(self.outcome / "manifest.json", outcome_manifest)
        self.outcome_sha = sha256_file(self.outcome / "manifest.json")
        (self.outcome / "_SUCCESS").write_text(self.outcome_sha + "\n")

    def _build(self):
        with (
            patch(
                "llm_router_labeling.training.frozen_outcome_projection._routerbench_matrices",
                return_value=self.matrices["routerbench"],
            ),
            patch(
                "llm_router_labeling.training.frozen_outcome_projection._xroute_matrices",
                return_value=self.matrices["xroutebench"],
            ),
            patch(
                "llm_router_labeling.objective_comparison.data.select_three_arms",
                side_effect=AssertionError("Arms must never be reselected"),
            ),
        ):
            return build_frozen_evaluation_outcomes(
                project_root=self.root,
                config_path=self.root / "config.json",
                outcome_dir=self.outcome,
                output_dir=self.root / "output",
                expected_base_manifest_sha256=self.base_sha,
                expected_outcome_manifest_sha256=self.outcome_sha,
                expected_config_sha256=self.config_sha,
            )

    def test_projection_preserves_roles_and_rows_and_uses_frozen_train_costs(self):
        before = {
            p: sha256_file(p)
            for directory in (self.base, self.outcome)
            for p in directory.rglob("*")
            if p.is_file()
        }
        report = self._build()
        self.assertEqual(report["coverage_status"], "complete")
        self.assertEqual(sum(report["row_counts"].values()), 8)
        self.assertEqual(report["dropped_row_count"], 0)
        self.assertEqual(report["stage"], "target_projection_only_no_model_evaluation")
        for split, role in OUTPUT_ROLES.items():
            frame = pd.read_parquet(self.root / "output" / f"{split}.parquet")
            original = self.frames[split]
            pd.testing.assert_frame_equal(
                frame.loc[:, original.columns].reset_index(drop=True), original
            )
            self.assertEqual(report["partition_roles"][split], role)
            self.assertTrue(np.allclose(frame["decision_cost_medium"], 0.25))
            self.assertTrue(np.allclose(frame["actual_cost_medium"], 0.004))
            self.assertTrue(frame["success_low"].eq(0).all())
            self.assertNotIn("oracle_quality_tier", frame)
        self.assertFalse((self.root / "output/train.parquet").exists())
        self.assertTrue(all(sha256_file(p) == digest for p, digest in before.items()))

    def test_missing_selected_outcome_fails_with_coverage_without_dropping_rows(self):
        quality = self.matrices["routerbench"][0]
        quality.iloc[0, 2] = np.nan
        with self.assertRaisesRegex(ObjectiveDataError, "incomplete"):
            self._build()
        report = load_config(self.root / "output/coverage-report.json")
        self.assertEqual(report["status"], "blocked_incomplete_outcomes")
        self.assertEqual(report["invalid_target_count"], 1)
        self.assertEqual(
            report["sources"]["routerbench"]["fields"]["quality_high"][
                "invalid_sample_ids"
            ],
            ["routerbench:gate"],
        )
        self.assertFalse((self.root / "output/_SUCCESS").exists())
        self.assertEqual(list((self.root / "output").glob("*.parquet")), [])

    def test_config_arms_statistics_and_source_tampering_are_rejected(self):
        paths = [
            self.root / "config.json",
            self.outcome / "arms.json",
            self.outcome / "candidate-statistics.parquet",
            self.root / "raw/rb.parquet",
        ]
        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with self.assertRaises((ObjectiveDataError, DatasetIOError)):
                    self._build()
                self.assertFalse((self.root / "output").exists())
                path.write_bytes(original)

    def test_train_and_gate_group_overlap_is_rejected_before_projection(self):
        path = self.base / PARTITIONS["gate"]
        frame = pd.read_parquet(path)
        frame.loc[0, "duplicate_group_id"] = "group-routerbench-train"
        frame.to_parquet(path, index=False)
        self._refresh_pins()
        with self.assertRaisesRegex(ObjectiveDataError, "crosses objective splits"):
            self._build()
        self.assertFalse((self.root / "output").exists())


if __name__ == "__main__":
    unittest.main()
