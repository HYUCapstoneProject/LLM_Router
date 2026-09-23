from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from llm_router_labeling.evaluation.routerarena_epoch_evaluation import (
    RouterArenaEpochEvaluationError,
    RouterArenaEpochEvaluator,
)
from llm_router_labeling.training.modernbert_checkpointing import (
    CHECKPOINT_MANIFEST_NAME,
    CHECKPOINT_SCHEMA_VERSION,
    seal_artifact_directory,
)


class StaticPredictor:
    def __init__(self, labels: list[str]) -> None:
        self.labels = labels
        self.calls = 0

    def predict(self, texts: list[str]) -> pd.DataFrame:
        self.calls += 1
        if len(texts) != len(self.labels):
            raise AssertionError("Unexpected evaluation row count")
        return pd.DataFrame(
            {
                "predicted_label": self.labels,
                "prediction_confidence": [0.9] * len(texts),
                "was_truncated": [False] * len(texts),
                "inference_latency_ms": [1.0] * len(texts),
            }
        )

    def info(self):
        return {
            "model_path": "test-checkpoint",
            "label_order": ["easy", "medium", "hard"],
        }


class RouterArenaEpochEvaluationTest(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path, Path, Path]:
        training_path = root / "routerbench-training.parquet"
        training_path.write_bytes(b"separate-training-input")
        evaluation_path = root / "routerarena-evaluation.parquet"
        pd.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "model_input_text": ["easy question", "medium question", "hard question"],
                "difficulty_label": ["easy", "medium", "hard"],
                "evaluation_split": ["full"] * 3,
                "source_sample_id": ["source-a", "source-b", "source-c"],
            }
        ).to_parquet(evaluation_path, index=False)
        metadata_path = root / "metadata.parquet"
        pd.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "domain": ["d1", "d1", "d2"],
                "source_dataset": ["s1", "s1", "s2"],
            }
        ).to_parquet(metadata_path, index=False)
        config_path = root / "routerarena-config.json"
        config_path.write_text(
            json.dumps(
                {
                    "evaluation_version": "epoch-test-v1",
                    "label_semantics": "inference-only-test",
                    "label_order": ["easy", "medium", "hard"],
                    "route_mapping": {
                        "easy": "small",
                        "medium": "medium",
                        "hard": "large",
                    },
                    "inference": {
                        "batch_size": 2,
                        "max_length": 32,
                        "device": "cpu",
                    },
                }
            ),
            encoding="utf-8",
        )
        return training_path, evaluation_path, metadata_path, config_path

    def test_three_epoch_results_are_persisted_and_visualized_without_training_use(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training, evaluation, metadata, config = self._inputs(root)
            evaluator = RouterArenaEpochEvaluator(
                training_path=training,
                evaluation_path=evaluation,
                metadata_path=metadata,
                config_path=config,
                local_output_dir=root / "local",
                persistent_output_dir=root / "drive",
                device="cpu",
            )
            epoch_predictions = (
                ["easy", "easy", "easy"],
                ["easy", "medium", "medium"],
                ["easy", "medium", "hard"],
            )
            for epoch_number, labels in enumerate(epoch_predictions, start=1):
                evaluator.evaluate_predictor(
                    StaticPredictor(labels),
                    epoch_number=epoch_number,
                    optimizer_steps=epoch_number * 10,
                    training_fingerprint="training-run-a",
                    checkpoint_path=root / f"checkpoint-{epoch_number}",
                )

            progress = evaluator.require_all_epochs(
                expected_epochs=3,
                training_fingerprint="training-run-a",
            )
            self.assertEqual(
                [row["epoch_number"] for row in progress["epochs"]], [1, 2, 3]
            )
            self.assertEqual(progress["epochs"][-1]["accuracy"], 1.0)
            self.assertGreater(
                progress["epochs"][-1]["delta_from_first_epoch"]["macro_f1"],
                0.0,
            )
            self.assertTrue((root / "drive" / "epoch_progress.svg").is_file())
            self.assertIn(
                "RouterArena inference-only metrics",
                (root / "drive" / "epoch_progress.svg").read_text(
                    encoding="utf-8"
                ),
            )
            for epoch_number in range(1, 4):
                epoch_dir = root / "drive" / f"epoch-{epoch_number:02d}"
                usage = json.loads(
                    (epoch_dir / "usage_contract.json").read_text(encoding="utf-8")
                )
                self.assertFalse(usage["affects_loss"])
                self.assertFalse(usage["affects_optimizer"])
                self.assertFalse(usage["affects_checkpoint_selection"])
                self.assertTrue((epoch_dir / "predictions.parquet").is_file())
                self.assertTrue((epoch_dir / "predicted_group_heatmap.svg").is_file())
                self.assertTrue((epoch_dir / "_SUCCESS").is_file())

    def test_routerarena_input_cannot_be_the_training_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, evaluation, metadata, config = self._inputs(root)
            with self.assertRaisesRegex(
                RouterArenaEpochEvaluationError, "must not be the training"
            ):
                RouterArenaEpochEvaluator(
                    training_path=evaluation,
                    evaluation_path=evaluation,
                    metadata_path=metadata,
                    config_path=config,
                    local_output_dir=root / "local",
                    persistent_output_dir=root / "drive",
                    device="cpu",
                )

    def test_epoch_boundary_checkpoint_recovers_only_a_missing_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training, evaluation, metadata, config = self._inputs(root)
            evaluator = RouterArenaEpochEvaluator(
                training_path=training,
                evaluation_path=evaluation,
                metadata_path=metadata,
                config_path=config,
                local_output_dir=root / "local",
                persistent_output_dir=root / "drive",
                device="cpu",
            )
            checkpoint = root / "checkpoint-step-10"
            checkpoint.mkdir()
            (checkpoint / "training_state.pt").write_bytes(b"state")
            (checkpoint / "model.safetensors").write_bytes(b"model")
            seal_artifact_directory(
                checkpoint,
                manifest_name=CHECKPOINT_MANIFEST_NAME,
                metadata={
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "training_fingerprint": "training-run-a",
                    "optimizer_steps": 10,
                    "epoch_index": 1,
                    "next_batch_index": 0,
                },
            )
            predictor = StaticPredictor(["easy", "medium", "hard"])
            with patch(
                "llm_router_labeling.evaluation.routerarena_epoch_evaluation."
                "HuggingFaceRouterPredictor",
                return_value=predictor,
            ):
                self.assertTrue(evaluator.recover_epoch_from_checkpoint(checkpoint))
                self.assertFalse(evaluator.recover_epoch_from_checkpoint(checkpoint))
            self.assertEqual(predictor.calls, 1)


if __name__ == "__main__":
    unittest.main()
