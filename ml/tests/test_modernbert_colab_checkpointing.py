from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

try:
    import torch
    import torch.nn.functional as torch_functional
except ImportError:
    torch = None
    torch_functional = None

from llm_router_labeling.training.modernbert_checkpointing import (
    CHECKPOINT_MANIFEST_NAME,
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointOptions,
    ModernBertCheckpointError,
    atomic_copy_checkpoint,
    completed_checkpoints,
    latest_completed_checkpoint,
    next_training_cursor,
    prune_checkpoints,
    seal_artifact_directory,
    training_fingerprint,
    verify_checkpoint,
)
from llm_router_labeling.training.modernbert_colab_training_cli import build_parser
from llm_router_labeling.training.modernbert_colab_training import (
    should_sync_checkpoint_to_drive,
)
from llm_router_labeling.training.modernbert_runtime import RuntimeContract
from llm_router_labeling.training.modernbert_training import (
    ID2LABEL,
    LABEL2ID,
    TrainingConfig,
    ValidationObservationState,
    run_training,
)
from llm_router_labeling.training.modernbert_validation_progress import (
    ModernBertValidationProgressError,
    RouterBenchValidationProgressTracker,
)


def make_checkpoint(root: Path, step: int, fingerprint: str = "fingerprint") -> Path:
    checkpoint = root / f"step-{step:08d}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "training_state.pt").write_bytes(f"state-{step}".encode())
    (checkpoint / "model.safetensors").write_bytes(f"model-{step}".encode())
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    seal_artifact_directory(
        checkpoint,
        manifest_name=CHECKPOINT_MANIFEST_NAME,
        metadata={
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "training_fingerprint": fingerprint,
            "optimizer_steps": step,
        },
    )
    return checkpoint


class ModernBertColabCheckpointingTest(unittest.TestCase):
    def test_training_fingerprint_is_stable_and_data_sensitive(self) -> None:
        first = training_fingerprint(
            training_data_sha256="data-a",
            config={"epochs": 3, "seed": 42},
            maximum_steps=None,
            runtime_version="runtime-v1",
        )
        reordered = training_fingerprint(
            training_data_sha256="data-a",
            config={"seed": 42, "epochs": 3},
            maximum_steps=None,
            runtime_version="runtime-v1",
        )
        changed = training_fingerprint(
            training_data_sha256="data-b",
            config={"epochs": 3, "seed": 42},
            maximum_steps=None,
            runtime_version="runtime-v1",
        )
        self.assertEqual(first, reordered)
        self.assertNotEqual(first, changed)

    def test_next_cursor_moves_to_next_epoch_only_after_last_batch(self) -> None:
        self.assertEqual(
            next_training_cursor(
                epoch_index=1, batch_index=2, batches_in_epoch=5
            ),
            (1, 3),
        )
        self.assertEqual(
            next_training_cursor(
                epoch_index=1, batch_index=4, batches_in_epoch=5
            ),
            (2, 0),
        )

    def test_only_sealed_checkpoints_are_discovered_and_pruned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_checkpoint(root, 10)
            make_checkpoint(root, 20)
            make_checkpoint(root, 30)
            incomplete = root / "step-00000040"
            incomplete.mkdir()
            (incomplete / "training_state.pt").write_bytes(b"partial")

            self.assertEqual(
                [path.name for path in completed_checkpoints(root)],
                ["step-00000010", "step-00000020", "step-00000030"],
            )
            self.assertEqual(latest_completed_checkpoint(root).name, "step-00000030")
            prune_checkpoints(root, keep_last=2)
            self.assertFalse((root / "step-00000010").exists())
            self.assertTrue((root / "step-00000020").exists())
            self.assertTrue(incomplete.exists())

    def test_atomic_copy_verifies_checksum_and_ignores_partial_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_checkpoint(root / "source", 50)
            destination = root / "drive" / source.name
            copied = atomic_copy_checkpoint(
                source, destination, expected_fingerprint="fingerprint"
            )
            self.assertEqual(copied, destination)
            verify_checkpoint(destination, expected_fingerprint="fingerprint")

            (destination / "model.safetensors").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ModernBertCheckpointError, "mismatch"):
                verify_checkpoint(destination)

    def test_checkpoint_rejects_different_training_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = make_checkpoint(Path(directory), 10, "run-a")
            with self.assertRaisesRegex(
                ModernBertCheckpointError, "fingerprint"
            ):
                verify_checkpoint(checkpoint, expected_fingerprint="run-b")

    def test_colab_cli_has_separate_persistent_resume_contract(self) -> None:
        args = build_parser().parse_args(
            [
                "--training-path",
                "training.parquet",
                "--persistent-run-dir",
                "/content/drive/MyDrive/LLM_Router/run-v1",
                "--save-every-optimizer-steps",
                "25",
                "--resume-mode",
                "required",
                "--drive-sync-every-optimizer-steps",
                "500",
                "--routerarena-evaluation-path",
                "routerarena.parquet",
                "--routerarena-batch-size",
                "4",
            ]
        )
        self.assertEqual(args.save_every_optimizer_steps, 25)
        self.assertEqual(args.keep_last_checkpoints, 2)
        self.assertEqual(args.resume_mode, "required")
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.drive_sync_every_optimizer_steps, 500)
        self.assertEqual(args.routerarena_evaluation_path, Path("routerarena.parquet"))
        self.assertEqual(args.routerarena_batch_size, 4)
        self.assertEqual(args.routerbench_validations_per_epoch, 3)

    def test_routerbench_validation_tracker_persists_progress_to_drive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracker = RouterBenchValidationProgressTracker(
                local_output_dir=root / "local",
                persistent_output_dir=root / "drive",
                points_per_epoch=3,
            )
            for point_number, step in enumerate((2, 3, 5), start=1):
                tracker.record(
                    ValidationObservationState(
                        epoch_number=1,
                        point_number=point_number,
                        points_per_epoch=3,
                        epoch_fraction=point_number / 3,
                        optimizer_steps=step,
                        training_fingerprint="run-fingerprint",
                        metrics={
                            "row_count": 3,
                            "accuracy": 0.2 * point_number,
                            "macro_f1": 0.1 * point_number,
                            "per_class": {},
                            "loss": 2.0 / point_number,
                            "weighted_loss": 1.8 / point_number,
                            "train_loss": 1.5 / point_number,
                            "train_weighted_loss": 1.3 / point_number,
                            "train_loss_row_count": 8,
                            "train_loss_optimizer_step_start_exclusive": (
                                0 if point_number == 1 else (2, 3)[point_number - 2]
                            ),
                            "train_loss_optimizer_step_end_inclusive": step,
                        },
                    )
                )

            progress = tracker.require_all(
                expected_epochs=1,
                training_fingerprint="run-fingerprint",
            )
            self.assertEqual(
                [row["optimizer_steps"] for row in progress["observations"]],
                [2, 3, 5],
            )
            self.assertTrue((root / "drive" / "validation_progress.svg").is_file())
            self.assertTrue((root / "drive" / "loss_progress.svg").is_file())
            self.assertAlmostEqual(
                progress["observations"][0]["train_loss"], 1.5
            )
            self.assertAlmostEqual(
                progress["observations"][0]["validation_loss"], 2.0
            )
            self.assertTrue(
                (root / "drive" / "observation-03" / "_SUCCESS").is_file()
            )
            incomplete_tracker = RouterBenchValidationProgressTracker(
                local_output_dir=root / "incomplete-local",
                persistent_output_dir=root / "incomplete-drive",
                points_per_epoch=3,
            )
            with self.assertRaisesRegex(
                ModernBertValidationProgressError, "new run directory"
            ):
                incomplete_tracker.require_through_optimizer_step(
                    optimizer_steps=4,
                    steps_per_epoch=4,
                    expected_epochs=3,
                    training_fingerprint="run-fingerprint",
                )

    def test_drive_sync_uses_500_steps_epoch_boundaries_and_final_step(self) -> None:
        base = {
            "next_batch_index": 17,
            "total_steps": 10_428,
        }
        self.assertFalse(
            should_sync_checkpoint_to_drive(
                {**base, "optimizer_steps": 450}, interval=500
            )
        )
        self.assertTrue(
            should_sync_checkpoint_to_drive(
                {**base, "optimizer_steps": 500}, interval=500
            )
        )
        self.assertTrue(
            should_sync_checkpoint_to_drive(
                {
                    **base,
                    "optimizer_steps": 3_476,
                    "next_batch_index": 0,
                },
                interval=500,
            )
        )
        self.assertTrue(
            should_sync_checkpoint_to_drive(
                {**base, "optimizer_steps": 10_428}, interval=500
            )
        )

    @unittest.skipUnless(torch is not None, "optional torch dependency is not installed")
    def test_interrupted_run_resumes_to_same_weights_as_uninterrupted_run(self) -> None:
        class TinyTokenizer:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

            def __call__(
                self,
                texts,
                *,
                return_tensors=None,
                **_kwargs,
            ):
                if isinstance(texts, str):
                    values = [max(1, len(texts) % 11)]
                    return {"input_ids": values}
                values = [[max(1, len(text) % 11)] for text in texts]
                if return_tensors == "pt":
                    return {"input_ids": torch.tensor(values, dtype=torch.float32)}
                return {"input_ids": values}

            def save_pretrained(self, directory: Path) -> None:
                (directory / "tokenizer_config.json").write_text(
                    "{}", encoding="utf-8"
                )

        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.classifier = torch.nn.Linear(1, 3)
                self.config = SimpleNamespace(id2label=ID2LABEL, label2id=LABEL2ID)

            @classmethod
            def from_pretrained(cls, source, **_kwargs):
                model = cls()
                source_path = Path(str(source))
                weights = source_path / "pytorch_model.bin"
                if weights.is_file():
                    model.load_state_dict(
                        torch.load(weights, map_location="cpu", weights_only=True)
                    )
                return model

            def gradient_checkpointing_enable(self) -> None:
                return None

            def forward(self, input_ids):
                return SimpleNamespace(logits=self.classifier(input_ids))

            def save_pretrained(
                self, directory: Path, *, safe_serialization: bool
            ) -> None:
                del safe_serialization
                directory.mkdir(parents=True, exist_ok=True)
                torch.save(self.state_dict(), directory / "pytorch_model.bin")
                (directory / "config.json").write_text(
                    json.dumps(
                        {
                            "id2label": {
                                str(index): label for index, label in ID2LABEL.items()
                            },
                            "label2id": LABEL2ID,
                        }
                    ),
                    encoding="utf-8",
                )

        class TinyScheduler:
            def __init__(self) -> None:
                self.steps = 0

            def step(self) -> None:
                self.steps += 1

            def state_dict(self):
                return {"steps": self.steps}

            def load_state_dict(self, state):
                self.steps = int(state["steps"])

        def scheduler_factory(*_args, **_kwargs):
            return TinyScheduler()

        backend = (
            torch,
            torch_functional,
            SimpleNamespace(__version__="test"),
            (TinyModel, TinyTokenizer),
            scheduler_factory,
        )
        runtime_report = {
            "runtime_version": "test-runtime",
            "contract_passed": True,
        }
        runtime_contract = RuntimeContract(
            runtime_version="test-runtime",
            allowed_python_minors=("3.12",),
            torch_version="test",
            transformers_version="test",
            numpy_version="test",
            pandas_version="test",
            pyarrow_version="test",
            cuda_version="test",
        )
        config = TrainingConfig(
            training_version="resume-test-v1",
            model_id="tiny-model",
            model_revision="immutable",
            train_split="train",
            validation_split="validation",
            max_length=16,
            truncation=True,
            train_batch_size=1,
            eval_batch_size=1,
            epochs=1,
            max_train_steps=4,
            gradient_accumulation_steps=1,
            gradient_checkpointing=False,
            learning_rate=0.01,
            weight_decay=0.0,
            warmup_ratio=0.0,
            seed=7,
            device="cpu",
            precision="fp32",
            attention_implementation="eager",
        )

        class PlannedInterruption(RuntimeError):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path = root / "training.parquet"
            pd.DataFrame(
                {
                    "sample_id": ["t1", "t2", "t3", "t4", "v1", "v2", "v3"],
                    "model_input_text": ["a", "bb", "ccc", "dddd", "e", "ff", "ggg"],
                    "difficulty_label": [
                        "easy",
                        "medium",
                        "hard",
                        "easy",
                        "easy",
                        "medium",
                        "hard",
                    ],
                    "sample_weight": [1.0] * 7,
                    "split": ["train"] * 4 + ["validation"] * 3,
                    "duplicate_group_id": [f"g{i}" for i in range(7)],
                }
            ).to_parquet(training_path, index=False)
            checkpoints = root / "checkpoints"

            def interrupt_after_step_two(checkpoint: Path) -> None:
                if checkpoint.name == "step-00000002":
                    raise PlannedInterruption

            common_patches = (
                patch(
                    "llm_router_labeling.training.modernbert_training._require_training_backends",
                    return_value=backend,
                ),
                patch(
                    "llm_router_labeling.training.modernbert_training.validate_runtime_contract",
                    return_value=runtime_report,
                ),
                patch(
                    "llm_router_labeling.training.modernbert_training.collect_runtime_versions",
                    return_value={},
                ),
            )
            with common_patches[0], common_patches[1], common_patches[2]:
                uninterrupted = run_training(
                    training_path,
                    root / "uninterrupted",
                    config,
                    runtime_contract=runtime_contract,
                    device_override="cpu",
                )

            with patch(
                "llm_router_labeling.training.modernbert_training._require_training_backends",
                return_value=backend,
            ), patch(
                "llm_router_labeling.training.modernbert_training.validate_runtime_contract",
                return_value=runtime_report,
            ), patch(
                "llm_router_labeling.training.modernbert_training.collect_runtime_versions",
                return_value={},
            ):
                with self.assertRaises(PlannedInterruption):
                    run_training(
                        training_path,
                        root / "interrupted",
                        config,
                        runtime_contract=runtime_contract,
                        device_override="cpu",
                        checkpoint_options=CheckpointOptions(
                            checkpoint_dir=checkpoints,
                            save_every_optimizer_steps=2,
                            on_checkpoint_saved=interrupt_after_step_two,
                        ),
                    )
                resumed = run_training(
                    training_path,
                    root / "resumed",
                    config,
                    runtime_contract=runtime_contract,
                    device_override="cpu",
                    checkpoint_options=CheckpointOptions(
                        checkpoint_dir=checkpoints,
                        save_every_optimizer_steps=2,
                        resume_from=checkpoints / "step-00000002",
                    ),
                )

            uninterrupted_weights = torch.load(
                root / "uninterrupted" / "pytorch_model.bin",
                map_location="cpu",
                weights_only=True,
            )
            resumed_weights = torch.load(
                root / "resumed" / "pytorch_model.bin",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(
                uninterrupted.metrics["optimization"]["optimizer_steps"], 4
            )
            self.assertEqual(resumed.metrics["optimization"]["resumed_from_step"], 2)
            self.assertAlmostEqual(
                uninterrupted.metrics["optimization"]["mean_train_loss"],
                resumed.metrics["optimization"]["mean_train_loss"],
                places=7,
            )
            for name in uninterrupted_weights:
                self.assertTrue(
                    torch.equal(uninterrupted_weights[name], resumed_weights[name]),
                    name,
                )

            three_epoch_config = replace(config, epochs=3, max_train_steps=12)
            observed_epochs: list[tuple[int, int, bool]] = []
            observed_validation_points: list[tuple[int, int, int, float]] = []
            observed_loss_pairs: list[tuple[float, float, int]] = []

            def observe_epoch(state) -> None:
                observed_epochs.append(
                    (state.epoch_number, state.optimizer_steps, state.model.training)
                )
                torch.rand(5)

            def observe_validation(state) -> None:
                observed_validation_points.append(
                    (
                        state.epoch_number,
                        state.point_number,
                        state.optimizer_steps,
                        state.epoch_fraction,
                    )
                )
                observed_loss_pairs.append(
                    (
                        state.metrics["train_loss"],
                        state.metrics["loss"],
                        state.metrics["train_loss_row_count"],
                    )
                )
                torch.rand(5)

            with patch(
                "llm_router_labeling.training.modernbert_training._require_training_backends",
                return_value=backend,
            ), patch(
                "llm_router_labeling.training.modernbert_training.validate_runtime_contract",
                return_value=runtime_report,
            ), patch(
                "llm_router_labeling.training.modernbert_training.collect_runtime_versions",
                return_value={},
            ):
                run_training(
                    training_path,
                    root / "three-epoch-control",
                    three_epoch_config,
                    runtime_contract=runtime_contract,
                    device_override="cpu",
                )
                run_training(
                    training_path,
                    root / "three-epoch-observed",
                    three_epoch_config,
                    runtime_contract=runtime_contract,
                    device_override="cpu",
                    checkpoint_options=CheckpointOptions(
                        checkpoint_dir=root / "three-epoch-checkpoints",
                        save_every_optimizer_steps=100,
                        keep_last_checkpoints=3,
                    ),
                    on_epoch_end=observe_epoch,
                    validation_observations_per_epoch=3,
                    on_validation_observation=observe_validation,
                )

            self.assertEqual(
                observed_epochs,
                [(1, 4, False), (2, 8, False), (3, 12, False)],
            )
            self.assertEqual(
                observed_validation_points,
                [
                    (1, 1, 1, 1 / 3),
                    (1, 2, 3, 2 / 3),
                    (1, 3, 4, 1.0),
                    (2, 1, 5, 1 / 3),
                    (2, 2, 7, 2 / 3),
                    (2, 3, 8, 1.0),
                    (3, 1, 9, 1 / 3),
                    (3, 2, 11, 2 / 3),
                    (3, 3, 12, 1.0),
                ],
            )
            self.assertEqual(len(observed_loss_pairs), 9)
            self.assertTrue(
                all(train_loss > 0 for train_loss, _, _ in observed_loss_pairs)
            )
            self.assertTrue(
                all(validation_loss > 0 for _, validation_loss, _ in observed_loss_pairs)
            )
            self.assertEqual(
                [row_count for _, _, row_count in observed_loss_pairs],
                [1, 2, 1] * 3,
            )
            control_weights = torch.load(
                root / "three-epoch-control" / "pytorch_model.bin",
                map_location="cpu",
                weights_only=True,
            )
            observed_weights = torch.load(
                root / "three-epoch-observed" / "pytorch_model.bin",
                map_location="cpu",
                weights_only=True,
            )
            for name in control_weights:
                self.assertTrue(
                    torch.equal(control_weights[name], observed_weights[name]),
                    name,
                )


if __name__ == "__main__":
    unittest.main()
