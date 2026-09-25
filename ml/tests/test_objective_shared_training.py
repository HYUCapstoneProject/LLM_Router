from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

try:
    import torch
    from torch.nn import functional
except ImportError:
    torch = None
    functional = None

from llm_router_labeling.objective_comparison.selection import (
    ObjectiveSelectionError,
    build_validation_lock,
)
from llm_router_labeling.objective_comparison.shared_training import (
    train_shared_objective,
)
from llm_router_labeling.shared.dataset_io import sha256_file, write_json
from llm_router_labeling.training.modernbert_checkpointing import (
    ARTIFACT_MANIFEST_NAME,
    seal_artifact_directory,
    verify_artifact_directory,
)
from llm_router_labeling.training.modernbert_runtime import RuntimeContract
from llm_router_labeling.training.modernbert_training import (
    ID2LABEL,
    LABEL2ID,
    TrainingConfig,
    run_training,
)


class PlannedInterruption(RuntimeError):
    pass


@unittest.skipUnless(torch is not None, "optional torch dependency is not installed")
class ObjectiveSharedTrainingIntegrationTest(unittest.TestCase):
    def test_import_r0_train_r1_resume_r2_and_seal_three_arm_lock(self) -> None:
        """Exercise real losses, optimizer, persistence, inference and locking offline."""
        seen_texts = []

        class TinyTokenizer:
            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

            def __call__(self, texts, *, return_tensors=None, **_kwargs):
                if isinstance(texts, str):
                    seen_texts.append(texts)
                    return {"input_ids": [max(1, len(texts) % 11)]}
                seen_texts.extend(texts)
                values = [[max(1, len(text) % 11)] for text in texts]
                if return_tensors == "pt":
                    return {"input_ids": torch.tensor(values, dtype=torch.float32)}
                return {"input_ids": values}

            def save_pretrained(self, directory):
                write_json(Path(directory) / "tokenizer_config.json", {})

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = torch.nn.Linear(1, 3)
                self.dropout = torch.nn.Dropout(0.2)
                self.config = SimpleNamespace(id2label=ID2LABEL, label2id=LABEL2ID)

            @classmethod
            def from_pretrained(cls, source, **kwargs):
                model = cls()
                if "id2label" in kwargs:
                    model.config.id2label = kwargs["id2label"]
                    model.config.label2id = kwargs["label2id"]
                path = Path(str(source))
                if str(source) == "offline-tiny-model":
                    return model
                values = json.loads((path / "config.json").read_text())
                values["id2label"] = {
                    int(key): value for key, value in values["id2label"].items()
                }
                model.config = SimpleNamespace(**values)
                model.load_state_dict(
                    torch.load(
                        path / "pytorch_model.bin",
                        map_location="cpu",
                        weights_only=True,
                    )
                )
                return model

            def gradient_checkpointing_enable(self):
                return None

            def forward(self, input_ids):
                return SimpleNamespace(logits=self.classifier(self.dropout(input_ids)))

            def save_pretrained(self, directory, *, safe_serialization):
                del safe_serialization
                path = Path(directory)
                path.mkdir(parents=True, exist_ok=True)
                torch.save(self.state_dict(), path / "pytorch_model.bin")
                write_json(path / "config.json", vars(self.config))

        class TinyScheduler:
            def __init__(self):
                self.steps = 0

            def step(self):
                self.steps += 1

            def state_dict(self):
                return {"steps": self.steps}

            def load_state_dict(self, state):
                self.steps = state["steps"]

        def scheduler_factory(*_args, **_kwargs):
            return TinyScheduler()

        config = TrainingConfig(
            training_version="tiny-r0-full-budget",
            model_id="offline-tiny-model",
            model_revision="immutable-fixture-revision",
            train_split="train",
            validation_split="validation",
            max_length=16,
            truncation=True,
            train_batch_size=1,
            eval_batch_size=2,
            epochs=1,
            max_train_steps=None,
            gradient_accumulation_steps=1,
            gradient_checkpointing=False,
            learning_rate=0.01,
            weight_decay=0.0,
            warmup_ratio=0.0,
            seed=42,
            device="cpu",
            precision="fp32",
            attention_implementation="eager",
            gradient_clip_norm=1.0,
        )
        runtime = RuntimeContract(
            runtime_version="offline-test-runtime",
            allowed_python_minors=("3.12",),
            torch_version="test",
            transformers_version="test",
            numpy_version="test",
            pandas_version="test",
            pyarrow_version="test",
            cuda_version="test",
        )
        runtime_versions = {
            "python_minor": "3.12",
            "torch_base_version": "test",
            "transformers_version": "test",
            "numpy_version": "test",
            "pandas_version": "test",
            "pyarrow_version": "test",
            "cuda_version": None,
            "cuda_available": False,
        }
        backend = (
            torch,
            functional,
            SimpleNamespace(__version__="test"),
            (TinyModel, TinyTokenizer),
            scheduler_factory,
        )
        with tempfile.TemporaryDirectory() as directory, ExitStack() as patches:
            root = Path(directory)
            patches.enter_context(
                patch(
                    "llm_router_labeling.training.modernbert_training._require_training_backends",
                    return_value=backend,
                )
            )
            patches.enter_context(
                patch(
                    "llm_router_labeling.training.modernbert_training.collect_runtime_versions",
                    return_value=runtime_versions,
                )
            )
            patches.enter_context(
                patch(
                    "llm_router_labeling.objective_comparison.shared_training._require_transformers",
                    return_value=(TinyModel, TinyTokenizer, scheduler_factory),
                )
            )
            training = pd.DataFrame(
                {
                    "sample_id": [f"s{i}" for i in range(7)],
                    "model_input_text": [
                        "a",
                        "bb",
                        "ccc",
                        "dddd",
                        "eeeee",
                        "ffffff",
                        "ggggggg",
                    ],
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
                    "source_dataset": [
                        "routerbench",
                        "xroutebench",
                        "routerbench",
                        "xroutebench",
                        "routerbench",
                        "xroutebench",
                        "routerbench",
                    ],
                    "category": ["fixture-category"] * 7,
                    "dataset_version": ["fixture-base-v1"] * 7,
                    "label_rule_version": ["fixture-T1"] * 7,
                    "preprocessing_version": ["fixture-structure-v2"] * 7,
                    "split_version": ["fixture-split"] * 7,
                }
            )
            training_path = root / "training.parquet"
            training.to_parquet(training_path, index=False)
            outcomes = training.copy()
            for tier, quality, cost in (
                ("low", 0.0, 1.0),
                ("medium", 0.5, 3.0),
                ("high", 1.0, 10.0),
            ):
                outcomes[f"quality_{tier}"] = quality
                outcomes[f"success_{tier}"] = float(quality >= 0.5)
                outcomes[f"actual_cost_{tier}"] = cost
                outcomes[f"decision_cost_{tier}"] = (cost - 1.0) / 9.0
            outcome_dir = root / "outcomes"
            outcome_dir.mkdir()
            for split in ("train", "validation"):
                outcomes.loc[outcomes["split"].eq(split)].to_parquet(
                    outcome_dir / f"{split}.parquet", index=False
                )
            write_json(outcome_dir / "arms.json", {"selection_split": "train_only"})
            pd.DataFrame({"training_row_count": [4]}).to_parquet(
                outcome_dir / "candidate-statistics.parquet", index=False
            )
            manifest = {
                "base_dataset_manifest_sha256": "frozen-base-manifest",
                "dataset_version": "fixture-outcomes-v1",
                "experiment_version": "fixture-A",
                "artifacts": {
                    name: {"sha256": sha256_file(outcome_dir / name)}
                    for name in [
                        "train.parquet",
                        "validation.parquet",
                        "arms.json",
                        "candidate-statistics.parquet",
                    ]
                },
            }
            write_json(outcome_dir / "manifest.json", manifest)
            (outcome_dir / "_SUCCESS").write_text(
                sha256_file(outcome_dir / "manifest.json")
            )
            experiment = {
                "experiment_version": "fixture-A",
                "dataset_version": "fixture-outcomes-v1",
                "base_dataset_manifest_sha256": "frozen-base-manifest",
                "base_training_sha256": sha256_file(training_path),
                "methods": ["ordinal", "pointwise", "pairwise"],
                "seed": 42,
                "finalist_seeds": [],
                "training": {**asdict(config), "pairwise_tie_weight": 0.25},
                "backbone": {
                    "model_id": config.model_id,
                    "revision": config.model_revision,
                    "max_length": config.max_length,
                },
                "selection": {"canonical_only": True, "require_contract_match": True},
                "evaluation": {
                    "penalty_grid": [0.0, 1.0, 100.0],
                    "relative_cost_budgets": [0.0, 0.5, 1.0],
                    "quality_retention_targets": [0.95],
                    "selection_metric": "source_macro_quality_cost_nauc",
                    "default_relative_cost_budget": 0.5,
                },
            }
            write_json(root / "experiment.json", experiment)
            write_json(
                root / "training.json",
                {**asdict(config), "id2label": ID2LABEL, "label2id": LABEL2ID},
            )
            write_json(root / "runtime.json", asdict(runtime))

            original = run_training(
                training_path, root / "original-r0", config, runtime_contract=runtime
            )
            seal_artifact_directory(
                original.output_dir,
                manifest_name=ARTIFACT_MANIFEST_NAME,
                metadata={
                    "training_fingerprint": original.metrics["optimization"][
                        "training_fingerprint"
                    ]
                },
            )
            imported_weights_sha = sha256_file(
                original.output_dir / "pytorch_model.bin"
            )
            common = {
                "base_training_path": training_path,
                "outcome_dir": outcome_dir,
                "experiment_config_path": root / "experiment.json",
                "training_config_path": root / "training.json",
                "runtime_config_path": root / "runtime.json",
            }
            results = {}
            results["ordinal"] = train_shared_objective(
                method="ordinal",
                output_dir=root / "runs/ordinal",
                import_r0_dir=original.output_dir,
                **common,
            )
            self.assertEqual(
                sha256_file(original.output_dir / "pytorch_model.bin"),
                imported_weights_sha,
            )
            results["pointwise"] = train_shared_objective(
                method="pointwise", output_dir=root / "runs/pointwise", **common
            )
            # Service feedback contains only explicitly observed candidates. Exercise
            # the real shared trainer with missing targets and no synthetic CE labels.
            partial = outcomes.drop(columns=["difficulty_label"]).copy()
            partial["observed_low"] = 1
            partial["observed_medium"] = 0
            partial["observed_high"] = 1
            # Null-only Arrow columns deserialize as None; this matches an
            # actual sparse export where no medium candidate was observed.
            partial["success_medium"] = None
            partial_path = root / "partial-r1.parquet"
            partial.to_parquet(partial_path, index=False)
            masked = run_training(
                partial_path,
                root / "partial-r1-model",
                config,
                runtime_contract=runtime,
                objective_method="pointwise",
            )
            self.assertEqual(masked.metrics["data"]["label_distribution"], None)
            self.assertEqual(
                masked.metrics["data"]["observed_candidate_counts"],
                {
                    "observed_low": 7,
                    "observed_medium": 0,
                    "observed_high": 7,
                },
            )
            self.assertEqual(masked.metrics["optimization"]["optimizer_steps"], 4)
            self.assertNotEqual(
                masked.metrics["optimization"]["training_fingerprint"],
                original.metrics["optimization"]["training_fingerprint"],
            )
            # A rehearsal merge can retain old difficulty metadata on base rows
            # and null metadata on new feedback. R1 never supervises with it.
            partial["difficulty_label"] = ["easy"] + [None] * (len(partial) - 1)
            partial.to_parquet(partial_path, index=False)
            mixed = run_training(
                partial_path,
                root / "partial-r1-mixed-metadata",
                config,
                runtime_contract=runtime,
                objective_method="pointwise",
            )
            self.assertEqual(mixed.metrics["optimization"]["optimizer_steps"], 4)
            lock_args = {
                "run_roots": [root / "runs"],
                "experiment_config_path": root / "experiment.json",
                "output_path": root / "lock.json",
                "stage": "final",
            }
            with self.assertRaisesRegex(ObjectiveSelectionError, "Missing canonical"):
                build_validation_lock(**lock_args)

            def interrupt_after_checkpoint(_checkpoint):
                raise PlannedInterruption(
                    "Final checkpoint was saved before transfer failure"
                )

            with self.assertRaises(PlannedInterruption):
                train_shared_objective(
                    method="pairwise",
                    output_dir=root / "runs/pairwise",
                    on_checkpoint_saved=interrupt_after_checkpoint,
                    **common,
                )
            self.assertFalse((root / "runs/pairwise/_SUCCESS").exists())
            results["pairwise"] = train_shared_objective(
                method="pairwise", output_dir=root / "runs/pairwise", **common
            )
            self.assertEqual(
                results["pairwise"]["metrics"]["training_metrics"]["optimization"][
                    "resumed_from_step"
                ],
                4,
            )

            self.assertEqual(set(seen_texts), set(training["model_input_text"]))
            projections = set()
            for method, result in results.items():
                run = root / "runs" / method
                manifest = result["manifest"]
                metrics = result["metrics"]
                projections.add(manifest["training_data_sha256"])
                self.assertEqual(metrics["optimizer_steps"], 4)
                self.assertEqual(metrics["train_rows"], 4)
                self.assertEqual(metrics["validation_rows"], 3)
                self.assertEqual(metrics["batch_1_latency_ms"]["sample_count"], 3)
                self.assertEqual(
                    set(metrics["validation_budget_penalties"]),
                    {"routerbench", "xroutebench"},
                )
                self.assertFalse(manifest["test_data_accessed"])
                self.assertFalse(manifest["routerarena_accessed"])
                self.assertEqual(
                    (run / "_SUCCESS").read_text().strip(),
                    sha256_file(run / "manifest.json"),
                )
                model_dir = Path(manifest["model_dir"])
                verify_artifact_directory(
                    model_dir, manifest_name=ARTIFACT_MANIFEST_NAME
                )
                model_config = json.loads((model_dir / "config.json").read_text())
                expected_labels = (
                    ["easy", "medium", "hard"]
                    if method == "ordinal"
                    else ["low", "medium", "high"]
                )
                self.assertEqual(
                    [model_config["id2label"][str(i)] for i in range(3)],
                    expected_labels,
                )
                if method != "ordinal":
                    self.assertEqual(model_config["router_objective"], method)
                    self.assertNotIn(
                        "accuracy", metrics["training_metrics"]["validation"]
                    )
            self.assertEqual(len(projections), 1)
            lock = build_validation_lock(**lock_args)
            self.assertEqual(set(lock["methods"]), {"ordinal", "pointwise", "pairwise"})
            self.assertIn(lock["winner_method"], lock["methods"])
            self.assertEqual(lock["selection_metric"], "source_macro_quality_cost_nauc")
            self.assertEqual(lock["seed_scope"], "canonical_only")
            self.assertFalse(lock["production_activation_authorized"])
            self.assertEqual(build_validation_lock(**lock_args), lock)


if __name__ == "__main__":
    unittest.main()
