from __future__ import annotations

import json
import tempfile
import tomllib
import unittest
from pathlib import Path

import pandas as pd
import llm_router_labeling

from llm_router_labeling.training.modernbert_training import (
    ID2LABEL,
    LABEL2ID,
    ModernBertTrainingError,
    TrainingConfig,
    balanced_effective_batch_indices,
    classification_metrics,
    load_training_config,
    validate_checkpoint_contract,
    validate_training_dataframe,
)
from llm_router_labeling.training.modernbert_training_cli import build_parser
from llm_router_labeling.training.modernbert_runtime import (
    RuntimeContract,
    RuntimeContractError,
    validate_runtime_contract,
)


def valid_config() -> dict[str, object]:
    return {
        "training_version": "test-v1",
        "model_id": "answerdotai/ModernBERT-base",
        "model_revision": "immutable-revision",
        "label2id": LABEL2ID,
        "id2label": {str(index): label for index, label in ID2LABEL.items()},
        "max_length": 128,
        "truncation": True,
        "train_batch_size": 2,
        "eval_batch_size": 2,
        "epochs": 1,
        "max_train_steps": 1,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.0,
        "seed": 42,
        "device": "auto",
        "precision": "fp32",
        "attention_implementation": "eager",
    }


def training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["train-1", "train-2", "validation-1"],
            "model_input_text": ["easy question", "medium question", "hard question"],
            "difficulty_label": ["easy", "medium", "hard"],
            "sample_weight": [1.0, 0.5, 1.0],
            "split": ["train", "train", "validation"],
            "duplicate_group_id": ["group-1", "group-2", "group-3"],
        }
    )


class ModernBertTrainingTest(unittest.TestCase):
    def test_balanced_sampler_balances_every_optimizer_unit(self) -> None:
        dataframe = pd.DataFrame(
            {
                "difficulty_label": [
                    label for label in ("easy", "medium", "hard") for _ in range(12)
                ]
            }
        )
        first = balanced_effective_batch_indices(
            dataframe, effective_batch_size=8, seed=42
        )
        second = balanced_effective_batch_indices(
            dataframe, effective_batch_size=8, seed=42
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        for start in range(0, len(first), 8):
            counts = dataframe.loc[first[start : start + 8], "difficulty_label"].value_counts()
            padded = [int(counts.get(label, 0)) for label in ("easy", "medium", "hard")]
            self.assertLessEqual(max(padded) - min(padded), 1)

    def test_balanced_sampler_keeps_feedback_and_cycles_minority_classes(self) -> None:
        dataframe = pd.DataFrame(
            {
                "difficulty_label": ["easy"] * 4
                + ["medium"] * 5
                + ["hard"] * 7
            }
        )
        order = balanced_effective_batch_indices(
            dataframe, effective_batch_size=8, seed=42
        )
        sampled = dataframe.loc[order, "difficulty_label"]
        self.assertEqual(len(order), 21)
        self.assertEqual(
            sampled.value_counts().to_dict(),
            {"easy": 7, "medium": 7, "hard": 7},
        )
        self.assertEqual(set(range(len(dataframe))), set(order))
        for start in range(0, len(order), 8):
            counts = sampled.iloc[start : start + 8].value_counts()
            padded = [
                int(counts.get(label, 0))
                for label in ("easy", "medium", "hard")
            ]
            self.assertLessEqual(max(padded) - min(padded), 1)

    def test_top_level_routerbench_exports_remain_lazy_and_compatible(self) -> None:
        self.assertTrue(callable(llm_router_labeling.label_questions))

    def test_config_fixes_label_order_and_immutable_revision(self) -> None:
        config = TrainingConfig.from_mapping(valid_config())
        self.assertEqual(config.max_length, 128)
        self.assertFalse(config.gradient_checkpointing)
        changed = valid_config()
        changed["label2id"] = {"easy": 0, "hard": 1, "medium": 2}
        with self.assertRaisesRegex(ModernBertTrainingError, "easy=0"):
            TrainingConfig.from_mapping(changed)
        changed = valid_config()
        changed["model_revision"] = "main"
        with self.assertRaisesRegex(ModernBertTrainingError, "immutable"):
            TrainingConfig.from_mapping(changed)

    def test_training_projection_contract(self) -> None:
        validate_training_dataframe(training_frame())
        invalid = training_frame()
        invalid.loc[0, "sample_weight"] = 0.0
        with self.assertRaisesRegex(ModernBertTrainingError, "sample_weight"):
            validate_training_dataframe(invalid)

    def test_classification_metrics_include_all_three_labels(self) -> None:
        metrics = classification_metrics([0, 1, 2, 2], [0, 2, 2, 1])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(set(metrics["per_class"]), set(LABEL2ID))
        self.assertAlmostEqual(metrics["macro_f1"], 0.5)

    def test_checkpoint_contract_requires_explicit_label_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "config.json").write_text(
                json.dumps(
                    {
                        "id2label": {str(index): label for index, label in ID2LABEL.items()},
                        "label2id": LABEL2ID,
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (output_dir / "model.safetensors").write_bytes(b"test")
            contract = validate_checkpoint_contract(output_dir)
            self.assertEqual(contract["label2id"], LABEL2ID)

    def test_cli_uses_shared_file_contract(self) -> None:
        args = build_parser().parse_args(
            [
                "--training-path",
                "input.parquet",
                "--output-dir",
                "checkpoint",
                "--runtime-contract",
                "runtime.json",
                "--device",
                "cpu",
                "--initial-model-path",
                "approved-model",
                "--initialization-mode",
                "approved_anchor",
            ]
        )
        self.assertEqual(args.training_path, Path("input.parquet"))
        self.assertEqual(args.output_dir, Path("checkpoint"))
        self.assertEqual(args.runtime_contract, Path("runtime.json"))
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.initial_model_path, Path("approved-model"))
        self.assertEqual(args.initialization_mode, "approved_anchor")

    def test_runtime_contract_checks_core_versions_and_cuda(self) -> None:
        contract = RuntimeContract.from_mapping(
            {
                "runtime_version": "test-runtime-v1",
                "allowed_python_minors": ["3.11", "3.12"],
                "torch_version": "2.5.1",
                "transformers_version": "4.48.3",
                "numpy_version": "2.1.2",
                "pandas_version": "2.3.3",
                "pyarrow_version": "21.0.0",
                "cuda_version": "12.4",
            }
        )
        runtime = {
            "python_version": "3.12.13",
            "python_minor": "3.12",
            "torch_version": "2.5.1+cu124",
            "torch_base_version": "2.5.1",
            "transformers_version": "4.48.3",
            "numpy_version": "2.1.2",
            "pandas_version": "2.3.3",
            "pyarrow_version": "21.0.0",
            "cuda_version": "12.4",
            "cuda_available": True,
        }
        report = validate_runtime_contract(contract, runtime, device="cuda")
        self.assertTrue(report["contract_passed"])
        incompatible = {**runtime, "transformers_version": "5.14.1"}
        with self.assertRaisesRegex(RuntimeContractError, "transformers_version"):
            validate_runtime_contract(contract, incompatible, device="cuda")

    def test_t4_full_config_uses_native_context_and_memory_controls(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "modernbert_t4_full.json"
        )
        config = load_training_config(config_path)
        self.assertEqual(config.max_length, 8192)
        self.assertEqual(config.precision, "fp16")
        self.assertTrue(config.gradient_checkpointing)
        self.assertEqual(config.gradient_accumulation_steps, 8)

    def test_t4_v2_config_balances_each_effective_batch(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "modernbert_t4_balanced_v2.json"
        )
        config = load_training_config(config_path)
        self.assertEqual(config.train_batch_size, 1)
        self.assertEqual(config.gradient_accumulation_steps, 8)
        self.assertEqual(
            config.train_sampling_strategy, "balanced_effective_batch"
        )

    def test_training_image_package_exposes_only_training_cli(self) -> None:
        ml_root = Path(__file__).resolve().parents[1]
        package_config = tomllib.loads(
            (ml_root / "docker" / "modernbert-training.pyproject.toml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            package_config["project"]["scripts"],
            {
                "modernbert-train": (
                    "llm_router_labeling.training.modernbert_training_cli:main"
                )
            },
        )
        dockerfile = (
            ml_root / "docker" / "modernbert-training.Dockerfile"
        ).read_text(encoding="utf-8")
        self.assertNotIn("COPY ml/src /opt/llm-router/ml/src", dockerfile)
        self.assertNotIn("routerbench_cli.py", dockerfile)
        self.assertNotIn("routerarena", dockerfile)


if __name__ == "__main__":
    unittest.main()
