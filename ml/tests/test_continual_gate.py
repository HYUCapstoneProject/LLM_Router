from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from llm_router_labeling.evaluation import routerarena_evaluation_cli
from llm_router_labeling.evaluation.continual_gate import (
    evaluate_continual_predictions,
    run_continual_evaluation,
    validate_continual_evaluation,
)
from llm_router_labeling.shared.dataset_io import sha256_file


class ContinualGateTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (
                Path(__file__).resolve().parents[1] / "config/continual_gate_v1.json"
            ).read_text()
        )
        self.frame = pd.DataFrame(
            [
                {
                    "sample_id": f"s{i}",
                    "model_input_text": f"question {i}",
                    "difficulty_label": label,
                    "duplicate_group_id": f"g{i}",
                    "source_dataset": source,
                    "category": category,
                    "split": "gate",
                    **self.config["dataset_contract"],
                }
                for i, (source, category, label) in enumerate(
                    (
                        ("routerbench", "math", "easy"),
                        ("routerbench", "math", "medium"),
                        ("routerbench", "math", "hard"),
                        ("xroutebench", "gsm8k", "easy"),
                        ("xroutebench", "gsm8k", "medium"),
                        ("xroutebench", "gsm8k", "hard"),
                    )
                )
            ]
        )
        self.output = pd.DataFrame(
            {"predicted_label": ["easy", "medium", "easy", "medium", "hard", "hard"]}
        )

    def test_source_domain_and_hard_errors_use_actual_labels(self):
        result = evaluate_continual_predictions(self.frame, self.output, self.config)
        self.assertEqual(result.metrics["accuracy"], 0.5)
        self.assertEqual(result.metrics["per_class"]["hard"]["recall"], 0.5)
        self.assertEqual(result.metrics["hard_to_easy_rate"], 0.5)
        self.assertEqual(result.metrics["hard_to_easy_count"], 1)
        self.assertEqual(
            result.metrics["per_source_dataset"]["routerbench"]["accuracy"], 2 / 3
        )
        self.assertEqual(result.metrics["per_domain"]["math"]["row_count"], 6)
        self.assertFalse(any("routerarena" in key for key in result.metrics))
        self.assertIn("difficulty_label", result.predictions)
        self.assertNotIn("routerarena_label", result.predictions)

    def test_wrong_partition_contract_or_role_is_rejected_before_inference(self):
        class NeverPredict:
            def predict(self, texts):
                raise AssertionError("Inference must not start")

        for column in ("split", *self.config["dataset_contract"]):
            with self.subTest(column=column):
                changed = self.frame.copy()
                changed[column] = "wrong-contract"
                with self.assertRaises(ValueError):
                    run_continual_evaluation(changed, NeverPredict(), self.config)
        changed_config = {**self.config, "evaluation_role": "development_calibration"}
        with self.assertRaisesRegex(ValueError, "role and partition"):
            run_continual_evaluation(self.frame, NeverPredict(), changed_config)

    def test_unknown_domain_and_missing_source_are_rejected(self):
        changed = self.frame.copy()
        changed.loc[0, "category"] = "unregistered"
        with self.assertRaisesRegex(ValueError, "Domain mapping"):
            validate_continual_evaluation(changed, self.config)
        with self.assertRaisesRegex(ValueError, "RouterBench and xRouteBench"):
            validate_continual_evaluation(self.frame.iloc[:3], self.config)

    def test_prediction_output_cannot_overwrite_ground_truth(self):
        changed = self.output.copy()
        changed["difficulty_label"] = "easy"
        with self.assertRaisesRegex(ValueError, "overwrite"):
            evaluate_continual_predictions(self.frame, changed, self.config)
        with self.assertRaisesRegex(ValueError, "rows"):
            evaluate_continual_predictions(
                self.frame, self.output.iloc[:-1], self.config
            )

    def test_only_input_text_reaches_predictor(self):
        case = self

        class Predictor:
            def predict(self, texts):
                case.assertEqual(texts, [f"question {i}" for i in range(6)])
                return case.output

            def info(self):
                return {"max_length": 512}

        frame = self.frame.assign(overall_solve_rate=0.5, answer="not a model input")
        result = run_continual_evaluation(frame, Predictor(), self.config)
        self.assertNotIn("answer", result.predictions)
        self.assertNotIn("overall_solve_rate", result.predictions)

    def test_cli_rejects_wrong_dataset_hash_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "gate.parquet"
            self.frame.to_parquet(path, index=False)
            config_path = root / "config.json"
            config = {**self.config, "evaluation_dataset_sha256": sha256_file(path)}
            config_path.write_text(json.dumps(config))
            changed = self.frame.copy()
            changed.loc[0, "model_input_text"] = (
                "tampered text, same partition and contract"
            )
            changed.to_parquet(path, index=False)
            args = argparse.Namespace(
                config=config_path,
                evaluation_path=path,
                metadata_path=None,
                model_path=str(root / "checkpoint"),
                output_dir=root / "output",
                max_length=None,
                batch_size=None,
                device=None,
                observability_run_uri=None,
                batch_id=None,
                candidate_model_version=None,
            )
            with (
                patch.object(argparse.ArgumentParser, "parse_args", return_value=args),
                patch.object(
                    routerarena_evaluation_cli, "HuggingFaceRouterPredictor"
                ) as predictor,
                self.assertRaisesRegex(ValueError, "checksum mismatch"),
            ):
                routerarena_evaluation_cli.main()
            predictor.assert_not_called()
            self.assertFalse(args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
