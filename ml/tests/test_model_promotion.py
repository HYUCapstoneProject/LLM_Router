from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_router_labeling.deployment.deploy_gate import (
    DeployGateError,
    evaluate_deploy_gate,
)
from llm_router_labeling.deployment.model_package import (
    ModelPackageError,
    assemble_model_package,
    create_model_manifest,
    validate_model_package,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _checkpoint(directory: Path) -> None:
    _write_json(
        directory / "config.json",
        {
            "id2label": {"0": "easy", "1": "medium", "2": "hard"},
            "label2id": {"easy": 0, "medium": 1, "hard": 2},
        },
    )
    _write_json(directory / "tokenizer.json", {"version": "1.0"})
    _write_json(directory / "tokenizer_config.json", {})
    _write_json(directory / "metrics.json", {"routerbench": {"macro_f1": 0.81}})
    _write_json(directory / "training_config.json", {"max_length": 8192})
    (directory / "model.safetensors").write_bytes(b"synthetic model weights")


def _metadata() -> dict[str, object]:
    return {
        "model_version": "router-modernbert-test-v1",
        "base_model": "answerdotai/ModernBERT-base@immutable",
        "dataset_version": "routerbench-test-v1",
        "preprocessing_version": "routerbench-preprocessing-v1",
        "label_rule_version": "routerbench-proxy-v1",
        "training_version": "modernbert-training-v1",
        "evaluation_version": "router-evaluation-v1",
        "tokenizer_max_length": 8192,
        "code_commit": "0123456789abcdef",
    }


class ModelPackageTest(unittest.TestCase):
    def test_candidate_heads_cannot_be_promoted_as_difficulty_classifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            _checkpoint(checkpoint)
            for objective in ("pointwise", "pairwise"):
                with self.subTest(objective=objective):
                    # Even a mislabeled head must not silently enter the CE gate.
                    _write_json(
                        checkpoint / "config.json",
                        {
                            "id2label": {"0": "easy", "1": "medium", "2": "hard"},
                            "label2id": {"easy": 0, "medium": 1, "hard": 2},
                            "router_objective": objective,
                        },
                    )
                    with self.assertRaisesRegex(ModelPackageError, "only supports.*R0"):
                        create_model_manifest(checkpoint, _metadata())

    def test_manifest_records_and_verifies_required_file_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package_dir = Path(directory)
            _checkpoint(package_dir)
            manifest = create_model_manifest(package_dir, _metadata())
            self.assertEqual(manifest["gate_status"], "candidate")
            self.assertEqual(
                manifest["id2label"],
                {"0": "easy", "1": "medium", "2": "hard"},
            )
            report = validate_model_package(package_dir)
            self.assertTrue(report["valid"])
            self.assertIn("model.safetensors", report["verified_files"])

    def test_modified_package_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package_dir = Path(directory)
            _checkpoint(package_dir)
            create_model_manifest(package_dir, _metadata())
            (package_dir / "model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ModelPackageError, "SHA-256 mismatch"):
                validate_model_package(package_dir)

    def test_assemble_copies_checkpoint_and_evaluation_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_dir = root / "checkpoint"
            package_dir = root / "package"
            checkpoint_dir.mkdir()
            _checkpoint(checkpoint_dir)
            evaluation_metrics = root / "routerarena-metrics.json"
            _write_json(evaluation_metrics, {"overall": {"macro_f1": 0.78}})
            metadata = {**_metadata(), "gate_status": "approved"}
            manifest = assemble_model_package(
                checkpoint_dir,
                package_dir,
                metadata,
                evaluation_metrics,
            )
            self.assertEqual(manifest["gate_status"], "candidate")
            report = validate_model_package(package_dir)
            self.assertIn("routerarena_metrics.json", report["verified_files"])

    def test_assemble_overrides_static_metadata_with_the_run_candidate_version(
        self,
    ) -> None:
        # package_metadata_uri is one static file shared by every pipeline run, so
        # its own model_version cannot vary per run without this override.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_dir = root / "checkpoint"
            checkpoint_dir.mkdir()
            _checkpoint(checkpoint_dir)
            metadata = {**_metadata(), "model_version": "static-placeholder-in-gcs"}

            manifest = assemble_model_package(
                checkpoint_dir,
                root / "package",
                metadata,
                model_version="router-modernbert-base-feedback-20260901-a1b2c3d",
            )

            self.assertEqual(
                manifest["model_version"],
                "router-modernbert-base-feedback-20260901-a1b2c3d",
            )

    def test_gate_metrics_are_named_and_bound_to_their_evaluation_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            _checkpoint(checkpoint)
            metrics = root / "metrics.json"
            metadata = {
                **_metadata(),
                "evaluation_role": "continual_gate",
                "split_version": "fixed-split",
                "logical_model_version": "1.1",
            }
            _write_json(metrics, {"evaluation_role": "development_calibration"})
            with self.assertRaisesRegex(ModelPackageError, "role does not match"):
                assemble_model_package(checkpoint, root / "bad", metadata, metrics)
            _write_json(metrics, {"evaluation_role": "continual_gate", "macro_f1": 0.5})
            manifest = assemble_model_package(
                checkpoint, root / "good", metadata, metrics
            )
            self.assertEqual(manifest["evaluation_role"], "continual_gate")
            self.assertEqual(manifest["logical_model_version"], "1.1")
            self.assertIn("gate_metrics.json", manifest["files"])
            self.assertNotIn("routerarena_metrics.json", manifest["files"])
            validate_model_package(root / "good")


class DeployGateTest(unittest.TestCase):
    def test_contract_mismatch_rejects_good_scores_for_candidate_and_baseline(
        self,
    ) -> None:
        metrics = {
            "macro_f1": 0.99,
            "evaluation_role": "continual_gate",
            "dataset_sha256": "frozen-gate-sha",
            "dataset_contract": {"split_version": "frozen-split"},
        }
        rules = {
            "minimum_metrics": {"macro_f1": 0.8},
            "required_evaluation_contract": {
                "evaluation_role": "continual_gate",
                "dataset_sha256": "frozen-gate-sha",
                "dataset_contract.split_version": "frozen-split",
            },
        }
        for subject in ("candidate", "baseline"):
            with self.subTest(subject=subject):
                changed = {**metrics, "evaluation_role": "development_calibration"}
                result = evaluate_deploy_gate(
                    changed if subject == "candidate" else metrics,
                    rules,
                    candidate_model_version="candidate",
                    package_validation={"valid": True, "model_version": "candidate"},
                    baseline_metrics=changed if subject == "baseline" else metrics,
                    baseline_model_version="baseline",
                )
                self.assertFalse(result["passed"])
                self.assertEqual(
                    [check["name"] for check in result["failed_checks"]],
                    [f"{subject}.evaluation_contract.evaluation_role"],
                )
        missing = {**metrics, "dataset_contract": {}}
        with self.assertRaisesRegex(DeployGateError, "contract path not found"):
            evaluate_deploy_gate(
                missing,
                rules,
                candidate_model_version="candidate",
                package_validation={"valid": True, "model_version": "candidate"},
            )

    def test_absolute_and_baseline_rules_are_auditable(self) -> None:
        result = evaluate_deploy_gate(
            {
                "routerbench": {"macro_f1": 0.82, "under_routing_rate": 0.04},
                "runtime": {"p95_latency_ms": 42.0},
            },
            {
                "rules_version": "test-gate-v1",
                "minimum_metrics": {"routerbench.macro_f1": 0.80},
                "maximum_metrics": {
                    "routerbench.under_routing_rate": 0.05,
                    "runtime.p95_latency_ms": 50.0,
                },
                "minimum_delta": {"routerbench.macro_f1": -0.01},
            },
            candidate_model_version="candidate-v2",
            package_validation={"valid": True, "model_version": "candidate-v2"},
            baseline_metrics={"routerbench": {"macro_f1": 0.825}},
            baseline_model_version="approved-v1",
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["rules_version"], "test-gate-v1")
        self.assertEqual(result["failed_checks"], [])

    def test_failed_rules_are_preserved_with_reason_values(self) -> None:
        result = evaluate_deploy_gate(
            {"routerbench": {"macro_f1": 0.70}},
            {"minimum_metrics": {"routerbench.macro_f1": 0.80}},
            candidate_model_version="candidate-v2",
            package_validation={"valid": True, "model_version": "candidate-v2"},
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_checks"][0]["actual"], 0.70)
        self.assertEqual(result["failed_checks"][0]["threshold"], 0.80)

    def test_delta_rule_requires_baseline(self) -> None:
        with self.assertRaisesRegex(DeployGateError, "requires baseline"):
            evaluate_deploy_gate(
                {"routerbench": {"macro_f1": 0.82}},
                {"minimum_delta": {"routerbench.macro_f1": -0.01}},
                candidate_model_version="candidate-v2",
                package_validation={"valid": True, "model_version": "candidate-v2"},
            )


if __name__ == "__main__":
    unittest.main()
