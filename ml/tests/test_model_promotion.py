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


class DeployGateTest(unittest.TestCase):
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
            package_validation={"valid": True},
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
            package_validation={"valid": True},
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
                package_validation={"valid": True},
            )


if __name__ == "__main__":
    unittest.main()
