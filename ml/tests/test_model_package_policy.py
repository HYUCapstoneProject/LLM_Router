from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from llm_router_labeling.deployment.model_package import (
    EXPECTED_ID2LABEL,
    ModelPackageError,
    assemble_model_package,
    create_model_manifest,
    validate_model_package,
)
from llm_router_labeling.deployment.model_package_cli import build_parser
from llm_router_labeling.shared.dataset_io import sha256_file
from test_model_promotion import _checkpoint, _metadata, _write_json
from test_routing_policy import fixture_policy_payload


def policy_checkpoint(directory: Path, objective: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _checkpoint(directory)
    labels = (
        ("easy", "medium", "hard")
        if objective == "ordinal"
        else ("low", "medium", "high")
    )
    config = {
        "id2label": {str(i): label for i, label in enumerate(labels)},
        "label2id": {label: i for i, label in enumerate(labels)},
    }
    if objective != "ordinal":
        config["router_objective"] = objective
    _write_json(directory / "config.json", config)
    _write_json(
        directory / "training_config.json",
        {"max_length": 512, "objective_method": objective},
    )


def policy_metadata(policy_path: Path, objective: str) -> dict:
    return {
        **_metadata(),
        "schema_version": "model-package-v2",
        "objective": objective,
        "tokenizer_max_length": 512,
        "input_contract": "structure-preserving-v2",
        "routing_policy_file": "routing_policy.json",
        "routing_policy_sha256": sha256_file(policy_path),
    }


class ModelPackagePolicyTest(unittest.TestCase):
    def test_v2_assembles_all_objectives_with_actual_labels_and_preserves_sources(
        self,
    ) -> None:
        for objective in ("ordinal", "pointwise", "pairwise"):
            with (
                self.subTest(objective=objective),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                checkpoint = root / "checkpoint"
                policy_checkpoint(checkpoint, objective)
                source_hashes = {
                    path.name: sha256_file(path) for path in checkpoint.iterdir()
                }
                policy = root / "policy.json"
                _write_json(policy, fixture_policy_payload(objective))
                package = root / "package"
                metadata = policy_metadata(policy, objective)
                manifest = assemble_model_package(
                    checkpoint, package, metadata, routing_policy_path=policy
                )
                self.assertEqual(manifest["schema_version"], "model-package-v2")
                self.assertEqual(manifest["gate_status"], "candidate")
                self.assertEqual(manifest["objective"], objective)
                self.assertEqual(manifest["routing_policy_sha256"], sha256_file(policy))
                self.assertEqual(
                    manifest["files"]["routing_policy.json"], sha256_file(policy)
                )
                self.assertEqual(
                    manifest["checkpoint_config_sha256"], source_hashes["config.json"]
                )
                self.assertEqual(
                    {path.name: sha256_file(path) for path in checkpoint.iterdir()},
                    source_hashes,
                )
                for filename in (
                    "model.safetensors",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ):
                    self.assertEqual(
                        sha256_file(package / filename), source_hashes[filename]
                    )
                output_config = json.loads((package / "config.json").read_text())
                self.assertEqual(output_config["router_objective"], objective)
                self.assertEqual(manifest["id2label"], output_config["id2label"])
                report = validate_model_package(package)
                self.assertEqual(report["id2label"], manifest["id2label"])
                self.assertEqual(
                    report["deployment_readiness"], "candidate_contract_only"
                )
                self.assertFalse(report["automatic_promotion_supported"])
                if objective == "ordinal":
                    self.assertNotEqual(
                        manifest["files"]["config.json"], source_hashes["config.json"]
                    )
                else:
                    self.assertEqual(
                        manifest["files"]["config.json"], source_hashes["config.json"]
                    )
                    self.assertEqual(
                        manifest["id2label"], {"0": "low", "1": "medium", "2": "high"}
                    )
        # The unchanged CURRENT pointer uses these service tier aliases.
        self.assertEqual(EXPECTED_ID2LABEL, {"0": "easy", "1": "medium", "2": "hard"})

    def test_content_copy_works_when_destination_rejects_posix_metadata(self) -> None:
        import errno
        from unittest.mock import patch

        for schema in ("model-package-v1", "model-package-v2"):
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint = root / "checkpoint"
                policy_checkpoint(checkpoint, "ordinal")
                policy = root / "policy.json"
                _write_json(policy, fixture_policy_payload("ordinal"))
                metadata = (
                    policy_metadata(policy, "ordinal")
                    if schema == "model-package-v2" else _metadata()
                )
                evaluation = root / "evaluation.json"
                _write_json(evaluation, {"accuracy": 0.7})
                package = root / "package"
                with patch("os.utime", side_effect=PermissionError(errno.EPERM, "unsupported")):
                    manifest = assemble_model_package(
                        checkpoint, package, metadata,
                        evaluation_metrics_path=evaluation,
                        routing_policy_path=policy if schema == "model-package-v2" else None,
                    )
                for name in ("model.safetensors", "tokenizer.json", "training_config.json"):
                    self.assertEqual((package / name).read_bytes(), (checkpoint / name).read_bytes())
                    self.assertEqual(manifest["files"][name], sha256_file(checkpoint / name))
                self.assertEqual(manifest["files"]["routerarena_metrics.json"], sha256_file(evaluation))
                if schema == "model-package-v2":
                    self.assertEqual(manifest["files"]["routing_policy.json"], sha256_file(policy))
                validate_model_package(package)
                (package / "model.safetensors").write_bytes(b"corrupted")
                with self.assertRaises(ModelPackageError):
                    validate_model_package(package)

    def test_v1_default_is_preserved_and_policy_cannot_silently_enter_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            _checkpoint(checkpoint)
            policy = root / "policy.json"
            _write_json(policy, fixture_policy_payload("ordinal"))
            manifest = assemble_model_package(checkpoint, root / "v1", _metadata())
            self.assertEqual(manifest["schema_version"], "model-package-v1")
            self.assertNotIn("objective", manifest)
            self.assertNotIn("routing_policy.json", manifest["files"])
            with self.assertRaisesRegex(ModelPackageError, "explicit model-package-v2"):
                assemble_model_package(
                    checkpoint, root / "bad", _metadata(), routing_policy_path=policy
                )

    def test_v2_requires_explicit_metadata_and_policy_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            policy_checkpoint(checkpoint, "pairwise")
            policy = root / "policy.json"
            _write_json(policy, fixture_policy_payload("pairwise"))
            metadata = policy_metadata(policy, "pairwise")
            with self.assertRaisesRegex(
                ModelPackageError, "explicit routing_policy_path"
            ):
                assemble_model_package(checkpoint, root / "no-policy", metadata)
            for field in (
                "objective",
                "input_contract",
                "routing_policy_file",
                "routing_policy_sha256",
            ):
                altered = {
                    key: value for key, value in metadata.items() if key != field
                }
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(ModelPackageError, "missing fields"),
                ):
                    assemble_model_package(
                        checkpoint, root / field, altered, routing_policy_path=policy
                    )

    def test_mismatched_checkpoint_policy_and_input_contracts_are_rejected(
        self,
    ) -> None:
        changes = (
            "objective",
            "labels",
            "length",
            "input_contract",
            "training_objective",
            "policy_objective",
            "canonical_instead_of_file_sha",
            "source_local",
            "source_config_sha",
        )
        for change in changes:
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                checkpoint = root / "checkpoint"
                policy_checkpoint(checkpoint, "pointwise")
                payload = fixture_policy_payload("pointwise")
                policy = root / "policy.json"
                _write_json(policy, payload)
                metadata = policy_metadata(policy, "pointwise")
                if change in {"objective", "labels"}:
                    config = json.loads((checkpoint / "config.json").read_text())
                    if change == "objective":
                        config["router_objective"] = "pairwise"
                    else:
                        config.update(
                            id2label={"0": "easy", "1": "medium", "2": "hard"},
                            label2id={"easy": 0, "medium": 1, "hard": 2},
                        )
                    _write_json(checkpoint / "config.json", config)
                elif change in {"length", "input_contract", "training_objective"}:
                    training = {"max_length": 512, "objective_method": "pointwise"}
                    training.update(
                        {"max_length": 8192}
                        if change == "length"
                        else {"input_contract": "legacy-word-only-v1"}
                        if change == "input_contract"
                        else {"objective_method": "pairwise"}
                    )
                    _write_json(checkpoint / "training_config.json", training)
                elif change == "policy_objective":
                    _write_json(policy, fixture_policy_payload("pairwise"))
                    metadata["routing_policy_sha256"] = sha256_file(policy)
                elif change == "canonical_instead_of_file_sha":
                    metadata["routing_policy_sha256"] = payload["policy_sha256"]
                elif change == "source_local":
                    payload["scope"] = "source-local"
                    _write_json(policy, payload)
                    metadata["routing_policy_sha256"] = sha256_file(policy)
                elif change == "source_config_sha":
                    metadata["checkpoint_config_sha256"] = "f" * 64
                with self.assertRaises(ModelPackageError):
                    assemble_model_package(
                        checkpoint,
                        root / "output",
                        metadata,
                        routing_policy_path=policy,
                    )

    def test_policy_file_digest_and_metadata_remain_bound_during_validation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            policy_checkpoint(checkpoint, "pairwise")
            policy = root / "policy.json"
            _write_json(policy, fixture_policy_payload("pairwise"))
            package = root / "package"
            original = assemble_model_package(
                checkpoint,
                package,
                policy_metadata(policy, "pairwise"),
                routing_policy_path=policy,
            )
            for mutation in (
                "missing_file_entry",
                "wrong_file_entry",
                "wrong_policy_digest",
                "wrong_output_labels",
                "wrong_original_sha",
            ):
                manifest = copy.deepcopy(original)
                if mutation == "missing_file_entry":
                    del manifest["files"]["routing_policy.json"]
                elif mutation == "wrong_file_entry":
                    manifest["files"]["routing_policy.json"] = "f" * 64
                elif mutation == "wrong_policy_digest":
                    manifest["routing_policy_sha256"] = "f" * 64
                elif mutation == "wrong_output_labels":
                    manifest["id2label"] = EXPECTED_ID2LABEL
                else:
                    del manifest["checkpoint_config_sha256"]
                _write_json(package / "model_manifest.json", manifest)
                with (
                    self.subTest(mutation=mutation),
                    self.assertRaises(ModelPackageError),
                ):
                    validate_model_package(package)
            _write_json(package / "model_manifest.json", original)
            payload = json.loads((package / "routing_policy.json").read_text())
            payload["lambda_cost"] = 0.5
            _write_json(package / "routing_policy.json", payload)
            with self.assertRaisesRegex(ModelPackageError, "SHA-256 mismatch"):
                validate_model_package(package)

    def test_v2_producer_cannot_claim_an_approved_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy_checkpoint(root, "pointwise")
            policy = root / "routing_policy.json"
            _write_json(policy, fixture_policy_payload("pointwise"))
            metadata = {
                **policy_metadata(policy, "pointwise"),
                "gate_status": "approved",
            }
            with self.assertRaisesRegex(ModelPackageError, "only prepares candidates"):
                create_model_manifest(root, metadata)

    def test_cli_exposes_explicit_policy_file_for_assembly(self) -> None:
        args = build_parser().parse_args(
            [
                "assemble",
                "--checkpoint-dir",
                "checkpoint",
                "--metadata",
                "metadata.json",
                "--output-dir",
                "package",
                "--routing-policy",
                "service-policy.json",
            ]
        )
        self.assertEqual(args.routing_policy, Path("service-policy.json"))


if __name__ == "__main__":
    unittest.main()
