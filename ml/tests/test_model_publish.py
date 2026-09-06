from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_router_labeling.deployment.model_package import create_model_manifest
from llm_router_labeling.deployment.model_publish import (
    ModelPublishError,
    approve_candidate,
    approved_uri,
    candidate_uri,
    publish_candidate,
    read_current_pointer,
    rollback_current_pointer,
)
from llm_router_labeling.shared.object_store import LocalObjectStore

ROOT_URI = "gs://llm-router/models/router-modernbert"
MODEL_VERSION = "router-modernbert-test-v1"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _package(directory: Path, *, model_version: str = MODEL_VERSION) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
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
    create_model_manifest(
        directory,
        {
            "model_version": model_version,
            "base_model": "answerdotai/ModernBERT-base@immutable",
            "dataset_version": "routerbench-test-v1",
            "preprocessing_version": "routerbench-preprocessing-v1",
            "label_rule_version": "routerbench-proxy-v1",
            "training_version": "modernbert-training-v1",
            "evaluation_version": "router-evaluation-v1",
            "tokenizer_max_length": 8192,
            "code_commit": "0123456789abcdef",
        },
    )
    return directory


class ModelPublishTest(unittest.TestCase):
    def test_publish_uploads_a_validated_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _package(root / "package")
            store = LocalObjectStore(root / "store")

            result = publish_candidate(
                package,
                root_uri=ROOT_URI,
                model_version=MODEL_VERSION,
                object_store=store,
                gate_result={"passed": True},
            )

            self.assertEqual(
                result["package_uri"], candidate_uri(ROOT_URI, MODEL_VERSION)
            )
            self.assertIn("model.safetensors", result["uploaded_files"])
            self.assertIn("model_manifest.json", result["uploaded_files"])
            self.assertEqual(result["gate_status"], "candidate")

    def test_publish_refuses_a_failed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _package(root / "package")
            store = LocalObjectStore(root / "store")

            with self.assertRaisesRegex(ModelPublishError, "failed the deploy gate"):
                publish_candidate(
                    package,
                    root_uri=ROOT_URI,
                    model_version=MODEL_VERSION,
                    object_store=store,
                    gate_result={"passed": False},
                )
            self.assertEqual(store.list_uris(ROOT_URI), [])

    def test_publish_refuses_a_version_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _package(root / "package")
            store = LocalObjectStore(root / "store")

            with self.assertRaisesRegex(ModelPublishError, "does not match"):
                publish_candidate(
                    package,
                    root_uri=ROOT_URI,
                    model_version="router-modernbert-other-v9",
                    object_store=store,
                )

    def test_approve_marks_the_package_and_moves_the_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = LocalObjectStore(root / "store")
            publish_candidate(
                _package(root / "package"),
                root_uri=ROOT_URI,
                model_version=MODEL_VERSION,
                object_store=store,
                gate_result={"passed": True},
            )

            pointer = approve_candidate(
                root_uri=ROOT_URI,
                model_version=MODEL_VERSION,
                object_store=store,
                approved_by="ml-owner",
                pipeline_run_id="projects/1/locations/us-west1/pipelineJobs/run-1",
                work_dir=root / "work",
                logical_model_version="2.1",
                training_mode="minor_fine_tune",
            )

            self.assertEqual(pointer["model_version"], MODEL_VERSION)
            self.assertEqual(
                pointer["package_uri"], approved_uri(ROOT_URI, MODEL_VERSION)
            )
            self.assertIsNone(pointer["previous_model_version"])
            self.assertEqual(pointer["logical_model_version"], "2.1")
            self.assertEqual(pointer["training_mode"], "minor_fine_tune")

            approved_manifest = json.loads(
                store.read_text(
                    f"{approved_uri(ROOT_URI, MODEL_VERSION)}/model_manifest.json"
                )
            )
            self.assertEqual(approved_manifest["gate_status"], "approved")
            self.assertEqual(approved_manifest["approved_by"], "ml-owner")
            self.assertEqual(approved_manifest["logical_model_version"], "2.1")
            # Approval keeps one verified package; the duplicate candidate copy
            # is removed to keep continual-training storage bounded.
            self.assertIsNone(
                store.read_text(
                    f"{candidate_uri(ROOT_URI, MODEL_VERSION)}/model_manifest.json"
                )
            )

    def test_approving_twice_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = LocalObjectStore(root / "store")
            publish_candidate(
                _package(root / "package"),
                root_uri=ROOT_URI,
                model_version=MODEL_VERSION,
                object_store=store,
                gate_result={"passed": True},
            )
            approve_candidate(
                root_uri=ROOT_URI,
                model_version=MODEL_VERSION,
                object_store=store,
                approved_by="ml-owner",
                work_dir=root / "work",
            )

            with self.assertRaisesRegex(ModelPublishError, "already the current"):
                approve_candidate(
                    root_uri=ROOT_URI,
                    model_version=MODEL_VERSION,
                    object_store=store,
                    approved_by="ml-owner",
                    work_dir=root / "work2",
                )

    def test_second_approval_records_the_previous_model_and_rolls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = LocalObjectStore(root / "store")
            second_version = "router-modernbert-test-v2"
            for index, version in enumerate((MODEL_VERSION, second_version)):
                publish_candidate(
                    _package(root / f"package-{index}", model_version=version),
                    root_uri=ROOT_URI,
                    model_version=version,
                    object_store=store,
                    gate_result={"passed": True},
                )
                approve_candidate(
                    root_uri=ROOT_URI,
                    model_version=version,
                    object_store=store,
                    approved_by="ml-owner",
                    work_dir=root / f"work-{index}",
                )

            pointer = read_current_pointer(ROOT_URI, object_store=store)
            self.assertEqual(pointer["model_version"], second_version)
            self.assertEqual(pointer["previous_model_version"], MODEL_VERSION)

            rolled_back = rollback_current_pointer(
                root_uri=ROOT_URI, object_store=store, approved_by="ml-owner"
            )
            self.assertEqual(rolled_back["model_version"], MODEL_VERSION)
            self.assertEqual(rolled_back["rolled_back_from"], second_version)
            self.assertEqual(
                read_current_pointer(ROOT_URI, object_store=store)["model_version"],
                MODEL_VERSION,
            )

    def test_approve_rejects_a_candidate_older_than_the_approved_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = LocalObjectStore(root / "store")
            package = _package(root / "package")
            # Backdate the candidate so it predates the model already approved.
            manifest_path = package / "model_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["created_at"] = "2020-01-01T00:00:00+00:00"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            publish_candidate(
                package,
                root_uri=ROOT_URI,
                model_version=MODEL_VERSION,
                object_store=store,
                gate_result={"passed": True},
            )
            store.write_text(
                f"{ROOT_URI}/CURRENT.json",
                json.dumps(
                    {
                        "schema_version": "router-current-model-v1",
                        "model_version": "router-modernbert-test-v9",
                        "package_uri": approved_uri(
                            ROOT_URI, "router-modernbert-test-v9"
                        ),
                        "approved_at": "2026-08-14T00:00:00Z",
                    }
                ),
                if_generation_match=None,
            )

            with self.assertRaisesRegex(ModelPublishError, "allow_pointer_rollback"):
                approve_candidate(
                    root_uri=ROOT_URI,
                    model_version=MODEL_VERSION,
                    object_store=store,
                    approved_by="ml-owner",
                    work_dir=root / "work",
                )

            approved = approve_candidate(
                root_uri=ROOT_URI,
                model_version=MODEL_VERSION,
                object_store=store,
                approved_by="ml-owner",
                work_dir=root / "work2",
                allow_pointer_rollback=True,
            )
            self.assertEqual(approved["model_version"], MODEL_VERSION)

    def test_approve_rejects_a_missing_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = LocalObjectStore(root / "store")

            with self.assertRaisesRegex(ModelPublishError, "No candidate package"):
                approve_candidate(
                    root_uri=ROOT_URI,
                    model_version=MODEL_VERSION,
                    object_store=store,
                    approved_by="ml-owner",
                    work_dir=root / "work",
                )

    def test_approve_rejects_a_tampered_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = LocalObjectStore(root / "store")
            publish_candidate(
                _package(root / "package"),
                root_uri=ROOT_URI,
                model_version=MODEL_VERSION,
                object_store=store,
                gate_result={"passed": True},
            )
            tampered = (
                root
                / "store"
                / "llm-router"
                / "models/router-modernbert"
                / MODEL_VERSION
                / "candidate"
                / "model.safetensors"
            )
            tampered.write_bytes(b"tampered weights")

            with self.assertRaisesRegex(ModelPublishError, "re-validation"):
                approve_candidate(
                    root_uri=ROOT_URI,
                    model_version=MODEL_VERSION,
                    object_store=store,
                    approved_by="ml-owner",
                    work_dir=root / "work",
                )


if __name__ == "__main__":
    unittest.main()
