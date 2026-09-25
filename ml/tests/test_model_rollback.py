from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from llm_router_labeling.deployment.model_publish import (
    PACKAGE_IDENTITY_FILENAME,
    ModelPublishError,
    approve_candidate,
    approved_uri,
    current_pointer_uri,
    publish_candidate,
    rollback_current_pointer,
)
from llm_router_labeling.shared.object_store import LocalObjectStore, ObjectStoreError
from test_model_publish import ROOT_URI, _package

PREVIOUS_VERSION = "previous-v1"
CURRENT_VERSION = "current-v2"


def _two_approved_versions(
    root: Path,
    store: LocalObjectStore,
    *,
    previous_metadata: dict[str, str] | None = None,
) -> None:
    for version, metadata in (
        (PREVIOUS_VERSION, previous_metadata or {}),
        (
            CURRENT_VERSION,
            {
                "logical_model_version": "2.1",
                "training_mode": "minor_fine_tune",
                "pipeline_run_id": "current-pipeline-run",
            },
        ),
    ):
        publish_candidate(
            _package(root / version, model_version=version),
            root_uri=ROOT_URI,
            model_version=version,
            object_store=store,
        )
        approve_candidate(
            root_uri=ROOT_URI,
            model_version=version,
            object_store=store,
            approved_by="fixture",
            work_dir=root / "approval",
            **metadata,
        )


def _replace_remote_json(store: LocalObjectStore, uri: str, payload: dict) -> None:
    store.write_text(
        uri, json.dumps(payload), if_generation_match=store.generation(uri)
    )


class ModelRollbackTest(unittest.TestCase):
    def test_restores_target_sha_and_metadata_and_supports_legacy_without_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = LocalObjectStore(root / "store")
            target_metadata = {
                "logical_model_version": "1.0",
                "training_mode": "initial_training",
                "pipeline_run_id": "initial-pipeline-run",
            }
            _two_approved_versions(root, store, previous_metadata=target_metadata)
            store.delete_uri(
                f"{ROOT_URI}/{PREVIOUS_VERSION}/{PACKAGE_IDENTITY_FILENAME}"
            )
            manifest_uri = (
                f"{approved_uri(ROOT_URI, PREVIOUS_VERSION)}/model_manifest.json"
            )
            target_manifest = store.read_text(manifest_uri)
            pointer_uri = current_pointer_uri(ROOT_URI)
            pointer = json.loads(store.read_text(pointer_uri))
            pointer["objective"] = "pairwise"
            pointer["routing_policy_sha256"] = "f" * 64
            pointer["unknown_current_version_metadata"] = "must not survive"
            pointer["retained_model_versions"].append("retention-anchor-v0")
            _replace_remote_json(store, pointer_uri, pointer)

            result = rollback_current_pointer(
                root_uri=ROOT_URI, object_store=store, approved_by="rollback-owner"
            )
            self.assertEqual(
                result["manifest_sha256"],
                hashlib.sha256(target_manifest.encode("utf-8")).hexdigest(),
            )
            self.assertNotEqual(result["manifest_sha256"], pointer["manifest_sha256"])
            self.assertEqual(result["model_version"], PREVIOUS_VERSION)
            self.assertEqual(
                result["package_uri"], approved_uri(ROOT_URI, PREVIOUS_VERSION)
            )
            self.assertEqual(result["previous_model_version"], CURRENT_VERSION)
            self.assertEqual(result["approved_by"], "rollback-owner")
            for field, value in target_metadata.items():
                self.assertEqual(result[field], value)
            for field in (
                "objective",
                "routing_policy_sha256",
                "unknown_current_version_metadata",
            ):
                self.assertNotIn(field, result)
            self.assertIn("retention-anchor-v0", result["retained_model_versions"])
            self.assertEqual(
                result["id2label"], {"0": "easy", "1": "medium", "2": "hard"}
            )
            self.assertEqual(json.loads(store.read_text(pointer_uri)), result)
            self.assertEqual(store.read_text(manifest_uri), target_manifest)

    def test_missing_optional_metadata_does_not_inherit_current_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = LocalObjectStore(root / "store")
            _two_approved_versions(root, store)
            result = rollback_current_pointer(
                root_uri=ROOT_URI, object_store=store, approved_by="fixture"
            )
            self.assertNotIn("logical_model_version", result)
            self.assertNotIn("training_mode", result)
            self.assertIsNone(result["pipeline_run_id"])

    def test_invalid_previous_package_preserves_current_pointer(self) -> None:
        for corruption in (
            "missing-manifest",
            "weights",
            "required-metadata",
            "candidate",
            "version",
            "optional-metadata",
        ):
            with (
                self.subTest(corruption=corruption),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                store = LocalObjectStore(root / "store")
                _two_approved_versions(root, store)
                pointer_uri = current_pointer_uri(ROOT_URI)
                original_pointer = store.read_text(pointer_uri)
                original_generation = store.generation(pointer_uri)
                base = approved_uri(ROOT_URI, PREVIOUS_VERSION)
                manifest_uri = f"{base}/model_manifest.json"
                if corruption == "missing-manifest":
                    store.delete_uri(manifest_uri)
                elif corruption == "weights":
                    store.write_text(
                        f"{base}/model.safetensors",
                        "corrupted weights",
                        if_generation_match=store.generation(
                            f"{base}/model.safetensors"
                        ),
                    )
                else:
                    manifest = json.loads(store.read_text(manifest_uri))
                    if corruption == "required-metadata":
                        del manifest["dataset_version"]
                    elif corruption == "candidate":
                        manifest["gate_status"] = "candidate"
                    elif corruption == "version":
                        manifest["model_version"] = "unrelated-version"
                    else:
                        manifest["training_mode"] = {"invalid": "metadata"}
                    _replace_remote_json(store, manifest_uri, manifest)
                with self.assertRaises(ModelPublishError):
                    rollback_current_pointer(
                        root_uri=ROOT_URI, object_store=store, approved_by="fixture"
                    )
                self.assertEqual(store.read_text(pointer_uri), original_pointer)
                self.assertEqual(store.generation(pointer_uri), original_generation)

    def test_previous_uri_must_be_expected_approved_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = LocalObjectStore(root / "store")
            _two_approved_versions(root, store)
            pointer_uri = current_pointer_uri(ROOT_URI)
            pointer = json.loads(store.read_text(pointer_uri))
            pointer["previous_package_uri"] = f"{ROOT_URI}/{PREVIOUS_VERSION}/candidate"
            _replace_remote_json(store, pointer_uri, pointer)
            with self.assertRaisesRegex(
                ModelPublishError, "previous version's approved"
            ):
                rollback_current_pointer(
                    root_uri=ROOT_URI, object_store=store, approved_by="fixture"
                )
            self.assertEqual(json.loads(store.read_text(pointer_uri)), pointer)

    def test_concurrent_current_change_during_download_is_not_overwritten(self) -> None:
        class RacingStore(LocalObjectStore):
            race_enabled = False

            def download_file(self, source_uri: str, destination: Path) -> None:
                super().download_file(source_uri, destination)
                if self.race_enabled:
                    self.race_enabled = False
                    pointer_uri = current_pointer_uri(ROOT_URI)
                    pointer = json.loads(self.read_text(pointer_uri))
                    pointer["model_version"] = "concurrent-promotion-v3"
                    _replace_remote_json(self, pointer_uri, pointer)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = RacingStore(root / "store")
            _two_approved_versions(root, store)
            store.race_enabled = True
            with self.assertRaisesRegex(
                ObjectStoreError, "Generation precondition failed"
            ):
                rollback_current_pointer(
                    root_uri=ROOT_URI, object_store=store, approved_by="fixture"
                )
            pointer = json.loads(store.read_text(current_pointer_uri(ROOT_URI)))
            self.assertEqual(pointer["model_version"], "concurrent-promotion-v3")
            self.assertNotIn("rolled_back_from", pointer)


if __name__ == "__main__":
    unittest.main()
