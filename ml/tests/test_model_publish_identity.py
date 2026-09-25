from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_router_labeling.deployment.model_package import assemble_model_package
from llm_router_labeling.deployment.model_publish import (
    PACKAGE_IDENTITY_FILENAME,
    ModelPublishError,
    approve_candidate,
    approved_uri,
    candidate_uri,
    publish_candidate,
)
from llm_router_labeling.shared.dataset_io import sha256_file
from llm_router_labeling.shared.object_store import LocalObjectStore, ObjectStoreError
from llm_router_labeling.training.routing_policy import compute_policy_sha256
from test_model_package_policy import policy_checkpoint, policy_metadata
from test_model_publish import MODEL_VERSION, ROOT_URI, _package, _write_json
from test_routing_policy import fixture_policy_payload


class RecordingStore(LocalObjectStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.uploads: list[str] = []
        self.identity_preconditions: list[int | None] = []

    def upload_file(self, source: Path, destination_uri: str) -> None:
        self.uploads.append(destination_uri)
        super().upload_file(source, destination_uri)

    def write_text(
        self, uri: str, text: str, *, if_generation_match: int | None = None
    ) -> None:
        if uri.endswith(f"/{PACKAGE_IDENTITY_FILENAME}"):
            self.identity_preconditions.append(if_generation_match)
        super().write_text(uri, text, if_generation_match=if_generation_match)


def _policy_package(root: Path, *, model_version: str = MODEL_VERSION) -> Path:
    root.mkdir(parents=True)
    checkpoint = root / "checkpoint"
    policy_checkpoint(checkpoint, "pairwise")
    policy = root / "policy.json"
    _write_json(policy, fixture_policy_payload("pairwise"))
    package = root / "package"
    assemble_model_package(
        checkpoint,
        package,
        {**policy_metadata(policy, "pairwise"), "model_version": model_version},
        routing_policy_path=policy,
    )
    return package


def _publish(package: Path, store: LocalObjectStore, version: str = MODEL_VERSION):
    return publish_candidate(
        package, root_uri=ROOT_URI, model_version=version, object_store=store
    )


class ModelPublishIdentityTest(unittest.TestCase):
    def test_upload_only_manifest_files_and_manifest_last_in_both_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _package(root / "package")
            (package / "credentials.secret").write_text("private fixture")
            (package / "scratch").mkdir()
            (package / "scratch" / "notes.txt").write_text("private fixture")
            store = RecordingStore(root / "store")

            result = _publish(package, store)
            manifest = json.loads((package / "model_manifest.json").read_text())
            expected = sorted(manifest["files"]) + ["model_manifest.json"]
            self.assertEqual(result["uploaded_files"], expected)
            self.assertEqual([uri.rsplit("/", 1)[1] for uri in store.uploads], expected)
            self.assertEqual(store.identity_preconditions, [None])
            claim_uri = f"{ROOT_URI}/{MODEL_VERSION}/{PACKAGE_IDENTITY_FILENAME}"
            identity = json.loads(store.read_text(claim_uri))
            self.assertEqual(identity["files"], manifest["files"])
            self.assertEqual(
                identity["candidate_manifest_sha256"],
                sha256_file(package / "model_manifest.json"),
            )

            staging = root / "approval" / MODEL_VERSION
            staging.mkdir(parents=True)
            (staging / "credentials.secret").write_text("private fixture")
            store.uploads.clear()
            approve_candidate(
                root_uri=ROOT_URI,
                model_version=MODEL_VERSION,
                object_store=store,
                approved_by="fixture",
                work_dir=root / "approval",
            )
            self.assertEqual([uri.rsplit("/", 1)[1] for uri in store.uploads], expected)
            self.assertEqual(json.loads(store.read_text(claim_uri)), identity)
            self.assertFalse(any("secret" in uri for uri in store.list_uris(ROOT_URI)))

    def test_interrupted_upload_resumes_only_identical_payload(self) -> None:
        class InterruptedStore(RecordingStore):
            fail_once = True

            def upload_file(self, source: Path, destination_uri: str) -> None:
                if self.fail_once and len(self.uploads) == 1:
                    self.fail_once = False
                    raise ObjectStoreError("simulated upload interruption")
                super().upload_file(source, destination_uri)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _policy_package(root / "v2")
            store = InterruptedStore(root / "store")
            with self.assertRaisesRegex(ObjectStoreError, "interruption"):
                _publish(package, store)
            claim_uri = f"{ROOT_URI}/{MODEL_VERSION}/{PACKAGE_IDENTITY_FILENAME}"
            identity = store.read_text(claim_uri)
            self.assertIsNotNone(identity)
            self.assertIsNone(
                store.read_text(
                    f"{candidate_uri(ROOT_URI, MODEL_VERSION)}/model_manifest.json"
                )
            )

            result = _publish(package, store)
            self.assertEqual(result["uploaded_files"][-1], "model_manifest.json")
            self.assertEqual(store.read_text(claim_uri), identity)
            self.assertEqual(store.identity_preconditions, [None])
            remote_manifest = json.loads(
                store.read_text(f"{result['package_uri']}/model_manifest.json")
            )
            self.assertEqual(
                remote_manifest["routing_policy_sha256"],
                json.loads(identity)["files"]["routing_policy.json"],
            )

    def test_changed_weights_policy_or_manifest_rejected_before_upload(self) -> None:
        for changed in ("weights", "policy", "manifest"):
            with (
                self.subTest(changed=changed),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                package = _policy_package(root / "v2")
                store = RecordingStore(root / "store")
                _publish(package, store)
                old_uploads = list(store.uploads)
                manifest_path = package / "model_manifest.json"
                manifest = json.loads(manifest_path.read_text())
                if changed == "weights":
                    (package / "model.safetensors").write_bytes(
                        b"different valid fixture"
                    )
                    manifest["files"]["model.safetensors"] = sha256_file(
                        package / "model.safetensors"
                    )
                elif changed == "policy":
                    policy_path = package / "routing_policy.json"
                    policy = json.loads(policy_path.read_text())
                    policy["lambda_cost"] = 0.75
                    policy["policy_sha256"] = compute_policy_sha256(policy)
                    _write_json(policy_path, policy)
                    digest = sha256_file(policy_path)
                    manifest["routing_policy_sha256"] = digest
                    manifest["files"]["routing_policy.json"] = digest
                else:
                    manifest["dataset_version"] = "different-dataset-version"
                _write_json(manifest_path, manifest)
                with self.assertRaisesRegex(ModelPublishError, "different package"):
                    _publish(package, store)
                self.assertEqual(store.uploads, old_uploads)

    def test_competing_claim_accepts_same_identity_and_rejects_different_identity(
        self,
    ) -> None:
        class RacingStore(RecordingStore):
            different = False

            def write_text(
                self, uri: str, text: str, *, if_generation_match: int | None = None
            ) -> None:
                if uri.endswith(f"/{PACKAGE_IDENTITY_FILENAME}"):
                    competitor = json.loads(text)
                    if self.different:
                        competitor["candidate_manifest_sha256"] = "f" * 64
                    super().write_text(
                        uri, json.dumps(competitor), if_generation_match=None
                    )
                    # The actual publisher's create-only write loses the race.
                super().write_text(uri, text, if_generation_match=if_generation_match)

        for different in (False, True):
            with (
                self.subTest(different=different),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                package = _policy_package(root / "v2")
                store = RacingStore(root / "store")
                store.different = different
                if different:
                    with self.assertRaisesRegex(ModelPublishError, "different package"):
                        _publish(package, store)
                    self.assertEqual(store.uploads, [])
                else:
                    _publish(package, store)
                    self.assertTrue(store.uploads[-1].endswith("/model_manifest.json"))
                self.assertEqual(store.identity_preconditions, [None, None])

    def test_unclaimed_legacy_paths_are_not_overwritten(self) -> None:
        for state in ("candidate", "approved"):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                package = _policy_package(root / "v2")
                store = RecordingStore(root / "store")
                legacy_uri = f"{ROOT_URI}/{MODEL_VERSION}/{state}/model.safetensors"
                store.upload_file(package / "model.safetensors", legacy_uri)
                store.uploads.clear()
                with self.assertRaisesRegex(ModelPublishError, "unclaimed legacy"):
                    _publish(package, store)
                self.assertEqual(store.uploads, [])
                self.assertEqual(store.identity_preconditions, [])
                self.assertEqual(store.list_uris(ROOT_URI), [legacy_uri])

    def test_identity_protects_v1_and_blocks_schema_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = _package(root / "v1")
            store = RecordingStore(root / "store")
            _publish(package, store)
            _publish(package, store)
            old_uploads = list(store.uploads)
            with self.assertRaisesRegex(ModelPublishError, "different package"):
                _publish(_policy_package(root / "v2"), store)
            self.assertEqual(store.uploads, old_uploads)

    def test_v2_legacy_gate_cannot_upload_approved_or_change_current(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = RecordingStore(root / "store")
            prior_version = "existing-v1"
            _publish(
                _package(root / "v1", model_version=prior_version), store, prior_version
            )
            approve_candidate(
                root_uri=ROOT_URI,
                model_version=prior_version,
                object_store=store,
                approved_by="fixture",
                work_dir=root / "prior-approval",
            )
            _publish(_policy_package(root / "v2"), store)
            original_pointer = store.read_text(f"{ROOT_URI}/CURRENT.json")
            original_inventory = store.list_uris(ROOT_URI)
            old_uploads = list(store.uploads)
            with self.assertRaisesRegex(
                ModelPublishError, "objective-aware deployment gate"
            ):
                approve_candidate(
                    root_uri=ROOT_URI,
                    model_version=MODEL_VERSION,
                    object_store=store,
                    approved_by="fixture",
                    work_dir=root / "blocked-approval",
                )
            self.assertEqual(
                store.read_text(f"{ROOT_URI}/CURRENT.json"), original_pointer
            )
            self.assertEqual(store.list_uris(ROOT_URI), original_inventory)
            self.assertEqual(store.uploads, old_uploads)
            self.assertEqual(store.list_uris(approved_uri(ROOT_URI, MODEL_VERSION)), [])


if __name__ == "__main__":
    unittest.main()
