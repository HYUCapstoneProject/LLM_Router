"""Generate synthetic producer -> verified serving -> Spring contract fixtures.

Run with the serving test environment and --serving-root pointing to an isolated
checkout. No downloads, network, real weights, gate decisions or publication are
performed. Candidate packages are deliberately loaded with require_approved=False;
the default approved-only loader is separately asserted to reject them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def run(serving_root: Path, output: Path) -> dict:
    ml_root = Path(__file__).resolve().parents[1]
    for directory in (
        ml_root / "src",
        ml_root / "tests",
        serving_root / "src",
        serving_root / "tests",
    ):
        sys.path.insert(0, str(directory))

    from conftest import FakeEmbedder, FakeEscalationService, policy_payload_factory
    from fastapi.testclient import TestClient
    from llm_router_labeling.deployment.model_package import (
        EXPECTED_ID2LABEL,
        assemble_model_package,
    )
    from llm_router_labeling.deployment.model_publish import rollback_current_pointer
    from llm_router_labeling.shared.dataset_io import sha256_file
    from llm_router_labeling.shared.object_store import LocalObjectStore
    from llm_router_labeling.training.routing_policy import (
        RoutingPolicy as ProducerPolicy,
    )
    from llm_router_labeling.training.routing_policy import compute_policy_sha256
    from llm_router_labeling.training.routing_policy import (
        route_scores as producer_route,
    )
    from llm_router_serving import main
    from llm_router_serving.config import Settings
    from llm_router_serving.model_manager import ModelManager
    from llm_router_serving.model_source import (
        ModelSourceError,
        read_current_model_pointer,
        verify_package,
    )
    from test_inference import FakeRouterModel, FakeTokenizer
    from test_model_promotion import _checkpoint, _metadata, _write_json

    fixture_policy_payload = policy_payload_factory.__wrapped__()

    def policy_checkpoint(directory, objective):
        directory.mkdir()
        _checkpoint(directory)
        labels = (
            ("easy", "medium", "hard")
            if objective == "ordinal"
            else ("low", "medium", "high")
        )
        config = {
            "id2label": dict(enumerate(labels)),
            "label2id": {label: i for i, label in enumerate(labels)},
        }
        if objective != "ordinal":
            config["router_objective"] = objective
        _write_json(directory / "config.json", config)
        _write_json(
            directory / "training_config.json",
            {"max_length": 512, "objective_method": objective},
        )

    def policy_metadata(path, objective):
        return {
            **_metadata(),
            "schema_version": "model-package-v2",
            "objective": objective,
            "tokenizer_max_length": 512,
            "input_contract": "structure-preserving-v2",
            "routing_policy_file": "routing_policy.json",
            "routing_policy_sha256": sha256_file(path),
        }

    class MemoryStore(LocalObjectStore):
        def read_bytes(self, uri):
            return self._path(uri).read_bytes()

    class FakeLoadedModel(FakeRouterModel):
        def __init__(self, directory):
            config = json.loads((Path(directory) / "config.json").read_text())
            config["id2label"] = {
                int(key): value for key, value in config["id2label"].items()
            }
            config["num_labels"] = len(config["id2label"])
            self.config = SimpleNamespace(**config)

        def to(self, device):
            return self

        def eval(self):
            return self

    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda directory: FakeTokenizer()
        ),
        AutoModelForSequenceClassification=SimpleNamespace(
            from_pretrained=FakeLoadedModel
        ),
    )
    report = {
        "schemaVersion": "synthetic-routing-cross-repo-v1",
        "syntheticOnly": True,
        "approvalMode": "candidate-contract-test-not-promotion",
        "cases": [],
    }
    with tempfile.TemporaryDirectory(prefix="router-cross-repo-") as temporary:
        root = Path(temporary)
        store = MemoryStore(root / "store")
        root_uri = "gs://synthetic-contract-only/models"
        pointer_uri = root_uri + "/CURRENT.json"
        packages = {}

        def upload_and_pointer(package, manifest):
            version = manifest["model_version"]
            prefix = f"{root_uri}/{version}/approved"
            for path in package.iterdir():
                store.upload_file(path, prefix + "/" + path.name)
            pointer = {
                "schema_version": "router-current-model-v1",
                "model_version": version,
                "package_uri": prefix,
                "manifest_sha256": sha256_file(package / "model_manifest.json"),
                "id2label": EXPECTED_ID2LABEL,
            }
            store.write_text(
                pointer_uri,
                json.dumps(pointer),
                if_generation_match=store.generation(pointer_uri),
            )
            parsed = read_current_model_pointer(pointer_uri, object_store=store)
            assert parsed.model_version == version
            packages[version] = (package, pointer)
            return parsed

        def loaded_manager(pointer):
            settings = Settings(
                internal_api_token="synthetic-secret",
                load_models_on_startup=False,
                router_model_require_approved=False,
                router_model_cache_dir=root / "cache",
                model_device="cpu",
                router_max_length=8192,
                retraining_db_path=root / "retraining.sqlite3",
            )
            with patch.object(main, "GcsPackageStore", return_value=store):
                loader_manager = main._build_model_manager(settings)
            engine = loader_manager._router_version_loader(
                pointer.package_uri, pointer.model_version, pointer.manifest_sha256
            )
            manager = ModelManager.loaded(
                embed_engine=FakeEmbedder(), router_engine=engine
            )
            manager._router_version_loader = loader_manager._router_version_loader
            return settings, manager

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            for objective in ("ordinal", "pointwise", "pairwise"):
                checkpoint = root / (objective + "-checkpoint")
                policy_checkpoint(checkpoint, objective)
                original_config = sha256_file(checkpoint / "config.json")
                policy = fixture_policy_payload(objective)
                assert policy["policy_sha256"] == compute_policy_sha256(policy)
                policy_path = root / (objective + "-policy.json")
                _write_json(policy_path, policy)
                package = root / (objective + "-package")
                version = "router-synthetic-" + objective
                manifest = assemble_model_package(
                    checkpoint,
                    package,
                    policy_metadata(policy_path, objective),
                    routing_policy_path=policy_path,
                    model_version=version,
                )
                assert manifest["gate_status"] == "candidate"
                try:
                    verify_package(package, expected_model_version=version)
                except ModelSourceError as error:
                    assert "gate_status" in str(error)
                else:
                    raise AssertionError(
                        "The default loader accepted an unapproved candidate"
                    )
                verify_package(
                    package, expected_model_version=version, require_approved=False
                )
                assert manifest["checkpoint_config_sha256"] == original_config
                assert sha256_file(checkpoint / "config.json") == original_config
                copied_config = json.loads((package / "config.json").read_text())
                assert copied_config["router_objective"] == objective
                assert (manifest["files"]["config.json"] != original_config) == (
                    objective == "ordinal"
                )
                assert manifest["routing_policy_sha256"] == sha256_file(policy_path)
                assert manifest["routing_policy_sha256"] != policy["policy_sha256"]
                pointer = upload_and_pointer(package, manifest)
                settings, manager = loaded_manager(pointer)
                engine = manager.classification_engine()
                assert engine.package_manifest_sha256 == pointer.manifest_sha256
                assert engine._max_length == 512
                assert tuple(engine._model.config.id2label.values()) == (
                    ("easy", "medium", "hard")
                    if objective == "ordinal"
                    else ("low", "medium", "high")
                )
                app = main.create_app(
                    settings=settings,
                    model_manager=manager,
                    escalation_service=FakeEscalationService(),
                )
                headers = {"Authorization": "Bearer synthetic-secret"}
                with TestClient(app) as client:
                    response = client.post(
                        "/router/classify/v2",
                        json={"message": "    x+y\r\n"},
                        headers=headers,
                    )
                    assert response.status_code == 200, response.text
                    payload = response.json()
                    assert engine._tokenizer.last_message == "    x+y"
                    expected = producer_route(
                        ProducerPolicy.from_mapping(policy),
                        [0.10000000149011612, 2.0, 0.30000001192092896],
                    )
                    assert payload["action"] == expected.action
                    assert payload["policySha256"] == policy["policy_sha256"]
                    assert payload["candidateModelIds"] == {
                        item["action"]: item["model_id"]
                        for item in policy["candidate_pool"]["candidates"]
                    }
                    assert payload["confidence"] == (
                        None
                        if objective == "pairwise"
                        else payload["actionScores"][payload["action"]]
                    )
                    blocked = client.post(
                        "/router/classify", json={"message": "q"}, headers=headers
                    )
                    assert blocked.status_code == 409
                report["cases"].append(
                    {
                        "objective": objective,
                        "packageManifestSha256": pointer.manifest_sha256,
                        "policyFileSha256": manifest["routing_policy_sha256"],
                        "canonicalPolicySha256": policy["policy_sha256"],
                        "checkpointConfigSha256": original_config,
                        "packagedConfigSha256": manifest["files"]["config.json"],
                        "actualOutputLabels": list(copied_config["id2label"].values()),
                        "currentTierAliases": list(EXPECTED_ID2LABEL.values()),
                        "response": payload,
                    }
                )

            legacy_checkpoint = root / "legacy-checkpoint"
            legacy_checkpoint.mkdir()
            _checkpoint(legacy_checkpoint)
            legacy = root / "legacy-package"
            legacy_manifest = assemble_model_package(
                legacy_checkpoint,
                legacy,
                _metadata(),
                model_version="router-synthetic-legacy",
            )
            # Simulate an already-approved historical v1 solely in this temporary
            # object store; this is not a gate decision or a publishing call.
            legacy_manifest["gate_status"] = "approved"
            _write_json(legacy / "model_manifest.json", legacy_manifest)
            legacy_pointer = upload_and_pointer(legacy, legacy_manifest)
            previous_v2 = packages["router-synthetic-pairwise"][1]
            current = {
                **previous_v2,
                "previous_model_version": legacy_pointer.model_version,
                "previous_package_uri": legacy_pointer.package_uri,
            }
            store.write_text(
                pointer_uri,
                json.dumps(current),
                if_generation_match=store.generation(pointer_uri),
            )
            rollback = rollback_current_pointer(
                root_uri=root_uri,
                object_store=store,
                approved_by="synthetic-contract-test",
            )
            report["rollbackPointerManifestCorrect"] = (
                rollback["manifest_sha256"] == legacy_pointer.manifest_sha256
            )
            # Separately check the runtime swap using a correctly bound old manifest.
            assert asyncio.run(
                manager.reload_router(
                    package_uri=legacy_pointer.package_uri,
                    model_version=legacy_pointer.model_version,
                    manifest_sha256=legacy_pointer.manifest_sha256,
                )
            )
            assert (
                manager.snapshot().previous_router_model_version
                == "router-synthetic-pairwise"
            )
            app = main.create_app(
                settings=settings,
                model_manager=manager,
                escalation_service=FakeEscalationService(),
            )
            with TestClient(app) as client:
                response = client.post(
                    "/router/classify", json={"message": "legacy q"}, headers=headers
                )
                assert response.status_code == 200, response.text
                report["legacyResponse"] = response.json()
                blocked = client.post(
                    "/router/classify/v2", json={"message": "q"}, headers=headers
                )
                assert blocked.status_code == 503
                assert blocked.json()["code"] == "ROUTER_POLICY_NOT_CONFIGURED"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.serving_root.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "fixture": str(args.output),
                "objectives": [item["objective"] for item in report["cases"]],
                "runtimeRollbackVerified": True,
                "publisherRollbackManifestCorrect": report[
                    "rollbackPointerManifestCorrect"
                ],
            }
        )
    )
    if not report["rollbackPointerManifestCorrect"]:
        raise SystemExit(
            "Publisher rollback retained the replaced model's manifest SHA instead of the target manifest SHA"
        )


if __name__ == "__main__":
    main()
