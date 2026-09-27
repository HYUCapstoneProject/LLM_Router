"""Shared producer contract plus connected approval-race regressions."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import test_r0_promotion_evidence as qualification_tests
from llm_router_labeling.deployment import (
    deploy_gate_cli,
    model_publish,
    model_publish_cli,
)
from llm_router_labeling.deployment.model_package import assemble_model_package
from llm_router_labeling.deployment.promotion_evidence import (
    SUPPORTED_PROMOTION_CONTRACTS,
    PromotionEvidenceError,
    build_promotion_evidence,
    promotion_contract,
)
from llm_router_labeling.shared.dataset_io import write_json
from llm_router_labeling.shared.object_store import ObjectStoreError
from test_r0_promotion_evidence import OBS, ROOT, VERSION, invoke, prepare


@pytest.fixture
def chain(tmp_path, monkeypatch):
    return qualification_tests.chain.__wrapped__(tmp_path, monkeypatch)


def test_recorded_fixture_is_exact_output_of_shared_producer():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/shared-promotion-producer.json").read_text()
    )
    expected = fixture["expected"]

    class Store:
        def read_text(self, uri):
            return fixture["objects"].get(uri)

        def generation(self, uri):
            return fixture["generations"].get(uri)

    root = expected["model_publish_root_uri"]
    run = expected["observability_run_uri"]
    evidence = build_promotion_evidence(
        root_uri=root,
        model_version=expected["candidate_model_version"],
        batch_id=expected["batch_id"],
        pipeline_run_id=expected["pipeline_job_resource_name"],
        result=json.loads(fixture["objects"][root + "/CURRENT.json"]),
        store=Store(),
        gate_text=fixture["objects"][run + "/gate/gate-result.json"],
    )
    stored = json.loads(fixture["objects"][run + "/result/publish-result.json"])
    assert stored.pop("source_schema_version") == evidence.pop("schema_version")
    stored.pop("schema_version")
    stored.pop("updated_at")
    assert evidence == stored
    assert SUPPORTED_PROMOTION_CONTRACTS == {
        ("ordinal", "model-package-v1", "deploy-gate-result-v1"),
        ("ordinal", "model-package-v1", "direct-promotion-result-v1"),
    }


@pytest.mark.parametrize(
    "objective,package,gate",
    [
        ("pointwise", "model-package-v2", "deploy-gate-result-v1"),
        ("pointwise", "model-package-v2", "future-pointwise-gate-v1"),
        ("pointwise", "model-package-v1", "deploy-gate-result-v1"),
        ("ordinal", "model-package-v1", "unknown-gate"),
    ],
)
def test_unimplemented_mapping_cannot_be_enabled_by_envelope_fields(
    objective, package, gate
):
    with pytest.raises(PromotionEvidenceError):
        promotion_contract(
            {"schema_version": package, "objective": objective},
            {"schema_version": gate},
        )


def test_approval_resume_reuses_original_stamp_after_partial_upload(chain, monkeypatch):
    c = chain
    prepare(c, monkeypatch)
    monkeypatch.setattr(model_publish_cli, "_object_store", lambda args: c.store)
    upload = c.store.upload_file
    original_manifest = []

    def fail_on_manifest(source, uri):
        if uri.endswith("/approved/model_manifest.json"):
            original_manifest.append(source.read_bytes())
            raise ObjectStoreError("simulated partial approval")
        upload(source, uri)

    monkeypatch.setattr(c.store, "upload_file", fail_on_manifest)
    with pytest.raises(ObjectStoreError, match="partial approval"):
        invoke(monkeypatch, model_publish_cli, c.approve_args)
    assert c.store.read_text(ROOT + "/CURRENT.json") is None

    class Later:
        @staticmethod
        def now(tz):
            return datetime(2030, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(model_publish, "datetime", Later)
    monkeypatch.setattr(c.store, "upload_file", upload)
    invoke(monkeypatch, model_publish_cli, c.approve_args)
    actual = c.store.read_text(
        f"{ROOT}/{VERSION}/approved/model_manifest.json"
    ).encode()
    assert actual == original_manifest[0]


def test_same_version_competing_approval_cannot_rewrite_approved_manifest(
    chain, monkeypatch
):
    c = chain
    prepare(c, monkeypatch)
    monkeypatch.setattr(model_publish_cli, "_object_store", lambda args: c.store)
    upload = c.store.upload_file
    attempted = []

    def concurrent_upload(source, uri):
        if "/approved/" in uri and not attempted:
            attempted.append(True)
            arguments = list(c.approve_args)
            arguments[arguments.index("--approved-by") + 1] = "competing-owner"
            with pytest.raises(
                model_publish.ModelPublishError, match="different approval identity"
            ):
                invoke(monkeypatch, model_publish_cli, arguments)
        upload(source, uri)

    monkeypatch.setattr(c.store, "upload_file", concurrent_upload)
    invoke(monkeypatch, model_publish_cli, c.approve_args)
    assert attempted
    manifest = json.loads(
        c.store.read_text(f"{ROOT}/{VERSION}/approved/model_manifest.json")
    )
    assert manifest["approved_by"] == "test"
    assert (
        json.loads(c.store.read_text(OBS + "/result/publish-result.json"))["outcome"]
        == "promoted"
    )


def prepare_length_chain(c):
    config = json.loads((c.checkpoint / "config.json").read_text())
    write_json(
        c.checkpoint / "config.json", {**config, "max_position_embeddings": 8192}
    )
    metrics = json.loads(c.evaluation.read_text())
    metrics["model_info"] = {"max_length": 8192}
    write_json(c.evaluation, metrics)
    write_json(c.baseline, metrics)
    metadata = json.loads((c.package / "model_manifest.json").read_text())
    metadata.update(
        length_contract_version="train-inference-lengths-v1",
        training_max_length=512,
        inference_max_length=8192,
        tokenizer_max_length=8192,
    )
    package = c.package.parent / "long-package"
    manifest = assemble_model_package(
        c.checkpoint, package, metadata, evaluation_metrics_path=c.evaluation
    )
    c.gate_args[c.gate_args.index("--candidate-package") + 1] = package
    c.publish_args[c.publish_args.index("--package-dir") + 1] = package
    return manifest


def test_length_binding_is_preserved_through_gate_publication_and_evidence(
    chain, monkeypatch
):
    c = chain
    manifest = prepare_length_chain(c)
    baseline = {**manifest, "model_version": "base-r0", "gate_status": "approved"}
    baseline_text = json.dumps(baseline)
    c.store.write_text(ROOT + "/base-r0/approved/model_manifest.json", baseline_text)
    c.store.write_text(
        ROOT + "/CURRENT.json",
        json.dumps(
            {
                "schema_version": "router-current-model-v1",
                "model_version": "base-r0",
                "package_uri": ROOT + "/base-r0/approved",
                "manifest_sha256": hashlib.sha256(baseline_text.encode()).hexdigest(),
            }
        ),
    )
    monkeypatch.setattr(deploy_gate_cli, "build_object_store", lambda root: c.store)
    c.gate_args.extend(["--model-publish-root-uri", ROOT])
    prepare(c, monkeypatch)
    invoke(
        monkeypatch,
        model_publish_cli,
        [*c.approve_args, "--training-mode", "minor_fine_tune"],
    )
    evidence = json.loads(c.store.read_text(OBS + "/result/publish-result.json"))
    assert evidence["inference_max_length"] == 8192
    assert evidence["length_contract_version"] == "train-inference-lengths-v1"
    assert evidence["previous_pointer_generation"] is not None


@pytest.mark.parametrize(
    "change", ["baseline_package", "baseline_metrics", "gate_contract"]
)
def test_wrong_inference_contract_never_publishes_current(chain, monkeypatch, change):
    c = chain
    manifest = prepare_length_chain(c)
    if change == "gate_contract":
        invoke(monkeypatch, deploy_gate_cli, c.gate_args)
        gate = json.loads(c.gate.read_text())
        gate["inference_max_length"] = 512
        write_json(c.gate, gate)
        with pytest.raises(SystemExit, match="length contract"):
            invoke(monkeypatch, model_publish_cli, c.publish_args)
    else:
        baseline = {**manifest, "model_version": "base-r0", "gate_status": "approved"}
        if change == "baseline_package":
            baseline.update(inference_max_length=512, tokenizer_max_length=512)
        else:
            metrics = json.loads(c.baseline.read_text())
            metrics["model_info"]["max_length"] = 512
            write_json(c.baseline, metrics)
            baseline["files"] = {
                **baseline["files"],
                "gate_metrics.json": hashlib.sha256(
                    c.baseline.read_bytes()
                ).hexdigest(),
            }
        c.store.write_text(
            ROOT + "/base-r0/approved/model_manifest.json", json.dumps(baseline)
        )
        monkeypatch.setattr(deploy_gate_cli, "build_object_store", lambda root: c.store)
        with pytest.raises(SystemExit, match="length"):
            invoke(
                monkeypatch,
                deploy_gate_cli,
                [*c.gate_args, "--model-publish-root-uri", ROOT],
            )
    assert c.store.read_text(ROOT + "/CURRENT.json") is None
