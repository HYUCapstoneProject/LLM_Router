"""Real producer, CAS and post-CURRENT retry coverage without model execution."""

import json
from pathlib import Path

import pytest
from llm_router_labeling.deployment import direct_promotion as direct
from llm_router_labeling.deployment.model_package import assemble_model_package
from llm_router_labeling.deployment.model_publish import (
    approve_candidate,
    publish_candidate,
    read_current_pointer,
)
from llm_router_labeling.shared.object_store import LocalObjectStore
from test_model_length_contract import producer_inputs

ROOT = "gs://test/models/router"
OBS = "gs://test/runs/direct-batch/observability-v1"
JOB = "projects/test/locations/test/pipelineJobs/direct-test"


def setup(root):
    checkpoint, metadata, _ = producer_inputs(root, schema="v1")
    baseline = "baseline-r0"
    package = root / "baseline-package"
    assemble_model_package(checkpoint, package, metadata, model_version=baseline)
    store = LocalObjectStore(root / "objects")
    publish_candidate(
        package, root_uri=ROOT, model_version=baseline, object_store=store
    )
    approve_candidate(
        root_uri=ROOT,
        model_version=baseline,
        object_store=store,
        approved_by="synthetic-test",
        pipeline_run_id="baseline-job",
        work_dir=root / "approve",
    )
    metadata.pop("created_at", None)
    (checkpoint / "model.safetensors").write_bytes(b"synthetic updated weights")
    args = {
        "checkpoint_dir": checkpoint,
        "output_dir": root / "output",
        "metadata": metadata,
        "model_version": "candidate-r0",
        "baseline_model_version": baseline,
        "root_uri": ROOT,
        "batch_id": "direct-batch",
        "pipeline_run_id": JOB,
        "observability_run_uri": OBS,
        "logical_model_version": "1.1",
        "training_mode": "minor_fine_tune",
        "retention_anchor_model_version": baseline,
        "object_store": store,
        "promotion_mode": "direct",
    }
    return args, store


def test_direct_produces_honest_evidence_and_retry(tmp_path):
    args, store = setup(tmp_path)
    result = direct.promote_direct(**args)
    proof = json.loads(store.read_text(OBS + "/promotion/direct-result.json"))
    assert "passed" not in proof
    assert result["candidate_metrics_sha256"] is None
    assert result["quality_gate_skipped"] is True
    pointer = read_current_pointer(ROOT, object_store=store)
    manifest = json.loads(
        store.read_text(pointer["package_uri"] + "/model_manifest.json")
    )
    assert manifest["inference_max_length"] == 8192
    assert manifest["training_max_length"] == 512
    assert manifest["evaluation_role"] == "not_evaluated"
    assert not any("gate_metrics" in key for key in manifest["files"])
    assert direct.promote_direct(**args) == result


def test_retry_after_current_before_publish_evidence(tmp_path, monkeypatch):
    args, store = setup(tmp_path)
    original = direct.write_observability_json

    def interrupt(uri, *a, **kw):
        if uri.endswith("/result/publish-result.json"):
            raise RuntimeError("simulated post-CURRENT interruption")
        return original(uri, *a, **kw)

    monkeypatch.setattr(direct, "write_observability_json", interrupt)
    with pytest.raises(RuntimeError):
        direct.promote_direct(**args)
    manifest = (tmp_path / "output/package/model_manifest.json").read_bytes()
    generation = store.generation(ROOT + "/CURRENT.json")
    monkeypatch.setattr(direct, "write_observability_json", original)
    assert direct.promote_direct(**args)["outcome"] == "promoted"
    assert store.generation(ROOT + "/CURRENT.json") == generation
    assert (tmp_path / "output/package/model_manifest.json").read_bytes() == manifest


@pytest.mark.parametrize(
    "field,value",
    [("promotion_mode", "quality_gate"), ("baseline_model_version", "wrong")],
)
def test_rejects_wrong_identity_without_current_mutation(tmp_path, field, value):
    args, store = setup(tmp_path)
    before = store.read_text(ROOT + "/CURRENT.json")
    args[field] = value
    with pytest.raises(ValueError):
        direct.promote_direct(**args)
    assert store.read_text(ROOT + "/CURRENT.json") == before


def make_fixture(root):
    args, store = setup(root)
    evidence = direct.promote_direct(**args)
    uris = store.list_uris("gs://test/")
    objects = {uri: store.read_text(uri) for uri in uris if uri.endswith(".json")}
    return {
        "schema_version": "direct-promotion-producer-fixture-v1",
        "synthetic_only": True,
        "objects": objects,
        "generations": {uri: store.generation(uri) for uri in objects},
        "expected": {
            "model_publish_root_uri": ROOT,
            "observability_run_uri": OBS,
            "candidate_model_version": args["model_version"],
            "batch_id": args["batch_id"],
            "pipeline_job_resource_name": JOB,
            "evidence": evidence,
        },
    }


def test_second_direct_run_uses_unevaluated_current_baseline(tmp_path):
    args, _store = setup(tmp_path)
    first = direct.promote_direct(**args)
    args.update(
        model_version="candidate-r0-second",
        baseline_model_version="candidate-r0",
        output_dir=tmp_path / "second-output",
        batch_id="direct-second",
        pipeline_run_id=JOB + "-second",
        observability_run_uri=OBS + "-second",
        logical_model_version="1.2",
    )
    second = direct.promote_direct(**args)
    assert second["baseline_manifest_sha256"] == first["approved_manifest_sha256"]
    assert second["previous_pointer_generation"] == first["current_pointer_generation"]
    assert second["candidate_metrics_sha256"] is None


def test_compiled_direct_pipeline_has_no_evaluation_or_gate_and_preflight_blocks_gpu(
    tmp_path,
):
    import yaml
    from kfp import compiler

    from ml.kubeflow.pipelines.modernbert_retraining import (
        modernbert_direct_retraining_pipeline,
    )

    path = tmp_path / "direct.yaml"
    compiler.Compiler().compile(modernbert_direct_retraining_pipeline, str(path))
    ir = yaml.safe_load(path.read_text())
    tasks = ir["root"]["dag"]["tasks"]
    assert not any("evaluation" in key or "deploy-gate" in key for key in tasks)
    assert (
        "direct-promotion-preflight-component"
        in tasks["feedback-prepare-component"]["dependentTasks"]
    )
    assert (
        "feedback-prepare-component"
        in tasks["feedback-merge-component"]["dependentTasks"]
    )
    assert (
        ir["root"]["inputDefinitions"]["parameters"]["promotion_mode"]["defaultValue"]
        == "direct"
    )


def test_shared_fixture_is_verified_by_real_producer():
    from llm_router_labeling.deployment.promotion_evidence import (
        build_promotion_evidence,
    )

    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/direct-promotion-producer-v1.json"
        ).read_text()
    )

    class Store:
        def read_text(self, uri):
            return fixture["objects"].get(uri)

        def generation(self, uri):
            return fixture["generations"].get(uri)

    expected = fixture["expected"]
    result = build_promotion_evidence(
        root_uri=ROOT,
        model_version=expected["candidate_model_version"],
        batch_id=expected["batch_id"],
        pipeline_run_id=JOB,
        result=json.loads(fixture["objects"][ROOT + "/CURRENT.json"]),
        store=Store(),
        gate_text=fixture["objects"][OBS + "/promotion/direct-result.json"],
        promotion_mode="direct",
    )
    assert result == expected["evidence"]
