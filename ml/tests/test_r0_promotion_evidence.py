"""Offline connected R0 qualification: real evaluation/package/gate/publication CLIs."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from llm_router_labeling.deployment import deploy_gate_cli, model_publish_cli
from llm_router_labeling.deployment.model_package import assemble_model_package
from llm_router_labeling.deployment.model_publish import current_pointer_uri
from llm_router_labeling.evaluation.continual_gate import evaluate_continual_predictions
from llm_router_labeling.shared.dataset_io import write_json
from llm_router_labeling.shared.object_store import LocalObjectStore
from llm_router_labeling.shared.observability import write_observability_json
from test_continual_gate import ContinualGateTest
from test_model_promotion import _checkpoint, _metadata

ROOT = "gs://test/models/router"
OBS = "gs://test/runs/batch-1/observability-v1"
JOB = "projects/test/locations/test/pipelineJobs/retrain-batch-1"
VERSION = "router-r0-candidate"


def invoke(monkeypatch, module, arguments):
    monkeypatch.setattr("sys.argv", ["command", *map(str, arguments)])
    module.main()


@pytest.fixture
def chain(tmp_path, monkeypatch):
    store = LocalObjectStore(tmp_path / "objects")
    fixture = ContinualGateTest()
    fixture.setUp()
    metrics = evaluate_continual_predictions(
        fixture.frame, fixture.output, fixture.config
    ).metrics
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _checkpoint(checkpoint)
    write_json(
        checkpoint / "training_config.json",
        {"max_length": 512, "objective_method": "ordinal"},
    )
    evaluation = tmp_path / "metrics.json"
    write_json(evaluation, metrics)
    package = tmp_path / "package"
    assemble_model_package(
        checkpoint,
        package,
        {
            **_metadata(),
            "model_version": VERSION,
            "evaluation_role": "continual_gate",
            "input_contract": "structure-preserving-v2",
            "tokenizer_max_length": 512,
        },
        evaluation_metrics_path=evaluation,
    )
    rules = tmp_path / "rules.json"
    write_json(
        rules,
        {
            "rules_version": "synthetic-test-only",
            "minimum_metrics": {"accuracy": 0.4},
            "minimum_delta": {"accuracy": 0.0},
            "required_evaluation_contract": {"evaluation_role": "continual_gate"},
        },
    )
    baseline = tmp_path / "baseline.json"
    write_json(baseline, metrics)
    gate = tmp_path / "gate.json"
    publish = tmp_path / "publish.json"
    monkeypatch.setattr(
        deploy_gate_cli,
        "write_observability_json",
        lambda uri, payload: write_observability_json(uri, payload, object_store=store),
    )
    gate_args = [
        "--candidate-package",
        package,
        "--candidate-metrics",
        evaluation,
        "--candidate-model-version",
        VERSION,
        "--baseline-metrics",
        baseline,
        "--baseline-model-version",
        "base-r0",
        "--rules",
        rules,
        "--output",
        gate,
        "--observability-run-uri",
        OBS,
        "--batch-id",
        "batch-1",
        "--pipeline-run-id",
        JOB,
    ]
    publish_args = [
        "--local-store-root",
        tmp_path / "objects",
        "publish-candidate",
        "--package-dir",
        package,
        "--root-uri",
        ROOT,
        "--model-version",
        VERSION,
        "--gate-result",
        gate,
        "--output",
        publish,
    ]
    approve_args = [
        "--local-store-root",
        tmp_path / "objects",
        "approve",
        "--root-uri",
        ROOT,
        "--model-version",
        VERSION,
        "--approved-by",
        "test",
        "--pipeline-run-id",
        JOB,
        "--publish-result",
        publish,
        "--observability-run-uri",
        OBS,
        "--batch-id",
        "batch-1",
    ]
    return SimpleNamespace(**locals())


def prepare(c, monkeypatch):
    invoke(monkeypatch, deploy_gate_cli, c.gate_args)
    invoke(monkeypatch, model_publish_cli, c.publish_args)


def test_connected_gate_publish_current_evidence_and_idempotent_retry(
    chain, monkeypatch
):
    c = chain
    prepare(c, monkeypatch)
    assert c.store.read_text(current_pointer_uri(ROOT)) is None
    invoke(monkeypatch, model_publish_cli, c.approve_args)
    pointer_text = c.store.read_text(current_pointer_uri(ROOT))
    evidence = json.loads(c.store.read_text(OBS + "/result/publish-result.json"))
    assert evidence["source_schema_version"] == "router-promotion-evidence-v1"
    assert evidence["pipeline_job_resource_name"] == JOB
    assert (
        evidence["current_pointer_sha256"]
        == hashlib.sha256(pointer_text.encode()).hexdigest()
    )
    gate_text = c.store.read_text(OBS + "/gate/gate-result.json")
    assert (
        evidence["gate_result_sha256"] == hashlib.sha256(gate_text.encode()).hexdigest()
    )
    c.store.delete_uri(OBS + "/result/publish-result.json")
    invoke(monkeypatch, model_publish_cli, c.approve_args)
    assert c.store.read_text(current_pointer_uri(ROOT)) == pointer_text
    assert (
        json.loads(c.store.read_text(OBS + "/result/publish-result.json"))[
            "manifest_sha256"
        ]
        == evidence["manifest_sha256"]
    )


@pytest.mark.parametrize(
    "mutation",
    ["gate_denied", "wrong_candidate", "package_changed", "gate_missing", "wrong_job"],
)
def test_rejection_never_moves_current(chain, monkeypatch, mutation):
    c = chain
    if mutation == "gate_denied":
        write_json(c.rules, {"minimum_metrics": {"accuracy": 0.9}})
        with pytest.raises(SystemExit):
            invoke(monkeypatch, deploy_gate_cli, c.gate_args)
        with pytest.raises(SystemExit):
            invoke(monkeypatch, model_publish_cli, c.publish_args)
    else:
        invoke(monkeypatch, deploy_gate_cli, c.gate_args)
        if mutation in {"wrong_candidate", "package_changed"}:
            gate = json.loads(c.gate.read_text())
            gate[
                "candidate_model_version"
                if mutation == "wrong_candidate"
                else "candidate_manifest_sha256"
            ] = "wrong"
            write_json(c.gate, gate)
            with pytest.raises(SystemExit):
                invoke(monkeypatch, model_publish_cli, c.publish_args)
        else:
            invoke(monkeypatch, model_publish_cli, c.publish_args)
            if mutation == "gate_missing":
                c.store.delete_uri(OBS + "/gate/gate-result.json")
            else:
                c.approve_args[c.approve_args.index("--pipeline-run-id") + 1] = (
                    "other-job"
                )
            with pytest.raises(SystemExit):
                invoke(monkeypatch, model_publish_cli, c.approve_args)
    assert c.store.read_text(current_pointer_uri(ROOT)) is None
    assert c.store.read_text(OBS + "/result/publish-result.json") is None


def test_gate_rejects_unbound_evaluation_before_publication(chain, monkeypatch):
    c = chain
    changed = json.loads(c.evaluation.read_text())
    changed["accuracy"] = 1.0
    write_json(c.evaluation, changed)
    with pytest.raises(SystemExit, match="not bound"):
        invoke(monkeypatch, deploy_gate_cli, c.gate_args)
    assert c.store.read_text(current_pointer_uri(ROOT)) is None


def test_approved_baseline_metrics_must_match_manifest(chain, monkeypatch):
    c = chain
    uri = ROOT + "/base-r0/approved/model_manifest.json"
    manifest = {
        "schema_version": "model-package-v1",
        "evaluation_role": "continual_gate",
        "id2label": {"0": "easy", "1": "medium", "2": "hard"},
        "model_version": "base-r0",
        "gate_status": "approved",
        "files": {"gate_metrics.json": "wrong"},
    }
    c.store.write_text(uri, json.dumps(manifest))
    monkeypatch.setattr(deploy_gate_cli, "build_object_store", lambda root: c.store)
    args = [*c.gate_args, "--model-publish-root-uri", ROOT]
    with pytest.raises(SystemExit, match="Baseline metrics"):
        invoke(monkeypatch, deploy_gate_cli, args)
    manifest["files"]["gate_metrics.json"] = hashlib.sha256(
        c.baseline.read_bytes()
    ).hexdigest()
    c.store.write_text(
        uri, json.dumps(manifest), if_generation_match=c.store.generation(uri)
    )
    invoke(monkeypatch, deploy_gate_cli, args)
    assert json.loads(c.gate.read_text())["passed"] is True


def test_baseline_changed_before_minor_approval_is_rejected(chain, monkeypatch):
    c = chain
    prepare(c, monkeypatch)
    c.store.write_text(
        current_pointer_uri(ROOT),
        json.dumps(
            {
                "schema_version": "router-current-model-v1",
                "model_version": "different-baseline",
            }
        ),
    )
    before = c.store.read_text(current_pointer_uri(ROOT))
    with pytest.raises(SystemExit, match="baseline changed"):
        invoke(
            monkeypatch,
            model_publish_cli,
            [*c.approve_args, "--training-mode", "minor_fine_tune"],
        )
    assert c.store.read_text(current_pointer_uri(ROOT)) == before


def test_competing_current_write_during_approval_is_not_overwritten(chain, monkeypatch):
    c = chain
    prepare(c, monkeypatch)
    monkeypatch.setattr(model_publish_cli, "_object_store", lambda args: c.store)
    upload = c.store.upload_file
    competing = json.dumps({"model_version": "competitor"})

    def racing_upload(source, destination):
        upload(source, destination)
        if (
            "/approved/" in destination
            and c.store.read_text(current_pointer_uri(ROOT)) is None
        ):
            c.store.write_text(current_pointer_uri(ROOT), competing)

    monkeypatch.setattr(c.store, "upload_file", racing_upload)
    from llm_router_labeling.shared.object_store import ObjectStoreError

    with pytest.raises(ObjectStoreError, match="Generation precondition"):
        invoke(monkeypatch, model_publish_cli, c.approve_args)
    assert c.store.read_text(current_pointer_uri(ROOT)) == competing
    assert c.store.read_text(OBS + "/result/publish-result.json") is None


def test_validated_batch_requires_reservation_before_current_then_retries(
    chain, monkeypatch
):
    c = chain
    prepare(c, monkeypatch)
    control = "gs://test/control"
    args = [
        *c.approve_args,
        "--version-policy-mode",
        "validatedUniqueBatch",
        "--retraining-control-root-uri",
        control,
    ]
    with pytest.raises(SystemExit, match="reservation"):
        invoke(monkeypatch, model_publish_cli, args)
    assert c.store.read_text(current_pointer_uri(ROOT)) is None
    pending = control + "/validated-batches/pending/digest.json"
    c.store.write_text(
        pending,
        json.dumps(
            {
                "batch_id": "batch-1",
                "candidate_model_version": VERSION,
                "canonical_batch_digest": "digest",
            }
        ),
    )
    invoke(monkeypatch, model_publish_cli, args)
    before = c.store.read_text(current_pointer_uri(ROOT))
    invoke(monkeypatch, model_publish_cli, args)
    assert c.store.read_text(current_pointer_uri(ROOT)) == before
    assert c.store.read_text(pending) is None
    # Recovery of the crash boundary between committed write and pending delete.
    c.store.write_text(
        pending,
        json.dumps(
            {
                "batch_id": "batch-1",
                "candidate_model_version": VERSION,
                "canonical_batch_digest": "digest",
            }
        ),
    )
    invoke(monkeypatch, model_publish_cli, args)
    assert c.store.read_text(pending) is None
    assert c.store.read_text(current_pointer_uri(ROOT)) == before


def test_baseline_race_between_cli_check_and_approve_entry_is_rejected(
    chain, monkeypatch
):
    c = chain
    prepare(c, monkeypatch)
    uri = current_pointer_uri(ROOT)
    baseline = {"schema_version": "router-current-model-v1", "model_version": "base-r0"}
    c.store.write_text(uri, json.dumps(baseline))
    real_approve = model_publish_cli.approve_candidate

    def race_before_approve(**kwargs):
        c.store.write_text(
            uri,
            json.dumps({**baseline, "model_version": "competitor"}),
            if_generation_match=c.store.generation(uri),
        )
        return real_approve(**kwargs)

    monkeypatch.setattr(model_publish_cli, "approve_candidate", race_before_approve)
    from llm_router_labeling.deployment.model_publish import ModelPublishError

    with pytest.raises(ModelPublishError, match="qualified baseline"):
        invoke(
            monkeypatch,
            model_publish_cli,
            [*c.approve_args, "--training-mode", "minor_fine_tune"],
        )
    assert json.loads(c.store.read_text(uri))["model_version"] == "competitor"
    assert c.store.read_text(OBS + "/result/publish-result.json") is None


@pytest.mark.parametrize("fault", ["v2", "wrong_filename", "wrong_role"])
def test_baseline_rejects_candidate_head_or_nonmetric_hash(chain, monkeypatch, fault):
    c = chain
    manifest = {
        "schema_version": "model-package-v1",
        "evaluation_role": "continual_gate",
        "id2label": {"0": "easy", "1": "medium", "2": "hard"},
        "model_version": "base-r0",
        "gate_status": "approved",
        "files": {
            "gate_metrics.json": hashlib.sha256(c.baseline.read_bytes()).hexdigest()
        },
    }
    if fault == "v2":
        manifest.update(schema_version="model-package-v2", objective="pointwise")
    elif fault == "wrong_role":
        manifest["evaluation_role"] = "routerarena"
    else:
        manifest["files"]["training_config.json"] = manifest["files"].pop(
            "gate_metrics.json"
        )
    c.store.write_text(
        ROOT + "/base-r0/approved/model_manifest.json", json.dumps(manifest)
    )
    monkeypatch.setattr(deploy_gate_cli, "build_object_store", lambda root: c.store)
    with pytest.raises(SystemExit):
        invoke(
            monkeypatch,
            deploy_gate_cli,
            [*c.gate_args, "--model-publish-root-uri", ROOT],
        )
    assert c.store.read_text(OBS + "/gate/gate-result.json") is None
