"""Producer contracts shared with the FastAPI package loader."""

import json
from pathlib import Path

import pytest
from llm_router_labeling.deployment.model_package import (
    ModelPackageError,
    assemble_model_package,
    validate_model_package,
)
from llm_router_labeling.shared.dataset_io import sha256_file
from llm_router_labeling.training.routing_policy import (
    LENGTH_CONTRACT_VERSION,
    RoutingPolicy,
    RoutingPolicyError,
    compute_policy_sha256,
)
from test_model_package_policy import policy_checkpoint, policy_metadata
from test_model_promotion import _metadata, _write_json
from test_routing_policy import fixture_policy_payload

FIXTURE_PATH = Path(__file__).parent / "fixtures/length-contract-producer-v1.json"


def producer_inputs(root, *, schema="v2", inference_length=8192, training_length=512):
    objective = "pointwise" if schema == "v2" else "ordinal"
    checkpoint = root / "checkpoint"
    policy_checkpoint(checkpoint, objective)
    config = json.loads((checkpoint / "config.json").read_text())
    config["max_position_embeddings"] = 8192
    _write_json(checkpoint / "config.json", config)
    _write_json(
        checkpoint / "training_config.json",
        {
            "max_length": training_length,
            "objective_method": objective,
            "input_contract": "structure-preserving-v2",
        },
    )
    policy_path = None
    metadata = _metadata()
    if schema == "v2":
        policy = fixture_policy_payload(objective)
        policy.update(
            length_contract_version=LENGTH_CONTRACT_VERSION, max_length=inference_length
        )
        policy["policy_sha256"] = compute_policy_sha256(policy)
        policy_path = root / "policy.json"
        _write_json(policy_path, policy)
        metadata = policy_metadata(policy_path, objective)
    metadata.update(
        model_version=f"router-synthetic-{schema}-train{training_length}-infer{inference_length}",
        created_at="2026-09-26T00:00:00+00:00",
        input_contract="structure-preserving-v2",
        length_contract_version=LENGTH_CONTRACT_VERSION,
        training_max_length=training_length,
        inference_max_length=inference_length,
        tokenizer_max_length=inference_length,
    )
    return checkpoint, metadata, policy_path


def assemble(root, **kwargs):
    checkpoint, metadata, policy = producer_inputs(root, **kwargs)
    package = root / "package"
    manifest = assemble_model_package(
        checkpoint, package, metadata, routing_policy_path=policy
    )
    return checkpoint, package, manifest


def producer_fixture(root):
    packages = {}
    for schema in ("v1", "v2"):
        _, package, manifest = assemble(root / schema, schema=schema)
        packages[schema] = {
            "model_version": manifest["model_version"],
            "files": {path.name: path.read_text() for path in sorted(package.iterdir())},
        }
    return {
        "schema_version": "length-contract-producer-fixture-v1",
        "synthetic_only": True,
        "producer": "llm_router_labeling.deployment.model_package.assemble_model_package",
        "packages": packages,
    }


@pytest.mark.parametrize("schema", ["v1", "v2"])
@pytest.mark.parametrize("training_length", [512, 8192])
@pytest.mark.parametrize("inference_length", [512, 8192])
def test_producer_separates_lengths_without_modifying_training_history(
    tmp_path, schema, training_length, inference_length
):
    checkpoint, metadata, policy = producer_inputs(
        tmp_path, schema=schema, training_length=training_length,
        inference_length=inference_length,
    )
    before = {path.name: sha256_file(path) for path in checkpoint.iterdir()}
    package = tmp_path / "package"
    manifest = assemble_model_package(
        checkpoint, package, metadata, routing_policy_path=policy
    )
    assert manifest["training_max_length"] == training_length
    assert manifest["inference_max_length"] == inference_length
    assert manifest["tokenizer_max_length"] == inference_length
    assert manifest["length_contract_version"] == LENGTH_CONTRACT_VERSION
    assert manifest["gate_status"] == "candidate"
    assert before == {path.name: sha256_file(path) for path in checkpoint.iterdir()}
    assert sha256_file(package / "training_config.json") == before["training_config.json"]
    assert validate_model_package(package)["inference_max_length"] == inference_length


@pytest.mark.parametrize("schema", ["v1", "v2"])
@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("metadata", "length_contract_version", None),
        ("metadata", "length_contract_version", "unknown"),
        ("metadata", "training_max_length", True),
        ("metadata", "training_max_length", "512"),
        ("metadata", "training_max_length", 8192),
        ("metadata", "inference_max_length", "8192"),
        ("metadata", "inference_max_length", True),
        ("metadata", "inference_max_length", 4096),
        ("metadata", "tokenizer_max_length", 512),
        ("metadata", "tokenizer_max_length", 8192.0),
        ("metadata", "input_contract", "legacy-word-only-v1"),
        ("training_config.json", "max_length", 512.0),
        ("training_config.json", "max_length", True),
        ("training_config.json", "input_contract", "legacy-word-only-v1"),
        ("config.json", "max_position_embeddings", 4096),
        ("config.json", "max_position_embeddings", "8192"),
        ("config.json", "max_position_embeddings", True),
        ("config.json", "max_position_embeddings", None),
    ],
)
def test_producer_rejects_inconsistent_or_nonstrict_length_metadata(
    tmp_path, schema, target, field, value
):
    checkpoint, metadata, policy = producer_inputs(tmp_path, schema=schema)
    if target == "metadata":
        metadata[field] = value
    else:
        payload = json.loads((checkpoint / target).read_text())
        payload[field] = value
        _write_json(checkpoint / target, payload)
    with pytest.raises(ModelPackageError):
        assemble_model_package(
            checkpoint, tmp_path / "package", metadata, routing_policy_path=policy
        )


@pytest.mark.parametrize("field", ["length_contract_version", "training_max_length", "inference_max_length"])
def test_producer_rejects_partial_length_contract(tmp_path, field):
    checkpoint, metadata, policy = producer_inputs(tmp_path)
    del metadata[field]
    with pytest.raises(ModelPackageError):
        assemble_model_package(
            checkpoint, tmp_path / "package", metadata, routing_policy_path=policy
        )


def test_policy_marker_must_match_the_package(tmp_path):
    checkpoint, metadata, policy = producer_inputs(tmp_path, inference_length=512)
    _write_json(policy, fixture_policy_payload())
    metadata["routing_policy_sha256"] = sha256_file(policy)
    with pytest.raises(ModelPackageError, match="input contract"):
        assemble_model_package(
            checkpoint, tmp_path / "package", metadata, routing_policy_path=policy
        )


def test_revalidation_checks_length_fields_after_file_hashes(tmp_path):
    _, package, manifest = assemble(tmp_path)
    manifest["training_max_length"] = 8192
    _write_json(package / "model_manifest.json", manifest)
    with pytest.raises(ModelPackageError, match="training history"):
        validate_model_package(package)


def test_length_policy_hash_matches_the_serving_contract():
    policy = fixture_policy_payload()
    policy.update(length_contract_version=LENGTH_CONTRACT_VERSION, max_length=8192)
    policy["policy_sha256"] = compute_policy_sha256(policy)
    assert policy["policy_sha256"] == (
        "3a4294af6a02b5c856fddc318735ac321b9bdee51db9022f9a9c381b70e42f2e"
    )
    assert RoutingPolicy.from_mapping(policy).max_length == 8192
    del policy["length_contract_version"]
    with pytest.raises(RoutingPolicyError):
        compute_policy_sha256(policy)


def test_producer_fixture_reproduces_actual_assembly_bytes(tmp_path):
    assert producer_fixture(tmp_path) == json.loads(FIXTURE_PATH.read_text())
