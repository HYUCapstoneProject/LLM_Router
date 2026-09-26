import os

from kfp import dsl

DEFAULT_EVALUATION_IMAGE = os.environ.get(
    "LLM_ROUTER_VERTEX_EVALUATION_IMAGE",
    (
        "us-west1-docker.pkg.dev/replace-project/llm-router/"
        "llm-router-modernbert-evaluation:cpu-v1"
    ),
)
DEFAULT_PROMOTION_IMAGE = os.environ.get(
    "LLM_ROUTER_VERTEX_PROMOTION_IMAGE",
    (
        "us-west1-docker.pkg.dev/replace-project/llm-router/"
        "llm-router-model-promotion:py312-v1"
    ),
)


@dsl.container_component
def routerarena_evaluation_component(
    trained_model: dsl.Input[dsl.Model],
    evaluation_dataset: dsl.Input[dsl.Dataset],
    evaluation_config: dsl.Input[dsl.Artifact],
    observability_run_uri: str,
    batch_id: str,
    candidate_model_version: str,
    evaluation_results: dsl.Output[dsl.Artifact],
):
    """Evaluate a candidate checkpoint on the canonical RouterArena dataset."""
    return dsl.ContainerSpec(
        image=DEFAULT_EVALUATION_IMAGE,
        command=["routerarena-evaluate"],
        args=[
            "--config",
            evaluation_config.path,
            "--evaluation-path",
            evaluation_dataset.path,
            "--model-path",
            trained_model.path,
            "--output-dir",
            evaluation_results.path,
            "--device",
            "cpu",
            "--observability-run-uri",
            observability_run_uri,
            "--batch-id",
            batch_id,
            "--candidate-model-version",
            candidate_model_version,
        ],
    )


@dsl.container_component
def model_package_component(
    trained_model: dsl.Input[dsl.Model],
    evaluation_results: dsl.Input[dsl.Artifact],
    package_metadata: dsl.Input[dsl.Artifact],
    candidate_model_version: str,
    observability_run_uri: str,
    batch_id: str,
    packaged_model: dsl.Output[dsl.Model],
):
    """Assemble and checksum a candidate package without approving deployment.

    ``package_metadata`` is one static, versioned GCS file shared by every run,
    so its own ``model_version`` field cannot track a specific run. ``--model-version``
    overrides it with this run's ``candidate_model_version`` so the manifest, the
    deploy gate, and the published GCS path all agree on the same version.
    """
    return dsl.ContainerSpec(
        image=DEFAULT_PROMOTION_IMAGE,
        command=["modernbert-package"],
        args=[
            "assemble",
            "--checkpoint-dir",
            trained_model.path,
            "--model-version",
            candidate_model_version,
            "--evaluation-metrics",
            dsl.ConcatPlaceholder(items=[evaluation_results.path, "/metrics.json"]),
            "--metadata",
            package_metadata.path,
            "--output-dir",
            packaged_model.path,
            "--observability-run-uri",
            observability_run_uri,
            "--batch-id",
            batch_id,
        ],
    )


@dsl.container_component
def model_publish_component(
    packaged_model: dsl.Input[dsl.Model],
    gate_result: dsl.Input[dsl.Artifact],
    model_publish_root_uri: str,
    candidate_model_version: str,
    publish_result: dsl.Output[dsl.Artifact],
):
    """Upload the gate-passed package to the immutable candidate GCS prefix.

    The next pipeline component re-verifies and promotes this exact publish
    result. Keeping candidate publication separate prevents a failed gate from
    moving ``CURRENT.json``; the duplicate candidate bytes are retired after
    approval and the approved manifest remains the durable audit record.
    """
    return dsl.ContainerSpec(
        image=DEFAULT_PROMOTION_IMAGE,
        command=["modernbert-publish"],
        args=[
            "publish-candidate",
            "--package-dir",
            packaged_model.path,
            "--root-uri",
            model_publish_root_uri,
            "--model-version",
            candidate_model_version,
            "--gate-result",
            dsl.ConcatPlaceholder(items=[gate_result.path, "/gate_result.json"]),
            "--output",
            dsl.ConcatPlaceholder(items=[publish_result.path, "/publish_result.json"]),
        ],
    )


@dsl.container_component
def model_auto_approve_component(
    publish_result: dsl.Input[dsl.Artifact],
    model_publish_root_uri: str,
    candidate_model_version: str,
    logical_model_version: str,
    training_mode: str,
    retention_anchor_model_version: str,
    observability_run_uri: str,
    batch_id: str,
    version_policy_mode: str,
    retraining_control_root_uri: str,
    approval_result: dsl.Output[dsl.Artifact],
):
    """Promote a gate-passed candidate and atomically move CURRENT.json.

    The deploy gate remains the safety boundary. This step removes only the
    manual approval action and records a stable automation identity in the
    approved package manifest.
    """
    return dsl.ContainerSpec(
        image=DEFAULT_PROMOTION_IMAGE,
        command=["modernbert-publish"],
        args=[
            "approve",
            "--root-uri",
            model_publish_root_uri,
            "--model-version",
            candidate_model_version,
            "--approved-by",
            "vertex-pipeline-deploy-gate",
            "--pipeline-run-id",
            dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER,
            "--logical-model-version",
            logical_model_version,
            "--training-mode",
            training_mode,
            "--retention-anchor-model-version",
            retention_anchor_model_version,
            "--publish-result",
            dsl.ConcatPlaceholder(items=[publish_result.path, "/publish_result.json"]),
            "--output",
            dsl.ConcatPlaceholder(
                items=[approval_result.path, "/approval_result.json"]
            ),
            "--observability-run-uri",
            observability_run_uri,
            "--batch-id",
            batch_id,
            "--version-policy-mode",
            version_policy_mode,
            "--retraining-control-root-uri",
            retraining_control_root_uri,
        ],
    )


@dsl.container_component
def deploy_gate_component(
    packaged_model: dsl.Input[dsl.Model],
    candidate_evaluation: dsl.Input[dsl.Artifact],
    baseline_metrics: dsl.Input[dsl.Artifact],
    gate_rules: dsl.Input[dsl.Artifact],
    candidate_model_version: str,
    baseline_model_version: str,
    model_publish_root_uri: str,
    observability_run_uri: str,
    batch_id: str,
    gate_result: dsl.Output[dsl.Artifact],
):
    """Fail the task when the candidate violates versioned promotion rules."""
    return dsl.ContainerSpec(
        image=DEFAULT_PROMOTION_IMAGE,
        command=["modernbert-deploy-gate"],
        args=[
            "--candidate-package",
            packaged_model.path,
            "--candidate-metrics",
            dsl.ConcatPlaceholder(items=[candidate_evaluation.path, "/metrics.json"]),
            "--candidate-model-version",
            candidate_model_version,
            "--baseline-metrics",
            baseline_metrics.path,
            "--baseline-model-version",
            baseline_model_version,
            "--model-publish-root-uri",
            model_publish_root_uri,
            "--rules",
            gate_rules.path,
            "--output",
            dsl.ConcatPlaceholder(items=[gate_result.path, "/gate_result.json"]),
            "--observability-run-uri",
            observability_run_uri,
            "--batch-id",
            batch_id,
            "--pipeline-run-id",
            dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER,
        ],
    )
