from kfp import dsl

from ml.kubeflow.components.model_evaluation_promotion import (
    DEFAULT_EVALUATION_IMAGE,
    DEFAULT_PROMOTION_IMAGE,
    deploy_gate_component,
    model_auto_approve_component,
    model_package_component,
    model_publish_component,
    routerarena_evaluation_component,
)
from ml.kubeflow.task_config import configure_cpu_task


def add_model_qualification_tasks(
    *,
    trained_model,
    routerarena_dataset_uri: str,
    evaluation_config_uri: str,
    package_metadata_uri: str,
    baseline_metrics_uri: str,
    gate_rules_uri: str,
    candidate_model_version: str,
    baseline_model_version: str,
    model_publish_root_uri: str,
    evaluation_image: str,
    promotion_image: str,
    observability_run_uri: str = "",
    batch_id: str = "initial-training",
    version_policy_mode: str = "successfulHttpRequest",
    retraining_control_root_uri: str = "",
    logical_model_version: str = "1.0",
    training_mode: str = "initial_training",
    retention_anchor_model_version: str = "",
):
    """Attach the shared evaluate, package, gate, and publish DAG to a trained model."""
    routerarena_dataset = dsl.importer(
        artifact_uri=routerarena_dataset_uri,
        artifact_class=dsl.Dataset,
        reimport=False,
    ).set_display_name("Import RouterArena evaluation data")
    evaluation_config = dsl.importer(
        artifact_uri=evaluation_config_uri,
        artifact_class=dsl.Artifact,
        reimport=False,
    ).set_display_name("Import evaluation config")
    package_metadata = dsl.importer(
        artifact_uri=package_metadata_uri,
        artifact_class=dsl.Artifact,
        reimport=False,
    ).set_display_name("Import package metadata")
    baseline_metrics = dsl.importer(
        artifact_uri=baseline_metrics_uri,
        artifact_class=dsl.Artifact,
        reimport=False,
    ).set_display_name("Import approved baseline metrics")
    gate_rules = dsl.importer(
        artifact_uri=gate_rules_uri,
        artifact_class=dsl.Artifact,
        reimport=False,
    ).set_display_name("Import deploy gate rules")

    evaluation_task = routerarena_evaluation_component(
        trained_model=trained_model,
        evaluation_dataset=routerarena_dataset.output,
        evaluation_config=evaluation_config.output,
        observability_run_uri=observability_run_uri,
        batch_id=batch_id,
        candidate_model_version=candidate_model_version,
    )
    evaluation_task.set_display_name("Evaluate candidate on RouterArena")
    configure_cpu_task(
        evaluation_task,
        image=evaluation_image,
        cpu_request="4",
        cpu_limit="8",
        memory_request="16Gi",
        memory_limit="32Gi",
    )

    package_task = model_package_component(
        trained_model=trained_model,
        evaluation_results=evaluation_task.outputs["evaluation_results"],
        package_metadata=package_metadata.output,
        candidate_model_version=candidate_model_version,
        observability_run_uri=observability_run_uri,
        batch_id=batch_id,
    )
    package_task.set_display_name("Assemble candidate model package")
    configure_cpu_task(
        package_task,
        image=promotion_image,
        cpu_request="1",
        cpu_limit="2",
        memory_request="1Gi",
        memory_limit="2Gi",
    )

    gate_task = deploy_gate_component(
        packaged_model=package_task.outputs["packaged_model"],
        candidate_evaluation=evaluation_task.outputs["evaluation_results"],
        baseline_metrics=baseline_metrics.output,
        gate_rules=gate_rules.output,
        candidate_model_version=candidate_model_version,
        baseline_model_version=baseline_model_version,
        observability_run_uri=observability_run_uri,
        batch_id=batch_id,
    )
    gate_task.set_display_name("Check candidate deploy gate")
    configure_cpu_task(
        gate_task,
        image=promotion_image,
        cpu_request="1",
        cpu_limit="1",
        memory_request="512Mi",
        memory_limit="1Gi",
    )

    publish_task = model_publish_component(
        packaged_model=package_task.outputs["packaged_model"],
        gate_result=gate_task.outputs["gate_result"],
        model_publish_root_uri=model_publish_root_uri,
        candidate_model_version=candidate_model_version,
    )
    publish_task.set_display_name("Publish candidate package to GCS")
    configure_cpu_task(
        publish_task,
        image=promotion_image,
        cpu_request="1",
        cpu_limit="2",
        memory_request="1Gi",
        memory_limit="2Gi",
    )
    # Publishing writes to a run-specific candidate prefix, so a cached re-run
    # would skip the upload the deployment depends on.
    publish_task.set_caching_options(enable_caching=False)

    approval_task = model_auto_approve_component(
        publish_result=publish_task.outputs["publish_result"],
        model_publish_root_uri=model_publish_root_uri,
        candidate_model_version=candidate_model_version,
        logical_model_version=logical_model_version,
        training_mode=training_mode,
        retention_anchor_model_version=retention_anchor_model_version,
        observability_run_uri=observability_run_uri,
        batch_id=batch_id,
        version_policy_mode=version_policy_mode,
        retraining_control_root_uri=retraining_control_root_uri,
    )
    approval_task.set_display_name("Automatically approve gate-passed model")
    configure_cpu_task(
        approval_task,
        image=promotion_image,
        cpu_request="1",
        cpu_limit="2",
        memory_request="1Gi",
        memory_limit="2Gi",
    )
    # Updating CURRENT.json is a side effect and must execute for every new run.
    approval_task.set_caching_options(enable_caching=False)
    return approval_task


@dsl.pipeline(
    name="modernbert-router-model-qualification",
    description=(
        "Evaluate a trained model, assemble a checksum package, enforce a deploy gate, "
        "publish it, and automatically promote a passing candidate."
    ),
)
def model_qualification_pipeline(
    trained_model_uri: str = "gs://replace-me/models/router/candidate/checkpoint/",
    routerarena_dataset_uri: str = (
        "gs://replace-me/datasets/routerarena/version/evaluation.parquet"
    ),
    evaluation_config_uri: str = "gs://replace-me/config/routerarena_evaluation.json",
    package_metadata_uri: str = "gs://replace-me/config/package_metadata.json",
    baseline_metrics_uri: str = (
        "gs://replace-me/evaluations/approved-version/routerarena/metrics.json"
    ),
    gate_rules_uri: str = "gs://replace-me/config/deploy_gate_rules.json",
    candidate_model_version: str = "replace-candidate-version",
    baseline_model_version: str = "replace-approved-version",
    model_publish_root_uri: str = "gs://replace-me/models/router-modernbert",
    evaluation_image: str = DEFAULT_EVALUATION_IMAGE,
    promotion_image: str = DEFAULT_PROMOTION_IMAGE,
) -> dsl.Artifact:
    """Run model qualification as CPU tasks in Vertex AI Pipelines."""
    trained_model = dsl.importer(
        artifact_uri=trained_model_uri,
        artifact_class=dsl.Model,
        reimport=False,
    ).set_display_name("Import trained model")
    approval_task = add_model_qualification_tasks(
        trained_model=trained_model.output,
        routerarena_dataset_uri=routerarena_dataset_uri,
        evaluation_config_uri=evaluation_config_uri,
        package_metadata_uri=package_metadata_uri,
        baseline_metrics_uri=baseline_metrics_uri,
        gate_rules_uri=gate_rules_uri,
        candidate_model_version=candidate_model_version,
        baseline_model_version=baseline_model_version,
        model_publish_root_uri=model_publish_root_uri,
        evaluation_image=evaluation_image,
        promotion_image=promotion_image,
    )
    return approval_task.outputs["approval_result"]
