from kfp import dsl

from ml.kubeflow.components.model_evaluation_promotion import (
    DEFAULT_EVALUATION_IMAGE,
    DEFAULT_PROMOTION_IMAGE,
    deploy_gate_component,
    model_package_component,
    routerarena_evaluation_component,
)
from ml.kubeflow.task_config import configure_cpu_task, configure_gpu_evaluation_task


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
    evaluation_image: str,
    promotion_image: str,
):
    """Attach the shared evaluation, package, and gate DAG to a trained model."""
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
    )
    evaluation_task.set_display_name("Evaluate candidate on RouterArena")
    configure_gpu_evaluation_task(evaluation_task, image=evaluation_image)

    package_task = model_package_component(
        trained_model=trained_model,
        evaluation_results=evaluation_task.outputs["evaluation_results"],
        package_metadata=package_metadata.output,
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
    return gate_task


@dsl.pipeline(
    name="modernbert-router-model-qualification",
    description=(
        "Evaluate a trained model, assemble a checksum package, and enforce a deploy gate."
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
    evaluation_image: str = DEFAULT_EVALUATION_IMAGE,
    promotion_image: str = DEFAULT_PROMOTION_IMAGE,
) -> dsl.Artifact:
    """Run the backend-independent model qualification path on GKE."""
    trained_model = dsl.importer(
        artifact_uri=trained_model_uri,
        artifact_class=dsl.Model,
        reimport=False,
    ).set_display_name("Import trained model")
    gate_task = add_model_qualification_tasks(
        trained_model=trained_model.output,
        routerarena_dataset_uri=routerarena_dataset_uri,
        evaluation_config_uri=evaluation_config_uri,
        package_metadata_uri=package_metadata_uri,
        baseline_metrics_uri=baseline_metrics_uri,
        gate_rules_uri=gate_rules_uri,
        candidate_model_version=candidate_model_version,
        baseline_model_version=baseline_model_version,
        evaluation_image=evaluation_image,
        promotion_image=promotion_image,
    )
    return gate_task.outputs["gate_result"]
