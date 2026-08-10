from kfp import dsl

from ml.kubeflow.components.modernbert_training import (
    DEFAULT_TRAINING_IMAGE,
    modernbert_training_component,
)
from ml.kubeflow.task_config import configure_gpu_training_task
from ml.kubeflow.components.model_evaluation_promotion import (
    DEFAULT_EVALUATION_IMAGE,
    DEFAULT_PROMOTION_IMAGE,
)
from ml.kubeflow.pipelines.model_qualification import add_model_qualification_tasks


@dsl.pipeline(
    name="modernbert-router-initial-training",
    description=(
        "Train ModernBERT, evaluate it, package it, and enforce a deploy gate."
    ),
)
def modernbert_training_pipeline(
    training_dataset_uri: str = (
        "gs://replace-me/datasets/routerbench/version/training.parquet"
    ),
    training_config_uri: str = "gs://replace-me/config/modernbert_t4_full.json",
    routerarena_dataset_uri: str = "gs://replace-me/datasets/routerarena/version/evaluation.parquet",
    evaluation_config_uri: str = "gs://replace-me/config/routerarena_evaluation.json",
    package_metadata_uri: str = "gs://replace-me/config/package_metadata.json",
    baseline_metrics_uri: str = "gs://replace-me/evaluations/baseline/metrics.json",
    gate_rules_uri: str = "gs://replace-me/config/deploy_gate_rules.json",
    candidate_model_version: str = "replace-candidate-version",
    baseline_model_version: str = "replace-baseline-version",
    training_image: str = DEFAULT_TRAINING_IMAGE,
    evaluation_image: str = DEFAULT_EVALUATION_IMAGE,
    promotion_image: str = DEFAULT_PROMOTION_IMAGE,
) -> dsl.Artifact:
    """Run the initial GPU training and shared candidate qualification DAG.

    Args:
        training_dataset_uri: GCS URI of the canonical six-column Parquet.
        training_config_uri: GCS URI of the versioned training JSON.
        training_image: Artifact Registry URI or local tag of the GPU image.

    Returns:
        Deploy gate report. The checkpoint, evaluation, and package remain MLMD artifacts.
    """
    dataset_importer = dsl.importer(
        artifact_uri=training_dataset_uri,
        artifact_class=dsl.Dataset,
        reimport=False,
    ).set_display_name("Import training dataset")
    config_importer = dsl.importer(
        artifact_uri=training_config_uri,
        artifact_class=dsl.Artifact,
        reimport=False,
    ).set_display_name("Import training config")

    training_task = modernbert_training_component(
        training_dataset=dataset_importer.output,
        training_config=config_importer.output,
        training_path_suffix="",
    )
    training_task.set_display_name("Train ModernBERT router")
    configure_gpu_training_task(training_task, image=training_image)
    gate_task = add_model_qualification_tasks(
        trained_model=training_task.outputs["trained_model"],
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
