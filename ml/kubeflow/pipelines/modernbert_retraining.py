from kfp import dsl

from ml.kubeflow.components.feedback_retraining import (
    DEFAULT_DATA_PROCESSING_IMAGE,
    feedback_merge_component,
    feedback_prepare_component,
)
from ml.kubeflow.components.modernbert_training import (
    DEFAULT_TRAINING_IMAGE,
    modernbert_training_component,
)
from ml.kubeflow.task_config import (
    configure_cpu_task,
    configure_gpu_training_task,
)
from ml.kubeflow.components.model_evaluation_promotion import (
    DEFAULT_EVALUATION_IMAGE,
    DEFAULT_PROMOTION_IMAGE,
)
from ml.kubeflow.pipelines.model_qualification import add_model_qualification_tasks


@dsl.pipeline(
    name="modernbert-router-feedback-retraining",
    description=(
        "Prepare feedback, retrain ModernBERT, and enforce the shared qualification gate."
    ),
)
def modernbert_retraining_pipeline(
    base_training_dataset_uri: str = (
        "gs://replace-me/datasets/routerbench/version/training.parquet"
    ),
    feedback_manifest_uri: str = (
        "gs://replace-me/datasets/feedback/batch/manifest.json"
    ),
    feedback_events_uri: str = (
        "gs://replace-me/datasets/feedback/batch/events.parquet"
    ),
    feedback_preprocessing_config_uri: str = (
        "gs://replace-me/config/feedback_retraining_preprocessing.json"
    ),
    training_config_uri: str = "gs://replace-me/config/modernbert_t4_full.json",
    routerarena_dataset_uri: str = "gs://replace-me/datasets/routerarena/version/evaluation.parquet",
    evaluation_config_uri: str = "gs://replace-me/config/routerarena_evaluation.json",
    package_metadata_uri: str = "gs://replace-me/config/package_metadata.json",
    baseline_metrics_uri: str = "gs://replace-me/evaluations/baseline/metrics.json",
    gate_rules_uri: str = "gs://replace-me/config/deploy_gate_rules.json",
    candidate_model_version: str = "replace-candidate-version",
    baseline_model_version: str = "replace-baseline-version",
    data_processing_image: str = DEFAULT_DATA_PROCESSING_IMAGE,
    training_image: str = DEFAULT_TRAINING_IMAGE,
    evaluation_image: str = DEFAULT_EVALUATION_IMAGE,
    promotion_image: str = DEFAULT_PROMOTION_IMAGE,
) -> dsl.Artifact:
    """Run feedback preparation, GPU retraining, evaluation, package, and gate."""
    base_dataset_importer = dsl.importer(
        artifact_uri=base_training_dataset_uri,
        artifact_class=dsl.Dataset,
        reimport=False,
    ).set_display_name("Import base training dataset")
    feedback_manifest_importer = dsl.importer(
        artifact_uri=feedback_manifest_uri,
        artifact_class=dsl.Artifact,
        reimport=False,
    ).set_display_name("Import feedback manifest")
    feedback_events_importer = dsl.importer(
        artifact_uri=feedback_events_uri,
        artifact_class=dsl.Dataset,
        reimport=False,
    ).set_display_name("Import feedback events")
    feedback_config_importer = dsl.importer(
        artifact_uri=feedback_preprocessing_config_uri,
        artifact_class=dsl.Artifact,
        reimport=False,
    ).set_display_name("Import feedback preprocessing config")
    training_config_importer = dsl.importer(
        artifact_uri=training_config_uri,
        artifact_class=dsl.Artifact,
        reimport=False,
    ).set_display_name("Import training config")

    prepare_task = feedback_prepare_component(
        feedback_manifest=feedback_manifest_importer.output,
        feedback_events=feedback_events_importer.output,
        preprocessing_config=feedback_config_importer.output,
    )
    prepare_task.set_display_name("Validate and normalize feedback")
    configure_cpu_task(
        prepare_task,
        image=data_processing_image,
        cpu_request="1",
        cpu_limit="2",
        memory_request="2Gi",
        memory_limit="4Gi",
    )

    merge_task = feedback_merge_component(
        base_training_dataset=base_dataset_importer.output,
        prepared_feedback=prepare_task.outputs["prepared_feedback"],
        preprocessing_config=feedback_config_importer.output,
    )
    merge_task.set_display_name("Merge feedback with base training data")
    configure_cpu_task(
        merge_task,
        image=data_processing_image,
        cpu_request="2",
        cpu_limit="4",
        memory_request="4Gi",
        memory_limit="8Gi",
    )

    training_task = modernbert_training_component(
        training_dataset=merge_task.outputs["merged_training_dataset"],
        training_config=training_config_importer.output,
        training_path_suffix="/training.parquet",
    )
    training_task.set_display_name("Retrain ModernBERT router")
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
