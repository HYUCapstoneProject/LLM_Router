from kfp import dsl

from ml.kubeflow.components.feedback_retraining import DEFAULT_DATA_PROCESSING_IMAGE
from ml.kubeflow.components.model_evaluation_promotion import (
    DEFAULT_EVALUATION_IMAGE,
    DEFAULT_PROMOTION_IMAGE,
)
from ml.kubeflow.components.routerbench_preprocessing import (
    routerbench_v2_preprocessing_component,
)
from ml.kubeflow.components.vertex_training import (
    vertex_modernbert_training_job_component,
)
from ml.kubeflow.pipelines.model_qualification import add_model_qualification_tasks
from ml.kubeflow.task_config import configure_cpu_task


@dsl.pipeline(
    name="modernbert-router-initial-training",
    description=(
        "Train ModernBERT, evaluate it, package it, and enforce a deploy gate."
    ),
)
def modernbert_training_pipeline(
    routerbench_zero_shot_uri: str = (
        "gs://replace-me/datasets/routerbench/raw/routerbench_0shot.pkl"
    ),
    routerbench_five_shot_uri: str = (
        "gs://replace-me/datasets/routerbench/raw/routerbench_5shot.pkl"
    ),
    routerbench_labeling_config_uri: str = (
        "gs://replace-me/config/routerbench_labeling_v2.json"
    ),
    training_config_uri: str = (
        "gs://replace-me/config/modernbert_t4_balanced_v2.json"
    ),
    routerarena_dataset_uri: str = "gs://replace-me/datasets/routerarena/version/evaluation.parquet",
    evaluation_config_uri: str = "gs://replace-me/config/routerarena_evaluation.json",
    package_metadata_uri: str = "gs://replace-me/config/package_metadata.json",
    baseline_metrics_uri: str = "gs://replace-me/evaluations/baseline/metrics.json",
    gate_rules_uri: str = "gs://replace-me/config/deploy_gate_rules.json",
    candidate_model_version: str = "replace-candidate-version",
    baseline_model_version: str = "replace-baseline-version",
    model_publish_root_uri: str = "gs://replace-me/models/router-modernbert",
    data_processing_image: str = DEFAULT_DATA_PROCESSING_IMAGE,
    training_service_account: str = "",
    training_network: str = "",
    training_base_output_directory: str = "",
    evaluation_image: str = DEFAULT_EVALUATION_IMAGE,
    promotion_image: str = DEFAULT_PROMOTION_IMAGE,
) -> dsl.Artifact:
    """Run the initial GPU training and shared candidate qualification DAG.

    Args:
        routerbench_zero_shot_uri: GCS URI of the verified RouterBench 0-shot Pickle.
        routerbench_five_shot_uri: GCS URI of the verified RouterBench 5-shot Pickle.
        routerbench_labeling_config_uri: GCS URI of routerbench_labeling_v2.json.
        training_config_uri: GCS URI of the versioned training JSON.
        training_service_account: Vertex CustomJob run-as service account.
        training_network: Optional full Vertex VPC network resource name.
        training_base_output_directory: Optional GCS output base for the CustomJob.

    Returns:
        Deploy gate report. The checkpoint, evaluation, and package remain MLMD artifacts.
    """
    zero_shot_importer = dsl.importer(
        artifact_uri=routerbench_zero_shot_uri,
        artifact_class=dsl.Dataset,
        reimport=False,
    ).set_display_name("Import RouterBench 0-shot source")
    five_shot_importer = dsl.importer(
        artifact_uri=routerbench_five_shot_uri,
        artifact_class=dsl.Dataset,
        reimport=False,
    ).set_display_name("Import RouterBench 5-shot source")
    labeling_config_importer = dsl.importer(
        artifact_uri=routerbench_labeling_config_uri,
        artifact_class=dsl.Artifact,
        reimport=False,
    ).set_display_name("Import RouterBench v2 labeling config")
    config_importer = dsl.importer(
        artifact_uri=training_config_uri,
        artifact_class=dsl.Artifact,
        reimport=False,
    ).set_display_name("Import training config")

    preprocessing_task = routerbench_v2_preprocessing_component(
        zero_shot_source=zero_shot_importer.output,
        five_shot_source=five_shot_importer.output,
        labeling_config=labeling_config_importer.output,
    )
    preprocessing_task.set_display_name("Build RouterBench v2 training data")
    configure_cpu_task(
        preprocessing_task,
        image=data_processing_image,
        cpu_request="2",
        cpu_limit="4",
        memory_request="4Gi",
        memory_limit="8Gi",
    )

    training_task = vertex_modernbert_training_job_component(
        training_dataset=preprocessing_task.outputs["training_dataset"],
        training_config=config_importer.output,
        training_path_suffix="/training.parquet",
        service_account=training_service_account,
        network=training_network,
        base_output_directory=training_base_output_directory,
    )
    training_task.set_display_name("Vertex T4: train ModernBERT router")
    training_task.set_caching_options(enable_caching=True)
    approval_task = add_model_qualification_tasks(
        trained_model=training_task.outputs["trained_model"],
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
