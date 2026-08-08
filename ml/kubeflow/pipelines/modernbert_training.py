from kfp import dsl

from ml.kubeflow.components.modernbert_training import (
    DEFAULT_TRAINING_IMAGE,
    modernbert_training_component,
)


@dsl.pipeline(
    name="modernbert-router-training",
    description="Import a canonical router dataset and train ModernBERT on one GPU.",
)
def modernbert_training_pipeline(
    training_dataset_uri: str = "gs://replace-me/datasets/router/training.parquet",
    training_config_uri: str = "gs://replace-me/config/modernbert_t4_full.json",
    training_image: str = DEFAULT_TRAINING_IMAGE,
) -> dsl.Model:
    """Run a single-GPU ModernBERT training task.

    Args:
        training_dataset_uri: GCS URI of the canonical six-column Parquet.
        training_config_uri: GCS URI of the versioned training JSON.
        training_image: Artifact Registry URI or local tag of the GPU image.

    Returns:
        Output: Hugging Face checkpoint and metrics as a KFP Model Artifact.
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
    )
    training_task.set_display_name("Train ModernBERT router")
    training_task.set_container_image(training_image)
    training_task.set_accelerator_type("nvidia.com/gpu")
    training_task.set_accelerator_limit(1)
    training_task.set_cpu_request("4")
    training_task.set_cpu_limit("8")
    training_task.set_memory_request("24Gi")
    training_task.set_memory_limit("32Gi")
    training_task.set_retry(
        num_retries=2,
        backoff_duration="60s",
        backoff_factor=2.0,
        backoff_max_duration="600s",
    )
    training_task.set_caching_options(enable_caching=True)
    return training_task.outputs["trained_model"]
