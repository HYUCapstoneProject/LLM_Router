from kfp import dsl

from ml.kubeflow.components.modernbert_training import (
    DEFAULT_TRAINING_IMAGE,
    modernbert_training_component,
)
from ml.kubeflow.task_config import configure_gpu_training_task


@dsl.pipeline(
    name="modernbert-router-initial-training",
    description=(
        "Import a locally prepared canonical dataset and train ModernBERT on one GPU."
    ),
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
        training_path_suffix="",
    )
    training_task.set_display_name("Train ModernBERT router")
    configure_gpu_training_task(training_task, image=training_image)
    return training_task.outputs["trained_model"]
