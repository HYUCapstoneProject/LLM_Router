from kfp import dsl


DEFAULT_DATA_PROCESSING_IMAGE = "llm-router-data-processing:py312-v2"
PREPARED_FEEDBACK_FILENAME = "feedback_training.parquet"


@dsl.container_component
def feedback_prepare_component(
    feedback_manifest: dsl.Input[dsl.Artifact],
    feedback_events: dsl.Input[dsl.Dataset],
    preprocessing_config: dsl.Input[dsl.Artifact],
    prepared_feedback: dsl.Output[dsl.Dataset],
):
    """Validate an exported feedback batch and create canonical training rows."""
    return dsl.ContainerSpec(
        image=DEFAULT_DATA_PROCESSING_IMAGE,
        command=["escalation-feedback-preprocessing"],
        args=[
            "--config",
            preprocessing_config.path,
            "prepare",
            "--manifest-path",
            feedback_manifest.path,
            "--events-path",
            feedback_events.path,
            "--output-dir",
            prepared_feedback.path,
        ],
    )


@dsl.container_component
def feedback_merge_component(
    base_training_dataset: dsl.Input[dsl.Dataset],
    prepared_feedback: dsl.Input[dsl.Dataset],
    preprocessing_config: dsl.Input[dsl.Artifact],
    merged_training_dataset: dsl.Output[dsl.Dataset],
):
    """Merge canonical feedback into base training while preserving holdouts."""
    return dsl.ContainerSpec(
        image=DEFAULT_DATA_PROCESSING_IMAGE,
        command=["escalation-feedback-preprocessing"],
        args=[
            "--config",
            preprocessing_config.path,
            "merge",
            "--base-training-path",
            base_training_dataset.path,
            "--feedback-training-path",
            dsl.ConcatPlaceholder(
                items=[prepared_feedback.path, f"/{PREPARED_FEEDBACK_FILENAME}"]
            ),
            "--output-dir",
            merged_training_dataset.path,
        ],
    )
