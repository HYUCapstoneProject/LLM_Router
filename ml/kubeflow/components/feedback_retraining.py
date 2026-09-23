import os

from kfp import dsl

DEFAULT_DATA_PROCESSING_IMAGE = os.environ.get(
    "LLM_ROUTER_VERTEX_DATA_PROCESSING_IMAGE",
    (
        "us-west1-docker.pkg.dev/replace-project/llm-router/"
        "llm-router-data-processing:py312-v2"
    ),
)
PREPARED_FEEDBACK_FILENAME = "feedback_training.parquet"


@dsl.container_component
def feedback_prepare_component(
    feedback_manifest: dsl.Input[dsl.Artifact],
    feedback_events: dsl.Input[dsl.Dataset],
    preprocessing_config: dsl.Input[dsl.Artifact],
    observability_run_uri: str,
    batch_id: str,
    candidate_model_version: str,
    version_policy_mode: str,
    retraining_control_root_uri: str,
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
            "--observability-run-uri",
            observability_run_uri,
            "--batch-id",
            batch_id,
            "--candidate-model-version",
            candidate_model_version,
            "--version-policy-mode",
            version_policy_mode,
            "--retraining-control-root-uri",
            retraining_control_root_uri,
        ],
    )


@dsl.container_component
def feedback_merge_component(
    base_training_dataset: dsl.Input[dsl.Dataset],
    prepared_feedback: dsl.Input[dsl.Dataset],
    preprocessing_config: dsl.Input[dsl.Artifact],
    training_data_strategy: str,
    observability_run_uri: str,
    batch_id: str,
    candidate_model_version: str,
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
            "--training-data-strategy",
            training_data_strategy,
            "--observability-run-uri",
            observability_run_uri,
            "--batch-id",
            batch_id,
            "--candidate-model-version",
            candidate_model_version,
        ],
    )
