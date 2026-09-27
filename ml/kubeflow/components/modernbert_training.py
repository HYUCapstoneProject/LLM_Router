import os

from kfp import dsl

DEFAULT_TRAINING_IMAGE = os.environ.get(
    "LLM_ROUTER_VERTEX_TRAINING_IMAGE",
    (
        "us-west1-docker.pkg.dev/replace-project/llm-router/"
        "llm-router-modernbert-training:cu124-v1"
    ),
)
RUNTIME_CONTRACT_PATH = "/opt/llm-router/config/modernbert_runtime_cu124_v1.json"
# Vertex CustomJob conversion leaves nested ConcatPlaceholder JSON unresolved.
# Pass the two plain placeholders separately and concatenate in the worker,
# without shell interpolation or changes to the training image/CLI.
TRAINING_ENTRYPOINT = [
    "python3",
    "-c",
    (
        "import os, sys; os.execvp('modernbert-train', "
        "['modernbert-train', '--training-path', sys.argv[1] + sys.argv[2], *sys.argv[3:]])"
    ),
]



@dsl.container_component
def modernbert_training_component(
    training_dataset: dsl.Input[dsl.Dataset],
    training_config: dsl.Input[dsl.Artifact],
    training_path_suffix: str,
    trained_model: dsl.Output[dsl.Model],
):
    """Train the three-class ModernBERT router with the shared GPU image.

    Args:
        training_dataset: Canonical training Parquet with train/validation splits.
        training_config: Versioned ModernBERT training JSON configuration.
        training_path_suffix: Empty for a direct Parquet artifact, or the bundle
            filename prefixed with a slash for a directory artifact.
        trained_model: Hugging Face checkpoint directory including training metrics.
    """
    return dsl.ContainerSpec(
        image=DEFAULT_TRAINING_IMAGE,
        command=TRAINING_ENTRYPOINT,
        args=[
            training_dataset.path,
            training_path_suffix,
            "--config",
            training_config.path,
            "--runtime-contract",
            RUNTIME_CONTRACT_PATH,
            "--output-dir",
            trained_model.path,
            "--device",
            "cuda",
        ],
    )


@dsl.container_component
def modernbert_continual_training_component(
    training_dataset: dsl.Input[dsl.Dataset],
    training_config: dsl.Input[dsl.Artifact],
    initial_model: dsl.Input[dsl.Model],
    initialization_mode: str,
    observability_run_uri: str,
    batch_id: str,
    candidate_model_version: str,
    training_path_suffix: str,
    trained_model: dsl.Output[dsl.Model],
):
    """Warm-start from an approved anchor, or periodically rebase from pristine."""
    return dsl.ContainerSpec(
        image=DEFAULT_TRAINING_IMAGE,
        command=TRAINING_ENTRYPOINT,
        args=[
            training_dataset.path,
            training_path_suffix,
            "--config",
            training_config.path,
            "--runtime-contract",
            RUNTIME_CONTRACT_PATH,
            "--output-dir",
            trained_model.path,
            "--device",
            "cuda",
            "--initial-model-path",
            initial_model.path,
            "--initialization-mode",
            initialization_mode,
            "--observability-run-uri",
            observability_run_uri,
            "--batch-id",
            batch_id,
            "--candidate-model-version",
            candidate_model_version,
        ],
    )
