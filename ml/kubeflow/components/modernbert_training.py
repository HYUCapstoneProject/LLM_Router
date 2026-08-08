from kfp import dsl


DEFAULT_TRAINING_IMAGE = "llm-router-modernbert-training:cu124-v1"
RUNTIME_CONTRACT_PATH = "/opt/llm-router/config/modernbert_runtime_cu124_v1.json"


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
        command=["modernbert-train"],
        args=[
            "--config",
            training_config.path,
            "--training-path",
            dsl.ConcatPlaceholder(
                items=[training_dataset.path, training_path_suffix]
            ),
            "--runtime-contract",
            RUNTIME_CONTRACT_PATH,
            "--output-dir",
            trained_model.path,
            "--device",
            "cuda",
        ],
    )
