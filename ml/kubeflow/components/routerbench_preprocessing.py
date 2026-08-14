from kfp import dsl

from ml.kubeflow.components.feedback_retraining import (
    DEFAULT_DATA_PROCESSING_IMAGE,
)


@dsl.container_component
def routerbench_v2_preprocessing_component(
    zero_shot_source: dsl.Input[dsl.Dataset],
    five_shot_source: dsl.Input[dsl.Dataset],
    labeling_config: dsl.Input[dsl.Artifact],
    training_dataset: dsl.Output[dsl.Dataset],
):
    """Verify raw RouterBench sources and build the v2 training bundle on CPU."""
    return dsl.ContainerSpec(
        image=DEFAULT_DATA_PROCESSING_IMAGE,
        command=["routerbench-labeling"],
        args=[
            "--config",
            labeling_config.path,
            "build-full",
            "--zero-shot-path",
            zero_shot_source.path,
            "--five-shot-path",
            five_shot_source.path,
            "--output-dir",
            training_dataset.path,
            "--seed",
            "42",
            "--trust-pickle",
        ],
    )
