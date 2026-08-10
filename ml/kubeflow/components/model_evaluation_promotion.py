from kfp import dsl


DEFAULT_EVALUATION_IMAGE = "llm-router-modernbert-evaluation:cu124-v1"
DEFAULT_PROMOTION_IMAGE = "llm-router-model-promotion:py312-v1"


@dsl.container_component
def routerarena_evaluation_component(
    trained_model: dsl.Input[dsl.Model],
    evaluation_dataset: dsl.Input[dsl.Dataset],
    evaluation_config: dsl.Input[dsl.Artifact],
    evaluation_results: dsl.Output[dsl.Artifact],
):
    """Evaluate a candidate checkpoint on the canonical RouterArena dataset."""
    return dsl.ContainerSpec(
        image=DEFAULT_EVALUATION_IMAGE,
        command=["routerarena-evaluate"],
        args=[
            "--config",
            evaluation_config.path,
            "--evaluation-path",
            evaluation_dataset.path,
            "--model-path",
            trained_model.path,
            "--output-dir",
            evaluation_results.path,
            "--device",
            "cuda",
        ],
    )


@dsl.container_component
def model_package_component(
    trained_model: dsl.Input[dsl.Model],
    evaluation_results: dsl.Input[dsl.Artifact],
    package_metadata: dsl.Input[dsl.Artifact],
    packaged_model: dsl.Output[dsl.Model],
):
    """Assemble and checksum a candidate package without approving deployment."""
    return dsl.ContainerSpec(
        image=DEFAULT_PROMOTION_IMAGE,
        command=["modernbert-package"],
        args=[
            "assemble",
            "--checkpoint-dir",
            trained_model.path,
            "--evaluation-metrics",
            dsl.ConcatPlaceholder(items=[evaluation_results.path, "/metrics.json"]),
            "--metadata",
            package_metadata.path,
            "--output-dir",
            packaged_model.path,
        ],
    )


@dsl.container_component
def deploy_gate_component(
    packaged_model: dsl.Input[dsl.Model],
    candidate_evaluation: dsl.Input[dsl.Artifact],
    baseline_metrics: dsl.Input[dsl.Artifact],
    gate_rules: dsl.Input[dsl.Artifact],
    candidate_model_version: str,
    baseline_model_version: str,
    gate_result: dsl.Output[dsl.Artifact],
):
    """Fail the task when the candidate violates versioned promotion rules."""
    return dsl.ContainerSpec(
        image=DEFAULT_PROMOTION_IMAGE,
        command=["modernbert-deploy-gate"],
        args=[
            "--candidate-package",
            packaged_model.path,
            "--candidate-metrics",
            dsl.ConcatPlaceholder(items=[candidate_evaluation.path, "/metrics.json"]),
            "--candidate-model-version",
            candidate_model_version,
            "--baseline-metrics",
            baseline_metrics.path,
            "--baseline-model-version",
            baseline_model_version,
            "--rules",
            gate_rules.path,
            "--output",
            dsl.ConcatPlaceholder(items=[gate_result.path, "/gate_result.json"]),
        ],
    )

