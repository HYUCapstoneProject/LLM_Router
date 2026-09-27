from kfp import dsl

from ml.kubeflow.components.model_evaluation_promotion import DEFAULT_PROMOTION_IMAGE


@dsl.container_component
def direct_promotion_component(
    trained_model: dsl.Input[dsl.Model],
    package_metadata: dsl.Input[dsl.Artifact],
    candidate_model_version: str,
    baseline_model_version: str,
    model_publish_root_uri: str,
    observability_run_uri: str,
    batch_id: str,
    logical_model_version: str,
    training_mode: str,
    retention_anchor_model_version: str,
    promotion_mode: str,
    approval_result: dsl.Output[dsl.Artifact],
):
    return dsl.ContainerSpec(
        image=DEFAULT_PROMOTION_IMAGE,
        command=["python3", "-m", "llm_router_labeling.deployment.direct_promotion"],
        args=[
            "--checkpoint-dir",
            trained_model.path,
            "--metadata",
            package_metadata.path,
            "--output-dir",
            approval_result.path,
            "--model-version",
            candidate_model_version,
            "--baseline-model-version",
            baseline_model_version,
            "--root-uri",
            model_publish_root_uri,
            "--batch-id",
            batch_id,
            "--pipeline-run-id",
            dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER,
            "--observability-run-uri",
            observability_run_uri,
            "--logical-model-version",
            logical_model_version,
            "--training-mode",
            training_mode,
            "--retention-anchor-model-version",
            retention_anchor_model_version,
            "--promotion-mode",
            promotion_mode,
        ],
    )


@dsl.container_component
def direct_promotion_preflight_component(promotion_mode: str, version_policy_mode: str):
    return dsl.ContainerSpec(
        image=DEFAULT_PROMOTION_IMAGE,
        command=["python3", "-c"],
        args=[
            "import sys; assert sys.argv[1:] == ['direct', 'successfulHttpRequest'], 'Direct promotion requires direct/successfulHttpRequest'",
            promotion_mode,
            version_policy_mode,
        ],
    )
