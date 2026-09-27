from kfp import dsl

from ml.kubeflow.components.feedback_retraining import (
    DEFAULT_DATA_PROCESSING_IMAGE,
    feedback_merge_component,
    feedback_prepare_component,
)
from ml.kubeflow.components.model_evaluation_promotion import (
    DEFAULT_EVALUATION_IMAGE,
    DEFAULT_PROMOTION_IMAGE,
)
from ml.kubeflow.components.vertex_training import (
    vertex_modernbert_continual_training_job_component,
)
from ml.kubeflow.pipelines.model_qualification import add_model_qualification_tasks
from ml.kubeflow.task_config import (
    configure_cpu_task,
)


def build_retraining_pipeline(*, direct: bool = False):
    @dsl.pipeline(
        name=(
            "modernbert-router-feedback-direct-retraining"
            if direct
            else "modernbert-router-feedback-retraining"
        ),
        description=(
            "Prepare feedback, retrain ModernBERT, and enforce the shared qualification gate."
        ),
    )
    def modernbert_retraining_pipeline(
        base_training_dataset_uri: str = (
            "gs://replace-me/datasets/routerbench/"
            "withmartian-routerbench-2024-03-27__minimum-sufficient-tier-v2__"
            "routerbench-training-preprocess-v2-balanced/training.parquet"
        ),
        feedback_manifest_uri: str = (
            "gs://replace-me/datasets/feedback/batch/manifest.json"
        ),
        feedback_events_uri: str = (
            "gs://replace-me/datasets/feedback/batch/events.parquet"
        ),
        feedback_preprocessing_config_uri: str = (
            "gs://replace-me/config/feedback_retraining_preprocessing.json"
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
        initial_model_uri: str = "gs://replace-me/models/router-modernbert/base/approved",
        initialization_mode: str = "approved_anchor",
        logical_model_version: str = "1.1",
        training_mode: str = "minor_fine_tune",
        retention_anchor_model_version: str = "replace-anchor-version",
        batch_id: str = "replace-batch-id",
        observability_run_uri: str = "gs://replace-me/retraining-runs/batch/observability-v1",
        version_policy_mode: str = "successfulHttpRequest",
        retraining_control_root_uri: str = "",
        model_publish_root_uri: str = "gs://replace-me/models/router-modernbert",
        data_processing_image: str = DEFAULT_DATA_PROCESSING_IMAGE,
        training_service_account: str = "",
        training_network: str = "",
        training_base_output_directory: str = "",
        evaluation_image: str = DEFAULT_EVALUATION_IMAGE,
        promotion_image: str = DEFAULT_PROMOTION_IMAGE,
        promotion_mode: str = "direct" if direct else "quality_gate",
    ) -> dsl.Artifact:
        """Run feedback preparation, GPU retraining, evaluation, package, and gate."""
        base_dataset_importer = dsl.importer(
            artifact_uri=base_training_dataset_uri,
            artifact_class=dsl.Dataset,
            reimport=False,
        ).set_display_name("Import base training dataset")
        feedback_manifest_importer = dsl.importer(
            artifact_uri=feedback_manifest_uri,
            artifact_class=dsl.Artifact,
            reimport=False,
        ).set_display_name("Import feedback manifest")
        feedback_events_importer = dsl.importer(
            artifact_uri=feedback_events_uri,
            artifact_class=dsl.Dataset,
            reimport=False,
        ).set_display_name("Import feedback events")
        feedback_config_importer = dsl.importer(
            artifact_uri=feedback_preprocessing_config_uri,
            artifact_class=dsl.Artifact,
            reimport=False,
        ).set_display_name("Import feedback preprocessing config")
        training_config_importer = dsl.importer(
            artifact_uri=training_config_uri,
            artifact_class=dsl.Artifact,
            reimport=False,
        ).set_display_name("Import training config")
        initial_model_importer = dsl.importer(
            artifact_uri=initial_model_uri,
            artifact_class=dsl.Model,
            reimport=False,
        ).set_display_name("Import approved anchor model")

        prepare_task = feedback_prepare_component(
            feedback_manifest=feedback_manifest_importer.output,
            feedback_events=feedback_events_importer.output,
            preprocessing_config=feedback_config_importer.output,
            observability_run_uri=observability_run_uri,
            batch_id=batch_id,
            version_policy_mode=version_policy_mode,
            retraining_control_root_uri=retraining_control_root_uri,
            candidate_model_version=candidate_model_version,
        )
        if direct:
            from ml.kubeflow.components.direct_promotion import (
                direct_promotion_preflight_component,
            )

            preflight = direct_promotion_preflight_component(
                promotion_mode=promotion_mode, version_policy_mode=version_policy_mode
            )
            configure_cpu_task(
                preflight,
                image=promotion_image,
                cpu_request="1",
                cpu_limit="1",
                memory_request="1Gi",
                memory_limit="1Gi",
            )
            preflight.set_caching_options(enable_caching=False)
            preflight.set_retry(num_retries=0)
            prepare_task.after(preflight)
        prepare_task.set_display_name("Validate and normalize feedback")
        configure_cpu_task(
            prepare_task,
            image=data_processing_image,
            cpu_request="1",
            cpu_limit="2",
            memory_request="2Gi",
            memory_limit="4Gi",
        )

        merge_task = feedback_merge_component(
            base_training_dataset=base_dataset_importer.output,
            prepared_feedback=prepare_task.outputs["prepared_feedback"],
            preprocessing_config=feedback_config_importer.output,
            training_data_strategy="compact_rehearsal",
            observability_run_uri=observability_run_uri,
            batch_id=batch_id,
            candidate_model_version=candidate_model_version,
        )
        merge_task.set_display_name("Merge feedback with base training data")
        configure_cpu_task(
            merge_task,
            image=data_processing_image,
            cpu_request="2",
            cpu_limit="4",
            memory_request="4Gi",
            memory_limit="8Gi",
        )

        training_task = vertex_modernbert_continual_training_job_component(
            training_dataset=merge_task.outputs["merged_training_dataset"],
            training_config=training_config_importer.output,
            initial_model=initial_model_importer.output,
            initialization_mode=initialization_mode,
            observability_run_uri=observability_run_uri,
            batch_id=batch_id,
            candidate_model_version=candidate_model_version,
            training_path_suffix="/training.parquet",
            service_account=training_service_account,
            network=training_network,
            base_output_directory=training_base_output_directory,
        )
        training_task.set_display_name("Vertex T4: retrain ModernBERT router")
        training_task.set_caching_options(enable_caching=True)
        if direct:
            from ml.kubeflow.components.direct_promotion import (
                direct_promotion_component,
            )

            metadata = dsl.importer(
                artifact_uri=package_metadata_uri,
                artifact_class=dsl.Artifact,
                reimport=False,
            )
            direct_task = direct_promotion_component(
                trained_model=training_task.outputs["trained_model"],
                package_metadata=metadata.output,
                candidate_model_version=candidate_model_version,
                baseline_model_version=baseline_model_version,
                model_publish_root_uri=model_publish_root_uri,
                observability_run_uri=observability_run_uri,
                batch_id=batch_id,
                logical_model_version=logical_model_version,
                training_mode=training_mode,
                retention_anchor_model_version=retention_anchor_model_version,
                promotion_mode=promotion_mode,
            )
            direct_task.set_display_name(
                "Publish R0 directly with integrity and CURRENT checks"
            )
            configure_cpu_task(
                direct_task,
                image=promotion_image,
                cpu_request="1",
                cpu_limit="2",
                memory_request="1Gi",
                memory_limit="2Gi",
            )
            direct_task.set_caching_options(enable_caching=False)
            return direct_task.outputs["approval_result"]
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
            logical_model_version=logical_model_version,
            training_mode=training_mode,
            retention_anchor_model_version=retention_anchor_model_version,
            evaluation_image=evaluation_image,
            promotion_image=promotion_image,
            observability_run_uri=observability_run_uri,
            batch_id=batch_id,
            version_policy_mode=version_policy_mode,
            retraining_control_root_uri=retraining_control_root_uri,
        )
        return approval_task.outputs["approval_result"]

    return modernbert_retraining_pipeline


modernbert_retraining_pipeline = build_retraining_pipeline()
modernbert_direct_retraining_pipeline = build_retraining_pipeline(direct=True)
