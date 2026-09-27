from __future__ import annotations

import argparse
from pathlib import Path

from kfp import compiler

from ml.kubeflow.components.feedback_retraining import (
    feedback_merge_component,
    feedback_prepare_component,
)
from ml.kubeflow.components.model_evaluation_promotion import (
    deploy_gate_component,
    model_auto_approve_component,
    model_package_component,
    model_publish_component,
    routerarena_evaluation_component,
)
from ml.kubeflow.components.routerbench_preprocessing import (
    routerbench_v2_preprocessing_component,
)
from ml.kubeflow.components.vertex_training import (
    vertex_modernbert_continual_training_job_component,
    vertex_modernbert_training_job_component,
)
from ml.kubeflow.pipelines.model_qualification import model_qualification_pipeline
from ml.kubeflow.pipelines.modernbert_retraining import (
    modernbert_direct_retraining_pipeline,
    modernbert_retraining_pipeline,
)
from ml.kubeflow.pipelines.modernbert_training import modernbert_training_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile CPU/GPU components and initial/retraining KFP workflows"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/kubeflow/training-workflows"),
    )
    return parser


def compile_workflows(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    definitions = {
        "routerbench_v2_preprocessing_component": (
            routerbench_v2_preprocessing_component,
            output_dir / "routerbench-v2-preprocessing-component.yaml",
        ),
        "feedback_prepare_component": (
            feedback_prepare_component,
            output_dir / "feedback-prepare-component.yaml",
        ),
        "feedback_merge_component": (
            feedback_merge_component,
            output_dir / "feedback-merge-component.yaml",
        ),
        "vertex_modernbert_training_job_component": (
            vertex_modernbert_training_job_component,
            output_dir / "vertex-modernbert-training-job-component.yaml",
        ),
        "vertex_modernbert_continual_training_job_component": (
            vertex_modernbert_continual_training_job_component,
            output_dir / "vertex-modernbert-continual-training-job-component.yaml",
        ),
        "routerarena_evaluation_component": (
            routerarena_evaluation_component,
            output_dir / "routerarena-evaluation-component.yaml",
        ),
        "model_package_component": (
            model_package_component,
            output_dir / "model-package-component.yaml",
        ),
        "deploy_gate_component": (
            deploy_gate_component,
            output_dir / "deploy-gate-component.yaml",
        ),
        "model_publish_component": (
            model_publish_component,
            output_dir / "model-publish-component.yaml",
        ),
        "model_auto_approve_component": (
            model_auto_approve_component,
            output_dir / "model-auto-approve-component.yaml",
        ),
        "initial_training_pipeline": (
            modernbert_training_pipeline,
            output_dir / "initial-training-pipeline.yaml",
        ),
        "feedback_retraining_pipeline": (
            modernbert_retraining_pipeline,
            output_dir / "feedback-retraining-pipeline.yaml",
        ),
        "feedback_direct_retraining_pipeline": (
            modernbert_direct_retraining_pipeline,
            output_dir / "feedback-direct-retraining-pipeline.yaml",
        ),
        "model_qualification_pipeline": (
            model_qualification_pipeline,
            output_dir / "model-qualification-pipeline.yaml",
        ),
    }
    paths: dict[str, Path] = {}
    for name, (definition, path) in definitions.items():
        compiler.Compiler().compile(definition, package_path=str(path))
        paths[name] = path
    return paths


def main() -> None:
    args = build_parser().parse_args()
    for path in compile_workflows(args.output_dir).values():
        print(path)


if __name__ == "__main__":
    main()
