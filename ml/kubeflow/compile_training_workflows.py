from __future__ import annotations

import argparse
from pathlib import Path

from kfp import compiler

from ml.kubeflow.components.feedback_retraining import (
    feedback_merge_component,
    feedback_prepare_component,
)
from ml.kubeflow.components.modernbert_training import modernbert_training_component
from ml.kubeflow.pipelines.modernbert_retraining import (
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
        "feedback_prepare_component": (
            feedback_prepare_component,
            output_dir / "feedback-prepare-component.yaml",
        ),
        "feedback_merge_component": (
            feedback_merge_component,
            output_dir / "feedback-merge-component.yaml",
        ),
        "modernbert_training_component": (
            modernbert_training_component,
            output_dir / "modernbert-training-component.yaml",
        ),
        "initial_training_pipeline": (
            modernbert_training_pipeline,
            output_dir / "initial-training-pipeline.yaml",
        ),
        "feedback_retraining_pipeline": (
            modernbert_retraining_pipeline,
            output_dir / "feedback-retraining-pipeline.yaml",
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
