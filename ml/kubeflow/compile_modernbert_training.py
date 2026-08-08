from __future__ import annotations

import argparse
from pathlib import Path

from kfp import compiler

from ml.kubeflow.components.modernbert_training import (
    modernbert_training_component,
)
from ml.kubeflow.pipelines.modernbert_training import modernbert_training_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile the ModernBERT KFP component and training pipeline"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/kubeflow/modernbert-training"),
    )
    return parser


def compile_definitions(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    component_path = output_dir / "modernbert-training-component.yaml"
    pipeline_path = output_dir / "modernbert-training-pipeline.yaml"
    compiler.Compiler().compile(
        modernbert_training_component,
        package_path=str(component_path),
    )
    compiler.Compiler().compile(
        modernbert_training_pipeline,
        package_path=str(pipeline_path),
    )
    return component_path, pipeline_path


def main() -> None:
    args = build_parser().parse_args()
    component_path, pipeline_path = compile_definitions(args.output_dir)
    print(component_path)
    print(pipeline_path)


if __name__ == "__main__":
    main()
