from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..shared.dataset_io import load_config, sha256_file, write_json
from .feedback_retraining_preprocessing import (
    load_feedback_events,
    merge_feedback_with_base_training,
    prepare_feedback_training_dataset,
    validate_feedback_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and prepare escalation feedback for router retraining"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/config/feedback_retraining_preprocessing.json"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Validate a feedback export and create canonical training rows"
    )
    prepare.add_argument("--manifest-path", type=Path, required=True)
    prepare.add_argument("--events-path", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    merge = subparsers.add_parser(
        "merge", help="Merge canonical feedback rows into a base training artifact"
    )
    merge.add_argument("--base-training-path", type=Path, required=True)
    merge.add_argument("--feedback-training-path", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)

    build = subparsers.add_parser(
        "build",
        help="Prepare feedback and merge it into a base artifact in one local run",
    )
    build.add_argument("--manifest-path", type=Path, required=True)
    build.add_argument("--events-path", type=Path, required=True)
    build.add_argument("--base-training-path", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    return parser


def _artifact_manifest(
    *,
    artifact_type: str,
    version: str,
    files: Mapping[str, Path],
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "version": version,
        "files": {
            name: {"filename": path.name, "sha256": sha256_file(path)}
            for name, path in files.items()
        },
        "lineage": dict(lineage),
    }


def command_prepare(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    manifest = load_config(args.manifest_path)
    validate_feedback_manifest(
        manifest,
        args.events_path,
        expected_schema_version=str(config["schema_version"]),
    )
    events = load_feedback_events(args.events_path, str(manifest["data_format"]))
    result = prepare_feedback_training_dataset(events, manifest, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_path = args.output_dir / "feedback_training.parquet"
    metadata_path = args.output_dir / "feedback_metadata.parquet"
    metrics_path = args.output_dir / "metrics.json"
    result.training.to_parquet(training_path, index=False)
    result.metadata.to_parquet(metadata_path, index=False)
    write_json(metrics_path, result.metrics)
    write_json(
        args.output_dir / "manifest.json",
        _artifact_manifest(
            artifact_type="canonical-feedback-training",
            version=str(config["adapter_version"]),
            files={
                "training": training_path,
                "metadata": metadata_path,
                "metrics": metrics_path,
            },
            lineage={
                "source_batch_id": str(manifest["batch_id"]),
                "source_schema_version": str(manifest["schema_version"]),
                "source_sha256": str(manifest["sha256"]),
            },
        ),
    )
    return result.metrics


def command_merge(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    base = pd.read_parquet(args.base_training_path)
    feedback = pd.read_parquet(args.feedback_training_path)
    result = merge_feedback_with_base_training(base, feedback, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_path = args.output_dir / "training.parquet"
    metadata_path = args.output_dir / "merge_metadata.parquet"
    metrics_path = args.output_dir / "metrics.json"
    result.training.to_parquet(training_path, index=False)
    result.metadata.to_parquet(metadata_path, index=False)
    write_json(metrics_path, result.metrics)
    write_json(
        args.output_dir / "manifest.json",
        _artifact_manifest(
            artifact_type="merged-router-training",
            version=str(config["merge_version"]),
            files={
                "training": training_path,
                "metadata": metadata_path,
                "metrics": metrics_path,
            },
            lineage={
                "base_training_sha256": sha256_file(args.base_training_path),
                "feedback_training_sha256": sha256_file(
                    args.feedback_training_path
                ),
            },
        ),
    )
    return result.metrics


def command_build(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    feedback_output = args.output_dir / "feedback"
    merged_output = args.output_dir / "merged"
    feedback_metrics = command_prepare(
        argparse.Namespace(
            manifest_path=args.manifest_path,
            events_path=args.events_path,
            output_dir=feedback_output,
        ),
        config,
    )
    merge_metrics = command_merge(
        argparse.Namespace(
            base_training_path=args.base_training_path,
            feedback_training_path=feedback_output / "feedback_training.parquet",
            output_dir=merged_output,
        ),
        config,
    )
    return {"feedback": feedback_metrics, "merge": merge_metrics}


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        result = command_prepare(args, config)
    elif args.command == "merge":
        result = command_merge(args, config)
    else:
        result = command_build(args, config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
