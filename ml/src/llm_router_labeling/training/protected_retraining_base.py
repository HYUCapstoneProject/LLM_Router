"""Build an immutable feedback base without consuming the feedback stream."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from ..shared.dataset_io import load_config, sha256_file, verify_file, write_json
from .feedback_retraining_preprocessing import FeedbackDataError

FROZEN_MANIFEST_SHA256 = (
    "f6a4cff1684bc2626a8901dbfda9f81e7d5d7e8753e5f48320cb1ada44c8cca4"
)
PARTITIONS = {
    "train": "train/training.parquet",
    "validation": "validation/validation.parquet",
    "gate": "sealed/gate.parquet",
    "retention_test": "sealed/retention_test.parquet",
    "test": "test/routerbench.parquet",
    "external_test": "test/xroutebench.parquet",
    "stream": "sealed/stream.parquet",
}
CONTRACT_COLUMNS = (
    "dataset_version",
    "label_rule_version",
    "preprocessing_version",
    "split_version",
)


def _verified_manifest(dataset_dir: Path, expected_sha256: str) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    if sha256_file(manifest_path) != expected_sha256:
        raise FeedbackDataError("Frozen source manifest SHA-256 mismatch")
    manifest = load_config(manifest_path)
    artifacts = manifest["artifacts"]
    if not set(PARTITIONS.values()).issubset(artifacts):
        raise FeedbackDataError("Frozen manifest is missing a required partition")
    for relative, record in sorted(artifacts.items()):
        path = (dataset_dir / relative).resolve()
        if not path.is_relative_to(dataset_dir) or not path.is_file():
            raise FeedbackDataError(f"Invalid frozen artifact path: {relative}")
        verify_file(path, record)
    if manifest["input_contract"] != "structure-preserving-v2":
        raise FeedbackDataError(
            "Protected base requires structure-preserving-v2 inputs"
        )
    if manifest["selected_thresholds"]["selection_data"] != "train_only_group_oof":
        raise FeedbackDataError(
            "Label thresholds must have been selected on train only"
        )
    if manifest["selected_thresholds"]["routerarena_used_for_selection"]:
        raise FeedbackDataError(
            "RouterArena must not have been used for threshold selection"
        )
    return manifest


def build_protected_retraining_base(
    dataset_dir: Path,
    output_dir: Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Preserve train/dev and retain sealed questions only as overlap blockers.

    The stream is read to validate group separation, then excluded from the base.
    No prediction or metric computation is performed on any sealed partition.
    """
    dataset_dir = dataset_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(dataset_dir):
        raise FeedbackDataError("Output must be outside the frozen source dataset")
    if output_dir.exists():
        raise FeedbackDataError(
            "Output already exists; immutable bases cannot be overwritten"
        )
    manifest = _verified_manifest(dataset_dir, expected_manifest_sha256)
    t1 = manifest["selected_thresholds"]["t1"]
    contract = {
        key: manifest[key] for key in CONTRACT_COLUMNS if key != "label_rule_version"
    }
    contract["label_rule_version"] = (
        f"{manifest['label_rule_version']}:{t1['candidate_id']}"
    )
    parts: list[pd.DataFrame] = []
    for partition, relative in PARTITIONS.items():
        frame = pd.read_parquet(dataset_dir / relative)
        required = {
            "sample_id",
            "model_input_text",
            "difficulty_label",
            "sample_weight",
            "duplicate_group_id",
            "split",
            "source_dataset",
            *CONTRACT_COLUMNS,
        }
        if required - set(frame.columns) or frame.empty:
            raise FeedbackDataError(f"Missing columns or empty partition: {partition}")
        if len(frame) != manifest["row_counts"][partition]:
            raise FeedbackDataError(f"Frozen row count mismatch: {partition}")
        if set(frame["split"].astype(str)) != {partition}:
            raise FeedbackDataError(f"Frozen partition mismatch: {partition}")
        for column, expected in contract.items():
            if set(frame[column].astype(str)) != {expected}:
                raise FeedbackDataError(
                    f"Frozen row contract mismatch: {partition}/{column}"
                )
        if set(frame["difficulty_label"].astype(str)) - {"easy", "medium", "hard"}:
            raise FeedbackDataError(f"Invalid difficulty label: {partition}")
        if (
            not frame["sample_weight"]
            .eq(float(manifest["effective_sample_weight"]))
            .all()
        ):
            raise FeedbackDataError(f"Frozen sample weights changed: {partition}")
        for column in ("sample_id", "duplicate_group_id", "model_input_text"):
            if frame[column].fillna("").astype(str).str.strip().eq("").any():
                raise FeedbackDataError(f"Empty {column}: {partition}")
        frame = frame.copy()
        frame["original_partition"] = partition
        frame["original_artifact"] = relative
        frame["protected_holdout"] = partition not in {"train", "validation", "stream"}
        parts.append(frame)

    all_rows = pd.concat(parts, ignore_index=True)
    if all_rows["sample_id"].duplicated().any():
        raise FeedbackDataError("Sample IDs overlap frozen partitions")
    group_partitions = all_rows.groupby("duplicate_group_id")[
        "original_partition"
    ].nunique()
    if group_partitions.gt(1).any():
        raise FeedbackDataError("Duplicate groups overlap frozen partitions")
    base = all_rows.loc[~all_rows["original_partition"].eq("stream")].copy()
    base.loc[base["protected_holdout"], "split"] = "test"
    base = base.sort_values(
        ["original_partition", "sample_id"], kind="stable"
    ).reset_index(drop=True)
    output_dir.mkdir(parents=True)
    training_path = output_dir / "training.parquet"
    base.to_parquet(training_path, index=False)
    output_manifest = {
        "schema_version": "protected-retraining-base-v1",
        "source_manifest_sha256": expected_manifest_sha256,
        "source_artifacts_verified": len(manifest["artifacts"]),
        "verified_file_count_including_manifest": len(manifest["artifacts"]) + 1,
        "dataset_contract": contract,
        "input_contract": manifest["input_contract"],
        "selected_thresholds": {
            "hard_threshold": t1["hard_threshold"],
            "easy_threshold": t1["easy_threshold"],
        },
        "row_count": len(base),
        "original_partition_row_counts": {
            str(key): int(value)
            for key, value in base["original_partition"]
            .value_counts()
            .sort_index()
            .items()
        },
        "split_row_counts": {
            str(key): int(value)
            for key, value in base["split"].value_counts().sort_index().items()
        },
        "excluded_stream_row_count": int(
            all_rows["original_partition"].eq("stream").sum()
        ),
        "protected_holdout_row_count": int(base["protected_holdout"].sum()),
        "holdout_policy": "test rows block feedback overlap; never train or calibrate on them",
        "stream_policy": "excluded from base; reserved as future feedback supply",
        "deployment_status": "blocked_pending_objective_selection",
        "deployment_blockers": [
            "Select R0/R1/R2 on frozen D_dev",
            "Define observable feedback targets and serving outputs for the selected objective",
        ],
        "artifacts": {
            "training.parquet": {
                "sha256": sha256_file(training_path),
                "size_bytes": training_path.stat().st_size,
            }
        },
    }
    write_json(output_dir / "manifest.json", output_manifest)
    return output_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("ml/artifacts/modernbert-initial-v1/dataset"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", default=FROZEN_MANIFEST_SHA256)
    args = parser.parse_args()
    result = build_protected_retraining_base(
        args.dataset_dir,
        args.output_dir,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    print(
        f"Built {result['row_count']} rows; excluded {result['excluded_stream_row_count']} stream rows; SHA-256 {result['artifacts']['training.parquet']['sha256']}"
    )


if __name__ == "__main__":
    main()
