from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..shared.dataset_io import download_source, load_config, write_json
from .routerbench_labeling import (
    ModelTiers,
    RouterBenchError,
    build_metrics,
    calibrate_model_tiers,
    canonicalize_sources,
    filter_eval_scope,
    label_questions,
    load_trusted_pickle,
    stratified_smoke_sample,
    validate_labeled_output,
)
from .routerbench_preprocessing import prepare_training_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RouterBench data labeling")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/config/routerbench_labeling.json"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download and verify source files")
    download.add_argument("--data-dir", type=Path, default=Path("ml/data/raw"))

    smoke = subparsers.add_parser("smoke", help="Label a reproducible smoke sample")
    smoke.add_argument("--data-dir", type=Path, default=Path("ml/data/raw"))
    smoke.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/routerbench-smoke-100"),
    )
    smoke.add_argument("--sample-size", type=int)
    smoke.add_argument("--seed", type=int)
    smoke.add_argument("--trust-pickle", action="store_true")

    label = subparsers.add_parser("label", help="Label the full scoped dataset")
    label.add_argument("--data-dir", type=Path, default=Path("ml/data/raw"))
    label.add_argument("--output-dir", type=Path)
    label.add_argument("--trust-pickle", action="store_true")

    preprocess = subparsers.add_parser(
        "preprocess", help="Prepare a labeled Parquet for model training"
    )
    preprocess.add_argument(
        "--input-path",
        type=Path,
    )
    preprocess.add_argument(
        "--output-dir",
        type=Path,
    )
    preprocess.add_argument(
        "--profile",
        choices=("smoke", "full"),
        default="smoke",
        help="Select versioned default input/output paths when paths are omitted",
    )
    preprocess.add_argument("--seed", type=int)

    build_full = subparsers.add_parser(
        "build-full",
        help=(
            "Build a versioned full labeling and training bundle from two "
            "verified RouterBench source artifacts"
        ),
    )
    build_full.add_argument("--zero-shot-path", type=Path, required=True)
    build_full.add_argument("--five-shot-path", type=Path, required=True)
    build_full.add_argument("--output-dir", type=Path, required=True)
    build_full.add_argument("--seed", type=int)
    build_full.add_argument("--trust-pickle", action="store_true")
    return parser


def command_download(config: dict[str, Any], data_dir: Path) -> dict[str, str]:
    downloaded = {}
    for source_name, source_config in config["sources"].items():
        downloaded[source_name] = str(download_source(source_config, data_dir))
    return downloaded


def full_label_output_dir(config: dict[str, Any]) -> Path:
    version = "__".join(
        (
            str(config["dataset_version"]),
            str(config["tier_config_version"]),
            str(config["label_rule_version"]),
        )
    )
    return Path("ml/artifacts/routerbench-labeled") / version


def full_preprocess_output_dir(config: dict[str, Any]) -> Path:
    version = "__".join(
        (
            str(config["dataset_version"]),
            str(config["label_rule_version"]),
            str(config["preprocessing_version"]),
        )
    )
    return Path("ml/artifacts/routerbench-training") / version


def resolve_preprocess_paths(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[Path, Path]:
    if args.profile == "full":
        default_input = full_label_output_dir(config) / "labeled.parquet"
        default_output = full_preprocess_output_dir(config)
    else:
        default_input = Path("ml/artifacts/routerbench-smoke-100/labeled.parquet")
        default_output = Path("ml/artifacts/routerbench-preprocessed-smoke-100")
    return args.input_path or default_input, args.output_dir or default_output


def validate_labeled_versions(
    labeled: pd.DataFrame, config: dict[str, Any]
) -> None:
    expected = {
        "dataset_version": str(config["dataset_version"]),
        "tier_config_version": str(config["tier_config_version"]),
        "label_rule_version": str(config["label_rule_version"]),
    }
    for column, expected_value in expected.items():
        if column not in labeled:
            raise RouterBenchError(
                f"Labeled input is missing version column: {column}"
            )
        actual_values = sorted(labeled[column].dropna().astype(str).unique().tolist())
        if actual_values != [expected_value]:
            raise RouterBenchError(
                f"Labeled input {column} must be {expected_value!r}, "
                f"got {actual_values}"
            )


def _load_canonical_dataset(
    data_dir: Path, trust_pickle: bool, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    source_frames = {}
    for source_name, source_config in config["sources"].items():
        path = data_dir / source_config["filename"]
        source_frames[source_name] = load_trusted_pickle(
            path, source_config, trust_pickle
        )

    canonical_all, models = canonicalize_sources(
        source_frames["0shot"],
        source_frames["5shot"],
        config["canonicalization"],
    )
    canonical, scope_metrics = filter_eval_scope(
        canonical_all, config["canonicalization"]
    )
    return canonical_all, canonical, models, scope_metrics


def _load_canonical_source_paths(
    zero_shot_path: Path,
    five_shot_path: Path,
    trust_pickle: bool,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    source_frames = {
        "0shot": load_trusted_pickle(
            zero_shot_path, config["sources"]["0shot"], trust_pickle
        ),
        "5shot": load_trusted_pickle(
            five_shot_path, config["sources"]["5shot"], trust_pickle
        ),
    }
    canonical_all, models = canonicalize_sources(
        source_frames["0shot"],
        source_frames["5shot"],
        config["canonicalization"],
    )
    canonical, scope_metrics = filter_eval_scope(
        canonical_all, config["canonicalization"]
    )
    return canonical_all, canonical, models, scope_metrics


def _write_labeling_artifacts(
    labeled: pd.DataFrame,
    canonical_all: pd.DataFrame,
    canonical: pd.DataFrame,
    tiers: ModelTiers,
    scope_metrics: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
    run_metrics: dict[str, Any],
) -> dict[str, Any]:
    for column, value in (
        ("dataset_version", config["dataset_version"]),
        ("tier_config_version", config["tier_config_version"]),
        ("label_rule_version", config["label_rule_version"]),
    ):
        labeled[column] = value

    output_dir.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(output_dir / "labeled.parquet", index=False)
    tier_payload = {
        "tier_config_version": config["tier_config_version"],
        "dataset_checksums": {
            name: source["sha256"] for name, source in config["sources"].items()
        },
        "ranking_metric": "eval_macro_mean_percentile_rank",
        **tiers.to_dict(),
    }
    write_json(output_dir / "model_tiers.json", tier_payload)
    metrics = build_metrics(labeled, canonical, tiers, config)
    metrics["unscoped_canonical_row_count"] = len(canonical_all)
    metrics.update(scope_metrics)
    metrics.update(run_metrics)
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def command_smoke(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    canonical_all, canonical, models, scope_metrics = _load_canonical_dataset(
        args.data_dir, args.trust_pickle, config
    )
    tiers = calibrate_model_tiers(canonical, models)
    sample_size = args.sample_size or int(config["smoke"]["sample_size"])
    seed = args.seed if args.seed is not None else int(config["smoke"]["seed"])
    smoke_sample = stratified_smoke_sample(canonical, sample_size, seed)
    labeled = label_questions(
        smoke_sample,
        tiers,
        config["acceptance"],
        config["labeling"],
    )
    validate_labeled_output(labeled, sample_size)
    return _write_labeling_artifacts(
        labeled,
        canonical_all,
        canonical,
        tiers,
        scope_metrics,
        config,
        args.output_dir,
        {
            "run_mode": "smoke",
            "sample_size": sample_size,
            "seed": seed,
            "sampling_strategy": (
                "one_per_eval_then_minimum_10_percent_per_domain_then_seeded_fill"
            ),
        },
    )


def command_label(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    canonical_all, canonical, models, scope_metrics = _load_canonical_dataset(
        args.data_dir, args.trust_pickle, config
    )
    tiers = calibrate_model_tiers(canonical, models)
    labeled = label_questions(
        canonical.sort_values("sample_id", kind="stable").reset_index(drop=True),
        tiers,
        config["acceptance"],
        config["labeling"],
    )
    validate_labeled_output(labeled, len(canonical))
    output_dir = args.output_dir or full_label_output_dir(config)
    return _write_labeling_artifacts(
        labeled,
        canonical_all,
        canonical,
        tiers,
        scope_metrics,
        config,
        output_dir,
        {"run_mode": "full", "sampling_strategy": "all_scoped_rows"},
    )


def command_preprocess(
    args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, Any]:
    input_path, output_dir = resolve_preprocess_paths(args, config)
    labeled = pd.read_parquet(input_path)
    validate_labeled_versions(labeled, config)
    prepared = prepare_training_dataset(
        labeled,
        config["preprocessing"],
        preprocessing_version=config["preprocessing_version"],
        seed=args.seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared.training.to_parquet(output_dir / "training.parquet", index=False)
    prepared.metadata.to_parquet(output_dir / "metadata.parquet", index=False)
    metrics = {"profile": args.profile, **prepared.metrics}
    write_json(output_dir / "metrics.json", metrics)
    return metrics


def command_build_full(
    args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, Any]:
    canonical_all, canonical, models, scope_metrics = _load_canonical_source_paths(
        args.zero_shot_path,
        args.five_shot_path,
        args.trust_pickle,
        config,
    )
    tiers = calibrate_model_tiers(canonical, models)
    labeled = label_questions(
        canonical.sort_values("sample_id", kind="stable").reset_index(drop=True),
        tiers,
        config["acceptance"],
        config["labeling"],
    )
    validate_labeled_output(labeled, len(canonical))
    labeling_metrics = _write_labeling_artifacts(
        labeled,
        canonical_all,
        canonical,
        tiers,
        scope_metrics,
        config,
        args.output_dir / "labeling",
        {"run_mode": "full", "sampling_strategy": "all_scoped_rows"},
    )
    prepared = prepare_training_dataset(
        labeled,
        config["preprocessing"],
        preprocessing_version=config["preprocessing_version"],
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared.training.to_parquet(args.output_dir / "training.parquet", index=False)
    prepared.metadata.to_parquet(args.output_dir / "metadata.parquet", index=False)
    write_json(args.output_dir / "metrics.json", prepared.metrics)
    result = {
        "dataset_version": config["dataset_version"],
        "label_rule_version": config["label_rule_version"],
        "preprocessing_version": config["preprocessing_version"],
        "labeling": labeling_metrics,
        "preprocessing": prepared.metrics,
    }
    write_json(args.output_dir / "build_metrics.json", result)
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "download":
        result = command_download(config, args.data_dir)
    elif args.command == "smoke":
        result = command_smoke(args, config)
    elif args.command == "label":
        result = command_label(args, config)
    elif args.command == "preprocess":
        result = command_preprocess(args, config)
    else:
        result = command_build_full(args, config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
