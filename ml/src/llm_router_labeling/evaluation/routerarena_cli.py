from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..shared.dataset_io import download_source, load_config, write_json
from .routerarena_preprocessing import (
    load_routerarena_source,
    prepare_routerarena_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RouterArena evaluation preprocessing")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/config/routerarena_preprocessing.json"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download RouterArena splits")
    download.add_argument(
        "--data-dir", type=Path, default=Path("ml/data/raw/routerarena")
    )
    download.add_argument(
        "--split", choices=("all", "full", "sub_10", "robustness"), default="all"
    )

    preprocess = subparsers.add_parser(
        "preprocess", help="Build an evaluation-only RouterArena artifact"
    )
    preprocess.add_argument(
        "--data-dir", type=Path, default=Path("ml/data/raw/routerarena")
    )
    preprocess.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/routerarena-evaluation-smoke-100"),
    )
    preprocess.add_argument(
        "--split", choices=("full", "sub_10", "robustness"), default="sub_10"
    )
    preprocess.add_argument("--sample-size", type=int)
    preprocess.add_argument("--seed", type=int)
    preprocess.add_argument(
        "--training-path",
        type=Path,
        help="Optional RouterBench training Parquet for exact overlap exclusion",
    )
    return parser


def command_download(
    args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, str]:
    source_names = config["sources"] if args.split == "all" else (args.split,)
    return {
        source_name: str(download_source(config["sources"][source_name], args.data_dir))
        for source_name in source_names
    }


def command_preprocess(
    args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, Any]:
    source_config = config["sources"][args.split]
    dataframe = load_routerarena_source(
        args.data_dir / source_config["filename"], source_config
    )
    reference_full = None
    if args.split == "robustness":
        full_config = config["sources"]["full"]
        reference_full = load_routerarena_source(
            args.data_dir / full_config["filename"], full_config
        )
    training = pd.read_parquet(args.training_path) if args.training_path else None
    seed = args.seed if args.seed is not None else int(config["smoke"]["seed"])
    prepared = prepare_routerarena_evaluation(
        dataframe,
        config["preprocessing"],
        source_split=args.split,
        dataset_version=config["dataset_version"],
        preprocessing_version=config["preprocessing_version"],
        reference_full=reference_full,
        training=training,
        sample_size=args.sample_size,
        seed=seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepared.evaluation.to_parquet(args.output_dir / "evaluation.parquet", index=False)
    prepared.metadata.to_parquet(args.output_dir / "metadata.parquet", index=False)
    write_json(args.output_dir / "metrics.json", prepared.metrics)
    return prepared.metrics


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "download":
        result = command_download(args, config)
    else:
        result = command_preprocess(args, config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
