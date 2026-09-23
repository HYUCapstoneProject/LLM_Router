from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..shared.dataset_io import load_config, write_json
from ..shared.observability import write_observability_json
from .routerarena_routing_evaluator import (
    HuggingFaceRouterPredictor,
    run_routerarena_routing_evaluation,
)
from .routerarena_visualization import write_routerarena_distribution_charts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a RouterBench-trained router on RouterArena"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/config/routerarena_evaluation.json"),
    )
    parser.add_argument(
        "--evaluation-path",
        type=Path,
        default=Path("ml/artifacts/routerarena-evaluation-full/evaluation.parquet"),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        help="Defaults to metadata.parquet beside --evaluation-path when present",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local Hugging Face checkpoint directory or model identifier",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/routerarena-routing-evaluation"),
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--observability-run-uri")
    parser.add_argument("--batch-id")
    parser.add_argument("--candidate-model-version")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    inference = config["inference"]
    metadata_path = args.metadata_path
    if metadata_path is None:
        candidate = args.evaluation_path.parent / "metadata.parquet"
        metadata_path = candidate if candidate.exists() else None
    evaluation = pd.read_parquet(args.evaluation_path)
    metadata = pd.read_parquet(metadata_path) if metadata_path else None
    predictor = HuggingFaceRouterPredictor(
        args.model_path,
        label_order=config["label_order"],
        batch_size=args.batch_size or int(inference["batch_size"]),
        max_length=args.max_length or int(inference["max_length"]),
        device=args.device or str(inference["device"]),
    )
    result = run_routerarena_routing_evaluation(
        evaluation,
        predictor,
        label_order=config["label_order"],
        route_mapping=config["route_mapping"],
        evaluation_version=config["evaluation_version"],
        label_semantics=config["label_semantics"],
        metadata=metadata,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(args.output_dir / "predictions.parquet", index=False)
    write_json(args.output_dir / "metrics.json", result.metrics)
    write_json(args.output_dir / "model_info.json", predictor.info())
    write_routerarena_distribution_charts(result.metrics, args.output_dir)
    if args.observability_run_uri:
        if not args.batch_id or not args.candidate_model_version:
            raise SystemExit(
                "observability requires --batch-id and --candidate-model-version"
            )
        write_observability_json(
            f"{args.observability_run_uri.rstrip('/')}/metrics/evaluation.json",
            {
                "batch_id": args.batch_id,
                "candidate_model_version": args.candidate_model_version,
                **result.metrics,
            },
        )
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2, sort_keys=True))
