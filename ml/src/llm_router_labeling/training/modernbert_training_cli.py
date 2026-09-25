from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..shared.observability import write_observability_json
from .modernbert_runtime import load_runtime_contract
from .modernbert_training import load_training_config, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the shared local/Kubeflow ModernBERT router"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/config/modernbert_smoke.json"),
    )
    parser.add_argument(
        "--training-path",
        type=Path,
        default=Path(
            "ml/artifacts/routerbench-preprocessed-smoke-100/training.parquet"
        ),
    )
    parser.add_argument(
        "--runtime-contract",
        type=Path,
        default=Path(
            os.environ.get(
                "MODERNBERT_RUNTIME_CONTRACT",
                "ml/config/modernbert_runtime_cu124_v1.json",
            )
        ),
        help="Version contract shared by local, Colab, Docker, and GKE training",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/modernbert-smoke"),
    )
    parser.add_argument(
        "--device",
        help="Override config device (auto, cpu, mps, cuda)",
    )
    parser.add_argument(
        "--objective",
        choices=("ordinal", "pointwise", "pairwise"),
        default="ordinal",
        help="Training objective; pointwise accepts explicit observation masks; pairwise requires complete targets",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        help="Override the config step limit for a bounded smoke run",
    )
    parser.add_argument(
        "--initial-model-path",
        type=Path,
        help="Approved Hugging Face package used as a weight-only warm start",
    )
    parser.add_argument(
        "--initialization-mode",
        choices=("approved_anchor", "pristine_modernbert"),
        default="pristine_modernbert",
    )
    parser.add_argument("--observability-run-uri")
    parser.add_argument("--batch-id")
    parser.add_argument("--candidate-model-version")
    parser.add_argument("--progress-every-steps", type=int, default=25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_training_config(args.config)
    runtime_contract = load_runtime_contract(args.runtime_contract)

    def publish_progress(state) -> None:
        if not args.observability_run_uri:
            return
        if state.optimizer_steps % args.progress_every_steps and (
            state.optimizer_steps < state.total_steps
        ):
            return
        write_observability_json(
            f"{args.observability_run_uri.rstrip('/')}/progress/training.json",
            {
                "batch_id": args.batch_id,
                "candidate_model_version": args.candidate_model_version,
                "sequence": state.optimizer_steps,
                "optimizer_step": state.optimizer_steps,
                "total_steps": state.total_steps,
                "epoch": state.epoch_number,
                "total_epochs": state.total_epochs,
            },
            mutable=True,
        )

    if args.observability_run_uri and (
        not args.batch_id or not args.candidate_model_version
    ):
        raise SystemExit(
            "observability requires --batch-id and --candidate-model-version"
        )
    if args.progress_every_steps < 1:
        raise SystemExit("--progress-every-steps must be positive")
    result = run_training(
        args.training_path,
        args.output_dir,
        config,
        runtime_contract=runtime_contract,
        device_override=args.device,
        max_train_steps_override=args.max_train_steps,
        initial_model_dir=(
            args.initial_model_path
            if args.initialization_mode == "approved_anchor"
            else None
        ),
        on_progress=publish_progress if args.observability_run_uri else None,
        objective_method=args.objective,
    )
    if args.observability_run_uri:
        write_observability_json(
            f"{args.observability_run_uri.rstrip('/')}/metrics/training-final.json",
            {
                "batch_id": args.batch_id,
                "candidate_model_version": args.candidate_model_version,
                **result.metrics,
            },
        )
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
