from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..evaluation.routerarena_epoch_evaluation import RouterArenaEpochEvaluator
from .modernbert_colab_training import run_colab_training
from .modernbert_runtime import load_runtime_contract
from .modernbert_training import load_training_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run resumable ModernBERT training for preemptible Colab sessions, "
            "with completed checkpoints mirrored to persistent storage"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/config/modernbert_t4_full.json"),
    )
    parser.add_argument("--training-path", type=Path, required=True)
    parser.add_argument(
        "--runtime-contract",
        type=Path,
        default=Path(
            os.environ.get(
                "MODERNBERT_RUNTIME_CONTRACT",
                "ml/config/modernbert_runtime_cu124_v1.json",
            )
        ),
    )
    parser.add_argument(
        "--local-run-dir",
        type=Path,
        default=Path("/content/llm-router-training/modernbert-router-baseline-v1"),
        help="Fast ephemeral Colab disk used during the active session",
    )
    parser.add_argument(
        "--persistent-run-dir",
        type=Path,
        required=True,
        help="Mounted Google Drive directory that survives Colab session loss",
    )
    parser.add_argument(
        "--save-every-optimizer-steps",
        type=int,
        default=50,
        help="Fast local checkpoint interval on the active Colab VM",
    )
    parser.add_argument(
        "--drive-sync-every-optimizer-steps",
        type=int,
        default=500,
        help="Persistent Google Drive sync interval; epoch boundaries always sync",
    )
    parser.add_argument("--keep-last-checkpoints", type=int, default=2)
    parser.add_argument(
        "--routerbench-validations-per-epoch",
        type=int,
        default=3,
        help=(
            "Inference-only RouterBench validation observations per epoch; "
            "the Colab baseline uses 1/3, 2/3 and 3/3"
        ),
    )
    parser.add_argument(
        "--resume-mode",
        choices=("auto", "required", "never"),
        default="auto",
        help="auto resumes the newest complete checkpoint; never refuses overwrite",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-train-steps",
        type=int,
        help="Optional bounded run used only for smoke verification",
    )
    parser.add_argument(
        "--routerarena-evaluation-path",
        type=Path,
        help=(
            "Optional RouterArena evaluation Parquet. When set, inference-only "
            "metrics and charts are persisted after every completed epoch."
        ),
    )
    parser.add_argument(
        "--routerarena-metadata-path",
        type=Path,
        help="Defaults to metadata.parquet beside the RouterArena evaluation input",
    )
    parser.add_argument(
        "--routerarena-config",
        type=Path,
        default=Path("ml/config/routerarena_evaluation.json"),
    )
    parser.add_argument("--routerarena-batch-size", type=int)
    parser.add_argument("--routerarena-max-length", type=int)
    parser.add_argument(
        "--routerarena-device",
        choices=("cpu", "cuda", "mps", "auto"),
        default="cuda",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    epoch_evaluator = None
    if args.routerarena_evaluation_path is not None:
        metadata_path = args.routerarena_metadata_path
        if metadata_path is None:
            candidate = args.routerarena_evaluation_path.parent / "metadata.parquet"
            metadata_path = candidate if candidate.is_file() else None
        epoch_evaluator = RouterArenaEpochEvaluator(
            training_path=args.training_path,
            evaluation_path=args.routerarena_evaluation_path,
            metadata_path=metadata_path,
            config_path=args.routerarena_config,
            local_output_dir=args.local_run_dir / "routerarena-epoch-evaluations",
            persistent_output_dir=(
                args.persistent_run_dir / "routerarena-epoch-evaluations"
            ),
            batch_size=args.routerarena_batch_size,
            max_length=args.routerarena_max_length,
            device=args.routerarena_device,
        )
    result = run_colab_training(
        args.training_path,
        args.local_run_dir,
        args.persistent_run_dir,
        load_training_config(args.config),
        runtime_contract=load_runtime_contract(args.runtime_contract),
        save_every_optimizer_steps=args.save_every_optimizer_steps,
        drive_sync_every_optimizer_steps=args.drive_sync_every_optimizer_steps,
        keep_last_checkpoints=args.keep_last_checkpoints,
        resume_mode=args.resume_mode,
        device_override=args.device,
        max_train_steps_override=args.max_train_steps,
        epoch_evaluator=epoch_evaluator,
        routerbench_validations_per_epoch=(
            args.routerbench_validations_per_epoch
        ),
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "persistent_model_dir": str(result.persistent_model_dir),
                "resumed_from": (
                    None if result.resumed_from is None else str(result.resumed_from)
                ),
                "metrics": result.metrics,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
