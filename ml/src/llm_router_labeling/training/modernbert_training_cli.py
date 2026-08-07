from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .modernbert_training import load_training_config, run_training
from .modernbert_runtime import load_runtime_contract


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
        "--max-train-steps",
        type=int,
        help="Override the config step limit for a bounded smoke run",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_training_config(args.config)
    runtime_contract = load_runtime_contract(args.runtime_contract)
    result = run_training(
        args.training_path,
        args.output_dir,
        config,
        runtime_contract=runtime_contract,
        device_override=args.device,
        max_train_steps_override=args.max_train_steps,
    )
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
