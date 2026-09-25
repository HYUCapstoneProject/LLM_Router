#!/usr/bin/env python3
"""Run one frozen R0/R1/R2 comparison arm, with resumable checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_router_labeling.objective_comparison.shared_training import (
    train_shared_objective,
)
from llm_router_labeling.shared.dataset_io import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("ordinal", "pointwise", "pairwise"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--import-r0-dir", type=Path)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    bundle_manifest = json.loads((root / "input-manifest.json").read_text())
    for name, expected in bundle_manifest["files"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Comparison input checksum mismatch: {name}")
    result = train_shared_objective(
        method=args.method,
        base_training_path=root / "data/training.parquet",
        outcome_dir=root / "data/outcomes",
        experiment_config_path=root / "ml/config/routing_objective_continual_v1.json",
        training_config_path=root / "ml/config/modernbert_initial_v1_training.json",
        runtime_config_path=root / "ml/config/modernbert_runtime_cu124_v1.json",
        output_dir=args.output_dir,
        import_r0_dir=args.import_r0_dir,
        maximum_train_steps=40 if args.pilot else 0,
    )
    print(
        json.dumps(
            {
                "method": args.method,
                "pilot": args.pilot,
                "optimizer_steps": result["metrics"]["optimizer_steps"],
                "summary": result["metrics"]["summary"],
                "model_dir": result["manifest"]["model_dir"],
                "production_activated": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
