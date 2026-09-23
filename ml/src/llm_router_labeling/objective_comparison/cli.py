from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ..shared.dataset_io import load_config
from .data import build_outcome_dataset
from .selection import (
    build_validation_lock,
    evaluate_routerarena_alignment,
    evaluate_sealed_tests,
)
from .training import train_objective


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and run the frozen routing-objective comparison"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/config/routing_objective_comparison_v1.json"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-data")
    build.add_argument("--project-root", type=Path, default=Path("."))
    build.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/routing-objective-comparison-v1/dataset"),
    )

    train = subparsers.add_parser("train")
    train.add_argument("--method", choices=("ordinal", "pointwise", "pairwise", "regret"), required=True)
    train.add_argument("--training-path", type=Path, required=True)
    train.add_argument("--validation-path", type=Path, required=True)
    train.add_argument("--outcome-manifest-path", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--label-column", default="difficulty_label")
    train.add_argument("--device", default="auto")
    train.add_argument("--maximum-train-rows", type=int, default=0)
    train.add_argument("--maximum-validation-rows", type=int, default=0)
    train.add_argument("--maximum-train-steps", type=int, default=0)

    lock = subparsers.add_parser("lock-validation")
    lock.add_argument("--run-root", type=Path, action="append", required=True)
    lock.add_argument("--output-path", type=Path, required=True)
    lock.add_argument("--stage", choices=("provisional", "final"), required=True)

    test = subparsers.add_parser("evaluate-test")
    test.add_argument("--validation-lock-path", type=Path, required=True)
    test.add_argument("--run-root", type=Path, action="append", required=True)
    test.add_argument("--routerbench-test-path", type=Path, required=True)
    test.add_argument("--xroutebench-test-path", type=Path, required=True)
    test.add_argument("--output-dir", type=Path, required=True)
    test.add_argument("--device", default="auto")

    arena = subparsers.add_parser("evaluate-routerarena")
    arena.add_argument("--validation-lock-path", type=Path, required=True)
    arena.add_argument("--run-root", type=Path, action="append", required=True)
    arena.add_argument("--routerarena-path", type=Path, required=True)
    arena.add_argument("--output-dir", type=Path, required=True)
    arena.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "build-data":
        result = build_outcome_dataset(
            config,
            project_root=args.project_root.resolve(),
            output_dir=args.output_dir,
        )
    elif args.command == "train":
        result = train_objective(
            method=args.method,
            training_path=args.training_path,
            validation_path=args.validation_path,
            outcome_manifest_path=args.outcome_manifest_path,
            experiment_config_path=args.config,
            output_dir=args.output_dir,
            seed=args.seed,
            label_column=args.label_column,
            device=args.device,
            maximum_train_rows=args.maximum_train_rows,
            maximum_validation_rows=args.maximum_validation_rows,
            maximum_train_steps=args.maximum_train_steps,
        )
    elif args.command == "lock-validation":
        result = build_validation_lock(
            run_roots=args.run_root,
            experiment_config_path=args.config,
            output_path=args.output_path,
            stage=args.stage,
        )
    elif args.command == "evaluate-test":
        result = evaluate_sealed_tests(
            validation_lock_path=args.validation_lock_path,
            run_roots=args.run_root,
            routerbench_test_path=args.routerbench_test_path,
            xroutebench_test_path=args.xroutebench_test_path,
            experiment_config_path=args.config,
            output_dir=args.output_dir,
            device=args.device,
        )
    else:
        result = evaluate_routerarena_alignment(
            validation_lock_path=args.validation_lock_path,
            run_roots=args.run_root,
            routerarena_path=args.routerarena_path,
            experiment_config_path=args.config,
            output_dir=args.output_dir,
            device=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
