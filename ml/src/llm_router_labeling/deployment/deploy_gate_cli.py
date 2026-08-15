from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..shared.dataset_io import write_json
from ..shared.observability import write_observability_json
from .deploy_gate import evaluate_deploy_gate
from .model_package import validate_model_package


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate model promotion rules")
    parser.add_argument("--candidate-package", type=Path, required=True)
    parser.add_argument("--candidate-metrics", type=Path, required=True)
    parser.add_argument("--candidate-model-version", required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--baseline-model-version")
    parser.add_argument("--observability-run-uri")
    parser.add_argument("--batch-id")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_deploy_gate(
        _read_json(args.candidate_metrics),
        _read_json(args.rules),
        candidate_model_version=args.candidate_model_version,
        package_validation=validate_model_package(args.candidate_package),
        baseline_metrics=(
            _read_json(args.baseline_metrics) if args.baseline_metrics else None
        ),
        baseline_model_version=args.baseline_model_version,
    )
    write_json(args.output, result)
    if args.observability_run_uri:
        if not args.batch_id:
            raise SystemExit("observability requires --batch-id")
        write_observability_json(
            f"{args.observability_run_uri.rstrip('/')}/gate/gate-result.json",
            {
                "batch_id": args.batch_id,
                "candidate_model_version": args.candidate_model_version,
                **result,
            },
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
