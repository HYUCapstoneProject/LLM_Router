from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .tfidf_baseline import run_tfidf_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen TF-IDF router baseline")
    parser.add_argument("--training-path", type=Path, required=True)
    parser.add_argument("--validation-path", type=Path, required=True)
    parser.add_argument("--dataset-manifest-path", type=Path, required=True)
    parser.add_argument("--experiment-config-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-role", choices=("t1", "t2"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_tfidf_baseline(
        training_path=arguments.training_path,
        validation_path=arguments.validation_path,
        dataset_manifest_path=arguments.dataset_manifest_path,
        experiment_config_path=arguments.experiment_config_path,
        output_dir=arguments.output_dir,
        threshold_role=arguments.threshold_role,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
