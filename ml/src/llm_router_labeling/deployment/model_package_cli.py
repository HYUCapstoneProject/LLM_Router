from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model_package import (
    assemble_model_package,
    create_model_manifest,
    validate_model_package,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or validate a model package")
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--package-dir", type=Path, required=True)
    create.add_argument("--metadata", type=Path, required=True)
    create.add_argument("--output", type=Path)
    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--checkpoint-dir", type=Path, required=True)
    assemble.add_argument("--metadata", type=Path, required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)
    assemble.add_argument("--evaluation-metrics", type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--package-dir", type=Path, required=True)
    validate.add_argument("--manifest", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command in {"create", "assemble"}:
        with args.metadata.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if args.command == "create":
            result = create_model_manifest(args.package_dir, metadata, args.output)
        else:
            result = assemble_model_package(
                args.checkpoint_dir,
                args.output_dir,
                metadata,
                args.evaluation_metrics,
            )
    else:
        result = validate_model_package(args.package_dir, args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
