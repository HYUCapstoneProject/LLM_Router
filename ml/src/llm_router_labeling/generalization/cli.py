"""Local-only dataset preparation and integrity verification CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import ROLES, build_generalization_dataset, verify_generalization_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--project-root", type=Path, default=Path.cwd())
    prepare.add_argument("--output-dir", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--dataset-dir", type=Path, required=True)
    verify.add_argument("--roles", choices=ROLES, nargs="+")
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = build_generalization_dataset(
            json.loads(args.config.read_text()),
            project_root=args.project_root,
            output_dir=args.output_dir,
        )
    else:
        manifest = verify_generalization_dataset(args.dataset_dir, roles=args.roles)
    print(
        json.dumps(
            {
                "dataset_version": manifest["dataset_version"],
                "config_sha256": manifest["config_sha256"],
                "roles": manifest["roles"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
