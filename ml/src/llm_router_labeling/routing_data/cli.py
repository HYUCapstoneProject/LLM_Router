from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..shared.dataset_io import download_source, load_config
from .preprocessing import build_comparison_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the RouterBench + xRouteBench comparison dataset"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/config/router_model_comparison_v1.json"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser(
        "download-xroutebench", help="Download and verify pinned xRouteBench sources"
    )
    download.add_argument("--data-dir", type=Path)

    build = subparsers.add_parser(
        "build", help="Build the versioned dataset and train-only threshold selection"
    )
    build.add_argument("--project-root", type=Path, default=Path("."))
    build.add_argument("--xroute-data-dir", type=Path)
    build.add_argument(
        "--routerarena-path",
        type=Path,
        default=Path("ml/artifacts/routerarena-evaluation-full/evaluation.parquet"),
    )
    build.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/router-model-comparison-v1/dataset"),
    )
    return parser


def _default_xroute_dir(config: dict[str, Any]) -> Path:
    return Path("ml/data/raw/xroutebench") / str(config["xroutebench"]["revision"])


def _download(config: dict[str, Any], data_dir: Path) -> dict[str, str]:
    dataset = config["xroutebench"]
    revision = str(dataset["revision"])
    base_url = f"https://huggingface.co/datasets/{dataset['repo_id']}/resolve/{revision}"
    results: dict[str, str] = {}
    for name, source in dataset["sources"].items():
        download_config = {
            **source,
            "url": f"{base_url}/{source['path']}",
        }
        results[name] = str(download_source(download_config, data_dir))
    return results


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "download-xroutebench":
        result: dict[str, Any] = _download(
            config, args.data_dir or _default_xroute_dir(config)
        )
    else:
        result = build_comparison_dataset(
            config,
            project_root=args.project_root.resolve(),
            xroute_data_dir=args.xroute_data_dir or _default_xroute_dir(config),
            routerarena_path=args.routerarena_path,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
