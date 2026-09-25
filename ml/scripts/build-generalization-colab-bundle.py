#!/usr/bin/env python3
"""Freeze G1 Colab code and pre-holdout data in a verified, new-only bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

DATA_FILES = (
    "manifest.json",
    "_SUCCESS",
    "config.json",
    "arms.json",
    "split-audit.json",
    "candidate-statistics.parquet",
    "train.parquet",
    "calibration.parquet",
    "selection.parquet",
)
FIXED_FILES = (
    "ml/pyproject.toml",
    "ml/requirements-training.lock",
    "ml/requirements-objective-initial-v1.lock",
    "ml/config/routing_generalization_v1.json",
    "ml/config/modernbert_runtime_cu124_v1.json",
    "ml/scripts/run-generalization-colab.py",
    "ml/scripts/build-generalization-colab-bundle.py",
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build(project_root: Path, output_dir: Path) -> dict:
    root = project_root.resolve()
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError("Bundle output must be a new directory")
    # Import the specified checkout, not a globally installed package.
    sys.path.insert(0, str(root / "ml/src"))
    from llm_router_labeling.generalization.data import (
        canonical_config_sha256,
        verify_generalization_dataset,
    )

    dataset = root / "ml/artifacts/routing-generalization-v1/dataset"
    roles = ["train", "calibration", "selection"]
    manifest = verify_generalization_dataset(dataset, roles=roles)
    config = json.loads((root / "ml/config/routing_generalization_v1.json").read_text())
    if manifest["config_sha256"] != canonical_config_sha256(config):
        raise ValueError("Dataset and requested experiment configuration differ")

    paths = {name: root / name for name in FIXED_FILES}
    for path in sorted((root / "ml/src").rglob("*.py")):
        if "__pycache__" not in path.parts:
            paths[path.relative_to(root).as_posix()] = path
    for name in DATA_FILES:
        paths[f"data/dataset/{name}"] = dataset / name
    for name, path in paths.items():
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError(
                f"Bundle input must be a regular file inside checkout: {name}"
            )
    paths = dict(sorted(paths.items()))
    hashes = {name: sha256(path) for name, path in paths.items()}
    sizes = {name: path.stat().st_size for name, path in paths.items()}
    code_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "ml/src",
            "ml/config",
            "ml/scripts",
            "ml/pyproject.toml",
            "ml/requirements-training.lock",
            "ml/requirements-objective-initial-v1.lock",
        ],
        cwd=root,
        text=True,
    ).strip()
    provenance = {
        "schema_version": "generalization-colab-input-v1",
        "experiment_version": config["experiment_version"],
        "code_commit": code_commit,
        "uncommitted_source_changes": bool(dirty),
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "dataset_config_sha256": manifest["config_sha256"],
        "dataset_dir": "data/dataset",
        "included_roles": roles,
        "included_role_rows": {role: manifest["roles"][role]["rows"] for role in roles},
        "task_holdout_included": False,
        "legacy_test_or_routerarena_included": False,
        "production_approved": False,
        "files": hashes,
        "file_sizes": sizes,
    }
    manifest_bytes = (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode()
    output.mkdir(parents=True, exist_ok=False)
    archive_path = output / "colab-input.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, path in paths.items():
            archive.add(path, arcname=name, recursive=False)
        entry = tarfile.TarInfo("input-manifest.json")
        entry.size = len(manifest_bytes)
        entry.mode = 0o644
        archive.addfile(entry, io.BytesIO(manifest_bytes))
    # Verify archived bytes, not just the source files; mutation during packing fails.
    with tarfile.open(archive_path, "r:gz") as archive:
        expected_names = set(paths) | {"input-manifest.json"}
        members = archive.getmembers()
        if (
            len(members) != len(expected_names)
            or {m.name for m in members} != expected_names
        ):
            raise ValueError("Archive member inventory mismatch")
        for member in members:
            if not member.isfile():
                raise ValueError("Archive must contain only regular files")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("Missing archive payload")
            with handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            expected = (
                hashlib.sha256(manifest_bytes).hexdigest()
                if member.name == "input-manifest.json"
                else hashes[member.name]
            )
            if digest != expected:
                raise ValueError(f"Archive input changed while packing: {member.name}")
    if any(sha256(path) != hashes[name] for name, path in paths.items()):
        raise ValueError(
            "Input changed during bundle creation; use a new output directory"
        )
    (output / "input-manifest.json").write_bytes(manifest_bytes)
    summary = {
        "archive": str(archive_path),
        "sha256": sha256(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "file_count": len(paths),
        "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
        "code_commit": code_commit,
        "uncommitted_source_changes": bool(dirty),
        "task_holdout_included": False,
        "archive_verified": True,
    }
    (output / "bundle-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
