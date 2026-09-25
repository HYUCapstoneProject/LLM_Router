#!/usr/bin/env python3
"""Build a verified Colab input bundle without sealed evaluation questions."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("ml/artifacts/modernbert-initial-v1/dataset"))
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts/modernbert-initial-v1/colab-input.tar.gz"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    dataset = (root / args.dataset).resolve()
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest["threshold_selection_partition"] != "D_initial_train" or manifest["coverage_resolution_status"] != "complete":
        raise ValueError("Freeze thresholds and resolve coverage before training")
    if manifest["effective_sample_weight"] != 1.0:
        raise ValueError("Initial anchor requires effective weight 1")
    for name, info in manifest["artifacts"].items():
        path = dataset / name
        if not path.resolve().is_relative_to(dataset) or sha256(path) != info["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    files = [
        root / "ml/pyproject.toml",
        root / "ml/requirements-training.lock",
        root / "ml/requirements-training-cu124.txt",
        root / "ml/config/modernbert_initial_v1_data.json",
        root / "ml/config/modernbert_initial_v1_training.json",
        root / "ml/config/modernbert_runtime_cu124_v1.json",
        root / "ml/scripts/run-modernbert-initial-v1.py",
        *sorted((root / "ml/src").rglob("*.py")),
    ]
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "ml/src", "ml/config", "ml/scripts"], cwd=root, text=True).strip()
    provenance = {
        "schema_version": "initial-colab-input-v1",
        "logical_model_version": "1.0",
        "code_commit": code_commit,
        "uncommitted_source_changes": bool(dirty),
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "files": {str(path.relative_to(root)): sha256(path) for path in files},
        "training_sha256": sha256(dataset / "training.parquet"),
        "selection": manifest["selected_thresholds"],
        "sealed_questions_included": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = args.output.parent / "colab-input-manifest.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    with tarfile.open(args.output, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(root)))
        archive.add(dataset / "training.parquet", arcname="data/training.parquet")
        archive.add(dataset / "manifest.json", arcname="data/dataset-manifest.json")
        archive.add(provenance_path, arcname="input-manifest.json")
    print(json.dumps({"archive": str(args.output), "sha256": sha256(args.output), "code_commit": code_commit, "uncommitted_source_changes": bool(dirty)}, indent=2))


if __name__ == "__main__":
    main()
