#!/usr/bin/env python3
"""Run the verified R0 anchor pilot or full Colab training, using D_dev only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from llm_router_labeling.training.modernbert_colab_training import run_colab_training
from llm_router_labeling.training.modernbert_training import load_training_config, validate_training_dataframe
from llm_router_labeling.training.modernbert_runtime import load_runtime_contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/content/llm-router-initial-v1"))
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "input-manifest.json").read_text())
    for name, digest in manifest["files"].items():
        with (root / name).open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                raise ValueError(f"Code/config checksum mismatch: {name}")
    training = root / "data/training.parquet"
    with training.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != manifest["training_sha256"]:
            raise ValueError("Training data checksum mismatch")
    frame = pd.read_parquet(training)
    validate_training_dataframe(frame)
    if set(frame["split"]) != {"train", "validation"} or not frame["sample_weight"].eq(1.0).all():
        raise ValueError("Initial anchor must use train/D_dev only with effective weight 1")
    if frame.groupby("duplicate_group_id")["split"].nunique().max() != 1:
        raise ValueError("Training and development validation groups overlap")
    run_name = "pilot" if args.pilot else "full"
    if args.pilot:
        # Deterministic natural-distribution pilot; full training starts afresh.
        pieces = []
        for split, limit in (("train", 1200), ("validation", 300)):
            rows = frame.loc[frame["split"].eq(split)]
            pieces.append(rows.sample(n=min(len(rows), limit), random_state=42))
        training = root / "data/pilot.parquet"
        pd.concat(pieces, ignore_index=True).to_parquet(training, index=False)
    if not args.persistent_root.is_dir():
        raise ValueError("Create the durable project directory after mounting Drive")
    config = load_training_config(root / "ml/config/modernbert_initial_v1_training.json")
    result = run_colab_training(
        training, root / "runs" / run_name, args.persistent_root / run_name, config,
        runtime_contract=load_runtime_contract(root / "ml/config/modernbert_runtime_cu124_v1.json"),
        max_train_steps_override=40 if args.pilot else None,
        routerbench_validations_per_epoch=0 if args.pilot else 1,
        save_every_optimizer_steps=50,
        drive_sync_every_optimizer_steps=500,
    )
    if not args.pilot:
        optimization = result.metrics["optimization"]
        if optimization["optimizer_steps"] != optimization["planned_steps"]:
            raise ValueError("Full initial training did not complete its planned steps")
    (args.persistent_root / run_name / "input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": result.status, "model_dir": str(result.persistent_model_dir), "metrics": result.metrics}, indent=2))


if __name__ == "__main__":
    main()
