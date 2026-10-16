"""RT-08 (02_rt08): retrain the old-definition R0 on v2-audited labels, same recipe as RT-04.

build  (local, CPU): A-split train + D_dev rows from 00_prepare, v2 exclusions dropped, label =
       old definition (overall solve rate, T1 .27/.64) recomputed on v2 outcomes. Writes
       training.parquet plus a manifest comparing the v1 and v2 label mix.
train  (Colab T4): run the RT-04 training function and config unchanged, with the optimizer
       step budget pinned to RT-04's 2,007 steps so only the labels differ.
Test rows are not read here; scoring happens once in 08_test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

A_SOURCES = ("routerbench", "xroutebench")
RT04_STEPS = 2007  # experiments/results/modernbert-initial-v1-20260924/run.json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(prepare_dir: Path, a_train: Path, a_dev: Path, output_dir: Path) -> dict:
    prep = json.loads((prepare_dir / "manifest.json").read_text())
    for name in ("train.parquet", "dev.parquet"):
        if sha256_file(prepare_dir / name) != prep["outputs"][name]["sha256"]:
            raise SystemExit(f"00_prepare output changed: {name}")
    original = pd.concat([pd.read_parquet(a_train), pd.read_parquet(a_dev)], ignore_index=True)
    original = original.set_index("sample_id")
    frames = []
    for role, split in (("train", "train"), ("dev", "validation")):
        part = pd.read_parquet(prepare_dir / f"{role}.parquet")
        part = part[part.source_dataset.isin(A_SOURCES)]
        frames.append(part.assign(split=split))
    rows = pd.concat(frames, ignore_index=True)
    dropped = int(rows["excluded_v2"].sum())
    rows = rows[~rows["excluded_v2"]].copy()
    meta = original.loc[rows["sample_id"], ["duplicate_group_id", "source_split", "category", "source_task"]]
    training = pd.DataFrame({
        "sample_id": rows["sample_id"].to_numpy(),
        "source_dataset": rows["source_dataset"].to_numpy(),
        "model_input_text": rows["model_input_text"].to_numpy(),
        "difficulty_label": rows["old_label_v2"].to_numpy(),
        "difficulty_label_v1": rows["old_label_v1"].to_numpy(),
        "sample_weight": 1.0,
        "split": rows["split"].to_numpy(),
        "duplicate_group_id": meta["duplicate_group_id"].to_numpy(),
        "source_split": meta["source_split"].to_numpy(),
        "category": meta["category"].to_numpy(),
        "source_task": meta["source_task"].to_numpy(),
    })
    if training.groupby("duplicate_group_id")["split"].nunique().max() != 1:
        raise SystemExit("a duplicate group crosses train and D_dev")
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "training.parquet"
    training.to_parquet(path, index=False)
    mix = lambda col: training.groupby("split")[col].value_counts(normalize=True).round(4).unstack(fill_value=0).to_dict("index")
    manifest = {
        "experiment": "RT-08", "label": "old definition on v2 outcomes (overall solve rate, hard < .27, easy >= .64)",
        "prepare_manifest_sha256": sha256_file(prepare_dir / "manifest.json"),
        "training_sha256": sha256_file(path), "rows": training.groupby(["split", "source_dataset"]).size().unstack().to_dict("index"),
        "dropped_v2_exclusions": dropped, "label_mix_v2": mix("difficulty_label"), "label_mix_v1": mix("difficulty_label_v1"),
        "label_changed_share": training.assign(c=training.difficulty_label != training.difficulty_label_v1)
        .groupby(["split", "source_dataset"])["c"].mean().round(4).unstack().to_dict("index"),
        "max_train_steps": RT04_STEPS,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def train(root: Path, data_dir: Path, persistent_dir: Path, pilot: bool) -> dict:
    from llm_router_labeling.training.modernbert_colab_training import run_colab_training
    from llm_router_labeling.training.modernbert_runtime import load_runtime_contract
    from llm_router_labeling.training.modernbert_training import load_training_config, validate_training_dataframe

    manifest = json.loads((data_dir / "manifest.json").read_text())
    path = data_dir / "training.parquet"
    if sha256_file(path) != manifest["training_sha256"]:
        raise SystemExit("training data checksum mismatch")
    frame = pd.read_parquet(path)
    validate_training_dataframe(frame)
    if set(frame["split"]) != {"train", "validation"} or not frame["sample_weight"].eq(1.0).all():
        raise SystemExit("RT-08 uses train/D_dev only with weight 1, as RT-04")
    if pilot:
        pieces = [frame[frame.split.eq(s)].sample(n=min(n, int(frame.split.eq(s).sum())), random_state=42)
                  for s, n in (("train", 1200), ("validation", 300))]
        path = data_dir.parent / "rt08-pilot.parquet"
        pd.concat(pieces, ignore_index=True).to_parquet(path, index=False)
    run_name = "pilot" if pilot else "full"
    config = load_training_config(root / "ml/config/modernbert_initial_v1_training.json")
    result = run_colab_training(
        path, Path("/content/rt08-runs") / run_name, persistent_dir / run_name, config,
        runtime_contract=load_runtime_contract(root / "ml/config/modernbert_runtime_cu124_v1.json"),
        max_train_steps_override=40 if pilot else RT04_STEPS,
        routerbench_validations_per_epoch=0 if pilot else 1,
        save_every_optimizer_steps=50, drive_sync_every_optimizer_steps=500,
    )
    if not pilot and result.metrics["optimization"]["optimizer_steps"] != RT04_STEPS:
        raise SystemExit("RT-08 did not complete RT-04's step budget")
    (persistent_dir / run_name / "rt08-data-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return {"status": result.status, "model_dir": str(result.persistent_model_dir), "metrics": result.metrics}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--prepare-dir", type=Path, required=True)
    b.add_argument("--a-train", type=Path, required=True)
    b.add_argument("--a-dev", type=Path, required=True)
    b.add_argument("--output-dir", type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("--root", type=Path, required=True, help="extracted source bundle")
    t.add_argument("--data-dir", type=Path, required=True)
    t.add_argument("--persistent-dir", type=Path, required=True)
    t.add_argument("--pilot", action="store_true")
    a = p.parse_args(argv)
    if a.command == "build":
        out = build(a.prepare_dir, a.a_train, a.a_dev, a.output_dir)
    else:
        out = train(a.root, a.data_dir, a.persistent_dir, a.pilot)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
