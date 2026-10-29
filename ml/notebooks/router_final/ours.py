"""EV-10 03_ours: train R0-group and R1-group (blended data) and R1-group-A (A split only).

build (local, CPU): from 00_prepare train/dev, drop v2-excluded rows and rows missing a tier
      probability, and write two training sets in the objective-comparison format:
        blended/  all four sources, per-row weight = equal total weight per source (mean 1)
        a_only/   RouterBench + xRouteBench only, same weighting over the two sources
      A tier whose whole group has no valid record is filled from its neighbours (training only).
      Targets: difficulty_label = needed tier (low/medium/high -> easy/medium/hard) for R0-group;
      success_* = quality_* = group success probability (soft) for R1-group; decision and actual
      costs = the fixed service cost (.2/.6/1.0).
train (Colab T4): the RT-05 trainer `train_objective` with the RT-05 experiment config, seed 42 (09_seeds: 17, 73).
      method ordinal = R0-group, pointwise = R1-group. Test rows are never read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import needed_tier

TIERS = ("low", "medium", "high")
TIER_TO_LABEL = {"low": "easy", "medium": "medium", "high": "hard"}
A_SOURCES = ("routerbench", "xroutebench")
RUNS = {  # notebook name -> (data variant, method)
    "r0_group": ("blended", "ordinal"),
    "r1_group": ("blended", "pointwise"),
    "r1_group_a": ("a_only", "pointwise"),
    "r2_group": ("blended", "pairwise"),
    "r3_group": ("blended", "regret"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balanced_weight(frame: pd.DataFrame) -> pd.Series:
    """Each source gets the same total weight inside a split; weights average 1."""
    counts = frame.groupby("source_dataset")["sample_id"].transform("count")
    weight = 1.0 / counts
    return weight * len(frame) / weight.sum()


def to_objective_frame(rows: pd.DataFrame, category: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({
        "sample_id": rows["sample_id"].to_numpy(),
        "model_input_text": rows["model_input_text"].to_numpy(),
        "source_dataset": rows["source_dataset"].to_numpy(),
        "category": category.to_numpy(),
        "difficulty_label": rows["needed_tier"].map(TIER_TO_LABEL).to_numpy(),
        "sample_weight": balanced_weight(rows).to_numpy(),
    })
    for t in TIERS:
        out[f"success_{t}"] = rows[f"p_{t}"].to_numpy(float)
        out[f"quality_{t}"] = rows[f"p_{t}"].to_numpy(float)
        out[f"decision_cost_{t}"] = rows[f"service_cost_{t}"].to_numpy(float)
        out[f"actual_cost_{t}"] = rows[f"service_cost_{t}"].to_numpy(float)
    return out


def impute_single_missing_tier(rows: pd.DataFrame):
    """Fill one missing tier probability from its neighbours (training only).

    A tier is missing when every model of that group has no valid record (failed calls are missing,
    not wrong answers). Dropping those rows would remove mostly hard questions, so a single missing
    tier is filled with the mean of the adjacent observed tiers (edge tiers copy their neighbour).
    Rows with two or more missing tiers are dropped. Evaluation never uses imputed values.
    """
    probs = rows[[f"p_{t}" for t in TIERS]].to_numpy(float).copy()
    missing = np.isnan(probs)
    count = missing.sum(1)
    for i in np.where(count == 1)[0]:
        j = int(np.where(missing[i])[0][0])
        neighbours = [probs[i, k] for k in (j - 1, j + 1) if 0 <= k < 3 and not np.isnan(probs[i, k])]
        probs[i, j] = float(np.mean(neighbours))
    out = rows.copy()
    for k, t in enumerate(TIERS):
        out[f"p_{t}"] = probs[:, k]
    return out, pd.Series(count == 1, index=rows.index), pd.Series(count >= 2, index=rows.index)


def build(prepare_dir: Path, a_train: Path, a_dev: Path, output_dir: Path) -> dict:
    prep = json.loads((prepare_dir / "manifest.json").read_text())
    for name in ("train.parquet", "dev.parquet"):
        if sha256_file(prepare_dir / name) != prep["outputs"][name]["sha256"]:
            raise SystemExit(f"00_prepare output changed: {name}")
    original = pd.concat([pd.read_parquet(a_train, columns=["sample_id", "category"]),
                          pd.read_parquet(a_dev, columns=["sample_id", "category"])]).set_index("sample_id")["category"]
    tau = float(prep["config"]["tau"])
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"tau": tau, "prepare_manifest_sha256": sha256_file(prepare_dir / "manifest.json"), "variants": {}}
    for variant, sources in (("blended", None), ("a_only", A_SOURCES)):
        target = output_dir / variant
        target.mkdir()
        record = {"files": {}, "rows": {}, "dropped": {}}
        for role, split in (("train", "train"), ("dev", "validation")):
            rows = pd.read_parquet(prepare_dir / f"{role}.parquet")
            if sources:
                rows = rows[rows.source_dataset.isin(sources)]
            excluded = rows["excluded_v2"].fillna(False).astype(bool)
            rows, imputed, too_sparse = impute_single_missing_tier(rows)
            record["dropped"][split] = {"v2_excluded": int(excluded.sum()), "two_or_more_tiers_missing": int((too_sparse & ~excluded).sum())}
            record.setdefault("imputed_one_tier", {})[split] = rows.loc[imputed & ~excluded & ~too_sparse].groupby("source_dataset").size().to_dict()
            rows = rows.assign(tier_imputed=imputed)[~excluded & ~too_sparse].reset_index(drop=True)
            # 00_prepare read a missing tier as 0; relabel from the filled probabilities.
            probs = rows[[f"p_{t}" for t in TIERS]].to_numpy(float)
            rows["needed_tier"] = [TIERS[i] for i in needed_tier(probs, tau)]
            category = rows["sample_id"].map(original).fillna(rows.get("slice", rows["source_dataset"])).fillna(rows["source_dataset"])
            frame = to_objective_frame(rows, category)
            path = target / f"{split}.parquet"
            frame.to_parquet(path, index=False)
            record["files"][path.name] = sha256_file(path)
            record["rows"][split] = frame.groupby("source_dataset").size().to_dict()
            record.setdefault("label_mix", {})[split] = frame.groupby("source_dataset")["difficulty_label"].value_counts(normalize=True).round(4).unstack(fill_value=0).to_dict("index")
            record.setdefault("weight_by_source", {})[split] = frame.groupby("source_dataset")["sample_weight"].sum().round(1).to_dict()
        outcome = {"dataset_version": f"ev10-router-final-v1-{variant}", "prepare_manifest_sha256": manifest["prepare_manifest_sha256"]}
        (target / "outcome-manifest.json").write_text(json.dumps(outcome, indent=2))
        manifest["variants"][variant] = record
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def train(root: Path, data_dir: Path, run: str, output_dir: Path, smoke: bool, device: str, seed: int = 42) -> dict:
    from llm_router_labeling.objective_comparison.training import train_objective
    manifest = json.loads((data_dir / "manifest.json").read_text())
    variant, method = RUNS[run]
    folder = data_dir / variant
    for name, digest in manifest["variants"][variant]["files"].items():
        if sha256_file(folder / name) != digest:
            raise SystemExit(f"training data checksum mismatch: {variant}/{name}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit("output directory must be new and empty")
    return train_objective(
        method=method, training_path=folder / "train.parquet", validation_path=folder / "validation.parquet",
        outcome_manifest_path=folder / "outcome-manifest.json",
        experiment_config_path=root / "ml/config/routing_objective_continual_v1.json",
        output_dir=output_dir, seed=seed, device=device,
        maximum_train_rows=64 if smoke else 0, maximum_validation_rows=64 if smoke else 0,
        maximum_train_steps=2 if smoke else 0,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--prepare-dir", type=Path, required=True)
    b.add_argument("--a-train", type=Path, required=True)
    b.add_argument("--a-dev", type=Path, required=True)
    b.add_argument("--output-dir", type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("--root", type=Path, required=True)
    t.add_argument("--data-dir", type=Path, required=True)
    t.add_argument("--run", choices=sorted(RUNS), required=True)
    t.add_argument("--output-dir", type=Path, required=True)
    t.add_argument("--smoke", action="store_true")
    t.add_argument("--device", default="auto")
    t.add_argument("--seed", type=int, default=42, help="42 for 03; 17 and 73 for 09_seeds")
    a = p.parse_args(argv)
    if a.command == "build":
        out = build(a.prepare_dir, a.a_train, a.a_dev, a.output_dir)
    else:
        out = train(a.root, a.data_dir, a.run, a.output_dir, a.smoke, a.device, a.seed)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
