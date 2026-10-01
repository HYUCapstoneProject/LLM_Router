from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..routing_data.outcome_audit import apply_routerbench_audit, load_outcome_audit
from ..routing_data.preprocessing import (
    _routerbench_accepted,
    _routerbench_models,
    _validate_xroute_rows,
)
from ..shared.dataset_io import write_json

TIERS = ("low", "medium", "high")
SPLIT_FILES = {
    "train": "train/training.parquet",
    "validation": "validation/validation.parquet",
    "test_routerbench": "test/routerbench.parquet",
    "test_xroutebench": "test/xroutebench.parquet",
}


class ObjectiveDataError(ValueError):
    """Raised when the common routing-outcome matrix is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ObjectiveDataError(f"Expected a JSON object: {path}")
    return value


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_anchor_rows(
    base_dataset_dir: Path,
    *,
    split_files: Mapping[str, str] = SPLIT_FILES,
    artifact_contracts: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for logical_split, relative in split_files.items():
        path = base_dataset_dir / relative
        if not path.is_file():
            raise ObjectiveDataError(f"Missing frozen comparison split: {path}")
        if artifact_contracts is not None:
            contract = artifact_contracts.get(relative)
            if not contract or _sha256(path) != contract["sha256"]:
                raise ObjectiveDataError(
                    f"Frozen comparison split checksum mismatch: {path}"
                )
        frame = pd.read_parquet(path)
        frame["objective_split"] = logical_split
        frames.append(frame)
    anchor = pd.concat(frames, ignore_index=True)
    if anchor.empty or anchor["sample_id"].duplicated().any():
        raise ObjectiveDataError("Frozen comparison rows must be non-empty and unique")
    required = {
        "sample_id",
        "source_dataset",
        "model_input_text",
        "difficulty_label",
        "duplicate_group_id",
    }
    missing = sorted(required - set(anchor.columns))
    if missing:
        raise ObjectiveDataError(
            f"Frozen comparison rows are missing columns: {missing}"
        )
    if anchor.groupby("duplicate_group_id")["objective_split"].nunique().gt(1).any():
        raise ObjectiveDataError("A duplicate group crosses objective splits")
    return anchor


def _routerbench_matrices(
    config: Mapping[str, Any], project_root: Path, anchor: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    section = config["routerbench"]
    labeled_path = _resolve(project_root, section["labeled_path"])
    tiers_path = _resolve(project_root, section["model_tiers_path"])
    labeled = pd.read_parquet(labeled_path)
    models = _routerbench_models(tiers_path)
    expected_columns = set(models) | {f"{model}|total_cost" for model in models}
    missing = sorted(expected_columns - set(labeled.columns))
    if missing:
        raise ObjectiveDataError(f"RouterBench outcomes are missing columns: {missing}")
    labeled = labeled.copy()
    labeled["sample_id"] = "routerbench:" + labeled["sample_id"].astype(str)
    wanted = anchor.loc[anchor["source_dataset"].eq("routerbench"), "sample_id"]
    labeled = labeled.set_index("sample_id").reindex(wanted)
    if labeled.index.has_duplicates or labeled["query_text"].isna().any():
        raise ObjectiveDataError(
            "RouterBench outcomes do not align with frozen sample IDs"
        )
    quality = labeled.loc[:, models].apply(pd.to_numeric, errors="coerce")
    quality.columns = models
    raw_for_acceptance = labeled.reset_index().copy()
    raw_for_acceptance["sample_id"] = raw_for_acceptance["sample_id"].str.removeprefix(
        "routerbench:"
    )
    success = _routerbench_accepted(raw_for_acceptance, models)
    audit = load_outcome_audit(config, project_root)
    if audit is not None:
        quality.index = range(len(quality))
        success, _ = apply_routerbench_audit(success, raw_for_acceptance, audit, quality)
        quality.index = wanted.to_numpy()
    success.index = (
        "routerbench:" + raw_for_acceptance["sample_id"].astype(str).to_numpy()
    )
    success = success.reindex(wanted)
    cost = labeled.loc[:, [f"{model}|total_cost" for model in models]].apply(
        pd.to_numeric, errors="coerce"
    )
    cost.columns = models
    return quality, success, cost


def _xroute_matrices(
    config: Mapping[str, Any], project_root: Path, anchor: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    section = config["xroutebench"]
    data_dir = _resolve(project_root, section["data_dir"])
    candidates = pd.read_parquet(data_dir / section["candidates_file"])
    models = candidates["model_name"].astype(str).tolist()
    price = candidates.set_index("model_name")
    frames: list[pd.DataFrame] = []
    for split, filename in (
        ("train", section["train_file"]),
        ("test", section["test_file"]),
    ):
        source_rows = anchor.loc[anchor["source_dataset"].eq("xroutebench")]
        if "source_split" in source_rows and split not in set(
            source_rows["source_split"]
        ):
            continue
        raw = pd.read_parquet(data_dir / filename)
        validated = _validate_xroute_rows(
            raw,
            split=split,
            candidates=models,
            exact_metrics=set(section["exact_metrics"]),
            continuous_metrics=set(section["continuous_metrics"]),
            structured=config.get("input_contract") == "structure-preserving-v2",
            outcome_audit=load_outcome_audit(config, project_root),
        )
        validated["sample_id"] = validated["query_key"].map(
            lambda key: (
                "xroutebench:"
                + hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:20]
            )
        )
        input_prices = price.loc[
            validated["model_name"].astype(str), "input_price_per_1m"
        ].to_numpy(float)
        output_prices = price.loc[
            validated["model_name"].astype(str), "output_price_per_1m"
        ].to_numpy(float)
        validated["actual_cost"] = (
            pd.to_numeric(validated["input_tokens"], errors="coerce").to_numpy(float)
            * input_prices
            + pd.to_numeric(validated["output_tokens"], errors="coerce").to_numpy(float)
            * output_prices
        ) / 1_000_000.0
        frames.append(validated)
    rows = pd.concat(frames, ignore_index=True)
    wanted = anchor.loc[anchor["source_dataset"].eq("xroutebench"), "sample_id"]
    quality = rows.pivot(index="sample_id", columns="model_name", values="performance")
    success = rows.pivot(index="sample_id", columns="model_name", values="accepted")
    cost = rows.pivot(index="sample_id", columns="model_name", values="actual_cost")
    for matrix in (quality, success, cost):
        matrix.index = matrix.index.astype(str)
    quality = quality.reindex(index=wanted, columns=models)
    success = success.reindex(index=wanted, columns=models)
    cost = cost.reindex(index=wanted, columns=models)
    if quality.index.has_duplicates or quality.isna().all(axis=1).any():
        raise ObjectiveDataError(
            "xRouteBench outcomes do not align with frozen sample IDs"
        )
    return quality, success, cost


def pareto_frontier(statistics: pd.DataFrame, tolerance: float = 1e-12) -> pd.DataFrame:
    required = {"model_name", "mean_quality", "mean_cost"}
    missing = sorted(required - set(statistics.columns))
    if missing:
        raise ObjectiveDataError(f"Candidate statistics are missing columns: {missing}")
    working = statistics.dropna(subset=list(required)).sort_values(
        ["mean_cost", "mean_quality", "model_name"],
        ascending=[True, False, True],
        kind="stable",
    )
    frontier_rows: list[pd.Series] = []
    best_quality = -math.inf
    for _, row in working.iterrows():
        quality = float(row["mean_quality"])
        if quality > best_quality + tolerance:
            frontier_rows.append(row)
            best_quality = quality
    if not frontier_rows:
        raise ObjectiveDataError("No Pareto candidate remains")
    return pd.DataFrame(frontier_rows).reset_index(drop=True)


def select_three_arms(
    quality: pd.DataFrame,
    cost: pd.DataFrame,
    train_ids: Sequence[str],
    *,
    tolerance: float = 1e-12,
    minimum_frontier_size: int = 3,
) -> tuple[dict[str, str], pd.DataFrame]:
    train_index = pd.Index(train_ids, dtype="object")
    quality_train = quality.reindex(train_index)
    cost_train = cost.reindex(train_index)
    statistics = pd.DataFrame(
        {
            "model_name": list(quality.columns),
            "mean_quality": [
                float(quality_train[column].mean()) for column in quality.columns
            ],
            "mean_cost": [
                float(cost_train[column].mean()) for column in quality.columns
            ],
            "valid_rows": [
                int(quality_train[column].notna().sum()) for column in quality.columns
            ],
        }
    )
    frontier = pareto_frontier(statistics, tolerance=tolerance)
    if len(frontier) < minimum_frontier_size:
        raise ObjectiveDataError(
            f"Need at least {minimum_frontier_size} Pareto candidates, got {len(frontier)}"
        )
    low = frontier.iloc[0]
    high = frontier.iloc[-1]
    middle = frontier.iloc[1:-1].copy()
    cost_span = float(high["mean_cost"] - low["mean_cost"])
    quality_span = float(high["mean_quality"] - low["mean_quality"])
    if cost_span <= 0.0 or quality_span <= 0.0 or middle.empty:
        raise ObjectiveDataError("Pareto extremes do not define three ordered actions")
    x = (middle["mean_cost"] - float(low["mean_cost"])) / cost_span
    y = (middle["mean_quality"] - float(low["mean_quality"])) / quality_span
    middle["knee_distance"] = np.abs(y - x) / math.sqrt(2.0)
    knee = middle.sort_values(
        ["knee_distance", "mean_quality", "model_name"],
        ascending=[False, False, True],
        kind="stable",
    ).iloc[0]
    selected = {
        "low": str(low["model_name"]),
        "medium": str(knee["model_name"]),
        "high": str(high["model_name"]),
    }
    statistics["on_pareto_frontier"] = statistics["model_name"].isin(
        set(frontier["model_name"].astype(str))
    )
    inverse = {model: tier for tier, model in selected.items()}
    statistics["selected_tier"] = statistics["model_name"].map(inverse).fillna("")
    return selected, statistics.sort_values("mean_cost", kind="stable").reset_index(
        drop=True
    )


def _project_source(
    anchor: pd.DataFrame,
    source: str,
    quality: pd.DataFrame,
    success: pd.DataFrame,
    cost: pd.DataFrame,
    selected: Mapping[str, str],
    statistics: pd.DataFrame,
) -> pd.DataFrame:
    source_rows = anchor.loc[anchor["source_dataset"].eq(source)].copy()
    source_rows = source_rows.set_index("sample_id", drop=False)
    output = source_rows.copy()
    selected_stats = statistics.set_index("model_name")
    low_cost = float(selected_stats.at[selected["low"], "mean_cost"])
    high_cost = float(selected_stats.at[selected["high"], "mean_cost"])
    span = high_cost - low_cost
    if span <= 0:
        raise ObjectiveDataError(
            f"Selected {source} arms do not have increasing train cost"
        )
    for tier in TIERS:
        model_name = selected[tier]
        output[f"model_{tier}"] = model_name
        output[f"quality_{tier}"] = pd.to_numeric(
            quality.loc[output.index, model_name], errors="coerce"
        )
        output[f"success_{tier}"] = success.loc[output.index, model_name].astype(
            "Float64"
        )
        output[f"actual_cost_{tier}"] = pd.to_numeric(
            cost.loc[output.index, model_name], errors="coerce"
        )
        decision_cost = (
            float(selected_stats.at[model_name, "mean_cost"]) - low_cost
        ) / span
        output[f"decision_cost_{tier}"] = float(decision_cost)
    required_values = [
        f"{prefix}_{tier}"
        for prefix in ("quality", "success", "actual_cost", "decision_cost")
        for tier in TIERS
    ]
    incomplete = output.loc[:, required_values].isna().any(axis=1)
    if incomplete.any():
        examples = output.loc[incomplete, "sample_id"].head(5).tolist()
        raise ObjectiveDataError(
            f"{source} selected-arm outcomes are incomplete for {int(incomplete.sum())} rows: {examples}"
        )
    values = output.loc[:, required_values].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ObjectiveDataError(f"{source} outcomes must be finite and non-negative")
    for tier in TIERS:
        if not output[f"quality_{tier}"].between(0.0, 1.0).all():
            raise ObjectiveDataError(f"{source} quality must be within [0, 1]")
        if not output[f"success_{tier}"].isin([0.0, 1.0]).all():
            raise ObjectiveDataError(f"{source} success must be observed binary values")
    output["oracle_quality_tier"] = (
        output.loc[:, [f"quality_{tier}" for tier in TIERS]]
        .to_numpy(float)
        .argmax(axis=1)
    )
    return output.reset_index(drop=True)


def build_outcome_dataset(
    config: Mapping[str, Any], *, project_root: Path, output_dir: Path
) -> dict[str, Any]:
    base_dataset_dir = _resolve(project_root, config["base_dataset_dir"])
    base_manifest_path = base_dataset_dir / "manifest.json"
    base_manifest = _load_json(base_manifest_path)
    if base_manifest.get("coverage_resolution_status") != "complete":
        raise ObjectiveDataError("Base comparison dataset coverage is not resolved")
    expected_manifest = config.get("base_dataset_manifest_sha256")
    if expected_manifest and _sha256(base_manifest_path) != expected_manifest:
        raise ObjectiveDataError("Base dataset manifest checksum mismatch")
    split_files = config.get("split_files", SPLIT_FILES)
    if not {"train", "validation"}.issubset(split_files):
        raise ObjectiveDataError(
            "Objective comparison requires train and validation splits"
        )
    anchor = _load_anchor_rows(
        base_dataset_dir,
        split_files=split_files,
        artifact_contracts=base_manifest.get("artifacts"),
    )
    t2_path = base_dataset_dir / "threshold" / "t2-all-splits.parquet"
    if t2_path.is_file():
        t2 = pd.read_parquet(t2_path, columns=["sample_id", "difficulty_label"])
        t2 = t2.rename(columns={"difficulty_label": "difficulty_label_t2"})
        anchor = anchor.merge(t2, on="sample_id", how="left", validate="one_to_one")
    else:
        anchor["difficulty_label_t2"] = pd.NA

    matrices = {
        "routerbench": _routerbench_matrices(config, project_root, anchor),
        "xroutebench": _xroute_matrices(config, project_root, anchor),
    }
    arm_config = config["arm_selection"]
    projected_frames: list[pd.DataFrame] = []
    arms_payload: dict[str, Any] = {
        "rule": arm_config["rule"],
        "selection_split": "train_only",
        "sources": {},
    }
    statistics_frames: list[pd.DataFrame] = []
    for source, (quality, success, cost) in matrices.items():
        train_ids = anchor.loc[
            anchor["source_dataset"].eq(source) & anchor["objective_split"].eq("train"),
            "sample_id",
        ].astype(str)
        selected, statistics = select_three_arms(
            quality,
            cost,
            train_ids,
            tolerance=float(arm_config["quality_tolerance"]),
            minimum_frontier_size=int(arm_config["minimum_frontier_size"]),
        )
        statistics.insert(0, "source_dataset", source)
        statistics_frames.append(statistics)
        projected_frames.append(
            _project_source(
                anchor,
                source,
                quality,
                success,
                cost,
                selected,
                statistics.drop(columns=["source_dataset"]),
            )
        )
        arms_payload["sources"][source] = {
            "selected_models": selected,
            "training_row_count": len(train_ids),
        }

    # Preserve the exact anchor row order so all methods share the same shuffle.
    projected = pd.concat(projected_frames, ignore_index=True).set_index(
        "sample_id", drop=False
    )
    projected = projected.loc[anchor["sample_id"].tolist()].reset_index(drop=True)
    if projected["sample_id"].duplicated().any() or len(projected) != len(anchor):
        raise ObjectiveDataError("Projected rows do not preserve the frozen sample set")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths: list[Path] = []
    for split in split_files:
        target = output_dir / f"{split}.parquet"
        projected.loc[projected["objective_split"].eq(split)].to_parquet(
            target, index=False
        )
        artifact_paths.append(target)
    statistics_path = output_dir / "candidate-statistics.parquet"
    pd.concat(statistics_frames, ignore_index=True).to_parquet(
        statistics_path, index=False
    )
    artifact_paths.append(statistics_path)
    arms_path = output_dir / "arms.json"
    write_json(arms_path, arms_payload)
    artifact_paths.append(arms_path)
    manifest = {
        "schema_version": "routing-objective-outcomes-v1",
        "experiment_version": config["experiment_version"],
        "dataset_version": config["dataset_version"],
        "created_at_unix": int(time.time()),
        "base_dataset_manifest_sha256": _sha256(base_manifest_path),
        "row_counts": {
            split: int(projected["objective_split"].eq(split).sum())
            for split in split_files
        },
        "source_row_counts": {
            source: int(projected["source_dataset"].eq(source).sum())
            for source in sorted(projected["source_dataset"].unique())
        },
        "routerarena_used_for_selection": False,
        "arms": arms_payload,
        "artifacts": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in artifact_paths
        },
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    (output_dir / "_SUCCESS").write_text(
        _sha256(manifest_path) + "\n", encoding="utf-8"
    )
    return manifest
