"""Independent sealed-test protocol; never a deployment or retraining gate.

The frozen training bundle is left untouched. Policies are locked using train
statistics and validation scores, inference is a separate explicit operation,
and evaluation uses those policies without selecting a test frontier or nAUC.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..shared.dataset_io import load_config, sha256_file, verify_file, write_json
from .data import TIERS
from .evaluation import route_actions

METHODS = ("ordinal", "pointwise", "pairwise")
TEST_PARTITIONS = ("test_routerbench", "test_xroutebench")
PROTOCOL_VERSION = "routing-sealed-protocol-v1"


class SealedProtocolError(ValueError):
    """An input, policy or comparison violated the sealed evaluation contract."""


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _child(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise SealedProtocolError(f"Artifact path leaves its directory: {relative}")
    return path


def _pin_artifacts(directory: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    records = {}
    for name, expected in manifest["artifacts"].items():
        path = _child(directory, name)
        verify_file(path, expected)
        records[name] = _record(path)
    return records


def _join_scores(outcomes: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if (
        outcomes["sample_id"].duplicated().any()
        or scores["sample_id"].duplicated().any()
    ):
        raise SealedProtocolError("Comparison sample IDs must be unique")
    if set(outcomes["sample_id"]) != set(scores["sample_id"]):
        raise SealedProtocolError(
            "Scores and frozen outcomes have different sample IDs"
        )
    score_columns = [f"score_{tier}" for tier in TIERS]
    aligned = scores.set_index("sample_id").loc[outcomes["sample_id"]]
    if not np.isfinite(aligned[score_columns].to_numpy(float)).all():
        raise SealedProtocolError("Non-finite model scores")
    for column in set(outcomes.columns) & set(aligned.columns):
        if (
            not aligned[column]
            .reset_index(drop=True)
            .equals(outcomes[column].reset_index(drop=True))
        ):
            raise SealedProtocolError(
                f"Scored metadata differs from frozen outcomes: {column}"
            )
    output = outcomes.reset_index(drop=True).copy()
    for column in score_columns:
        output[column] = aligned[column].to_numpy()
    return output


def _source_arrays(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices = tuple(
        rows[[f"{prefix}_{tier}" for tier in TIERS]].to_numpy(float)
        for prefix in ("quality", "actual_cost", "decision_cost")
    )
    if any(not np.isfinite(matrix).all() or (matrix < 0).any() for matrix in matrices):
        raise SealedProtocolError("Outcome values must be finite and non-negative")
    if (matrices[0] > 1).any():
        raise SealedProtocolError("Quality must lie within [0, 1]")
    return matrices


def _validate_candidate_rows(
    rows: pd.DataFrame, arms: Mapping[str, Any], statistics: pd.DataFrame
) -> None:
    if set(rows["source_dataset"]) != set(arms["sources"]):
        raise SealedProtocolError(
            "Evaluated sources differ from frozen candidate pools"
        )
    for source, group in rows.groupby("source_dataset", sort=True):
        selected = arms["sources"][source]["selected_models"]
        source_stats = statistics.loc[
            statistics["source_dataset"].eq(source)
        ].set_index("model_name")
        costs = [float(source_stats.at[selected[tier], "mean_cost"]) for tier in TIERS]
        if not np.isfinite(costs).all() or not 0 <= costs[0] < costs[1] < costs[2]:
            raise SealedProtocolError("Frozen candidate decision costs must increase")
        for index, tier in enumerate(TIERS):
            statistic = source_stats.loc[selected[tier]]
            if (
                not np.isfinite(float(statistic["mean_quality"]))
                or not 0 <= float(statistic["mean_quality"]) <= 1
                or statistic["selected_tier"] != tier
                or int(statistic["valid_rows"])
                != int(arms["sources"][source]["training_row_count"])
            ):
                raise SealedProtocolError(
                    "Selected-arm statistics lack complete train provenance"
                )
            if set(group[f"model_{tier}"]) != {selected[tier]}:
                raise SealedProtocolError(
                    "Outcome candidate IDs differ from frozen arms"
                )
            expected = (costs[index] - costs[0]) / (costs[2] - costs[0])
            if not np.allclose(
                group[f"decision_cost_{tier}"], expected, rtol=0, atol=1e-12
            ):
                raise SealedProtocolError(
                    "Decision costs differ from frozen train statistics"
                )
    _source_arrays(rows)


def create_sealed_protocol(
    *,
    validation_lock_path: Path,
    config_path: Path,
    train_outcome_dir: Path,
    projection_dir: Path,
    output_path: Path,
    run_dirs: Mapping[str, Path] | None = None,
    model_dirs: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Lock baselines and uncertainty settings without reading held-out rows."""
    seal_path = output_path.with_suffix(output_path.suffix + ".sha256")
    if output_path.exists() or seal_path.exists():
        raise SealedProtocolError("Refusing to overwrite an existing sealed protocol")
    lock = load_config(validation_lock_path)
    config = load_config(config_path)
    config_sha = sha256_file(config_path)
    if lock.get("stage") != "final" or lock.get("selection_data") != "validation_only":
        raise SealedProtocolError("A final validation-only selection lock is required")
    if lock.get("experiment_config_sha256") != config_sha:
        raise SealedProtocolError("Config differs from the final validation lock")
    if set(lock["methods"]) != set(METHODS) or set(config["methods"]) != set(METHODS):
        raise SealedProtocolError("Protocol requires the frozen R0/R1/R2 comparison")
    if lock["winner_method"] not in METHODS:
        raise SealedProtocolError("The validation winner is not a required method")
    train_manifest_path = train_outcome_dir / "manifest.json"
    train_manifest = load_config(train_manifest_path)
    train_sha = sha256_file(train_manifest_path)
    if (
        train_manifest["base_dataset_manifest_sha256"]
        != config["base_dataset_manifest_sha256"]
    ):
        raise SealedProtocolError("Train outcome and config base contracts differ")
    if (train_outcome_dir / "_SUCCESS").read_text().strip() != train_sha:
        raise SealedProtocolError("Train outcome completion marker is invalid")
    train_files = _pin_artifacts(train_outcome_dir, train_manifest)
    arms = load_config(train_outcome_dir / "arms.json")
    if arms != train_manifest["arms"] or arms["selection_split"] != "train_only":
        raise SealedProtocolError("Candidate arms lack frozen train-only provenance")
    statistics = pd.read_parquet(train_outcome_dir / "candidate-statistics.parquet")
    if statistics.duplicated(["source_dataset", "model_name"]).any():
        raise SealedProtocolError("Duplicate candidate statistics")
    projection_manifest_path = projection_dir / "manifest.json"
    projection = load_config(projection_manifest_path)
    if (projection_dir / "_SUCCESS").read_text().strip() != sha256_file(
        projection_manifest_path
    ):
        raise SealedProtocolError("Outcome projection completion marker is invalid")
    if (
        projection["train_outcome_manifest_sha256"] != train_sha
        or projection["base_dataset_manifest_sha256"]
        != config["base_dataset_manifest_sha256"]
        or projection["objective_config_sha256"] != config_sha
        or projection["arms_sha256"] != train_files["arms.json"]["sha256"]
        or projection["candidate_statistics_sha256"]
        != train_files["candidate-statistics.parquet"]["sha256"]
        or projection["coverage_status"] != "complete"
        or projection["dropped_row_count"] != 0
    ):
        raise SealedProtocolError(
            "Sealed outcome projection differs from frozen training contracts"
        )
    projection_files = _pin_artifacts(projection_dir, projection)
    inputs = {
        "validation_lock": _record(validation_lock_path),
        "config": _record(config_path),
        "train_manifest": _record(train_manifest_path),
        "projection_manifest": _record(projection_manifest_path),
        **{f"train/{name}": value for name, value in train_files.items()},
        **{f"projection/{name}": value for name, value in projection_files.items()},
    }
    best_single = {}
    for source, source_arms in sorted(arms["sources"].items()):
        selected = source_arms["selected_models"]
        source_stats = statistics.loc[
            statistics["source_dataset"].eq(source)
        ].set_index("model_name")
        tier = min(
            TIERS,
            key=lambda tier: (
                -float(source_stats.at[selected[tier], "mean_quality"]),
                float(source_stats.at[selected[tier], "mean_cost"]),
                selected[tier],
            ),
        )
        best_single[source] = {
            "tier": tier,
            "model_id": selected[tier],
            "selection_split": "train_only",
            "candidate_universe": "frozen_three_arms",
            "tie_breakers": [
                "train_mean_quality_desc",
                "train_mean_cost_asc",
                "model_id_asc",
            ],
        }
    validation = pd.read_parquet(train_outcome_dir / "validation.parquet")
    _validate_candidate_rows(validation, arms, statistics)
    methods = {}
    for method in METHODS:
        run_dir = Path(
            (run_dirs or {}).get(method, lock["methods"][method]["canonical_run_dir"])
        ).resolve()
        run_manifest_path = run_dir / "manifest.json"
        manifest = load_config(run_manifest_path)
        if (run_dir / "_SUCCESS").read_text().strip() != sha256_file(run_manifest_path):
            raise SealedProtocolError(f"Run completion marker is invalid: {method}")
        if (
            manifest["method"] != method
            or manifest["seed"] != lock["canonical_seed"]
            or manifest["experiment_config_sha256"] != config_sha
            or manifest["outcome_manifest_sha256"] != train_sha
            or manifest["validation_data_sha256"]
            != train_files["validation.parquet"]["sha256"]
            or manifest.get("test_data_accessed") is not False
            or manifest.get("routerarena_accessed") is not False
        ):
            raise SealedProtocolError(
                f"Run does not match validation-only frozen comparison: {method}"
            )
        run_files = _pin_artifacts(run_dir, manifest)
        metrics = load_config(run_dir / "metrics.json")
        policies = lock["methods"][method]["locked_budget_penalties"]
        if metrics["validation_budget_penalties"] != policies:
            raise SealedProtocolError(
                "Run policies differ from validation selection lock"
            )
        if (
            metrics["summary"][config["evaluation"]["selection_metric"]]
            != lock["methods"][method]["canonical_quality_cost_nauc"]
        ):
            raise SealedProtocolError(
                "Run selection score differs from validation lock"
            )
        model_dir = Path(
            (model_dirs or {}).get(method, manifest.get("model_dir", run_dir / "model"))
        ).resolve()
        model_files = {}
        if not {"config.json", "tokenizer.json", "tokenizer_config.json"}.issubset(
            manifest["model_artifacts"]
        ):
            raise SealedProtocolError(
                "Model manifest does not seal its tokenizer/config"
            )
        if not any(
            name.endswith((".safetensors", ".bin"))
            for name in manifest["model_artifacts"]
        ):
            raise SealedProtocolError("Model manifest does not seal its model weights")
        for name, expected in manifest["model_artifacts"].items():
            path = _child(model_dir, name)
            verify_file(path, expected)
            model_files[name] = _record(path)
        model_config = load_config(model_dir / "config.json")
        if model_config.get("router_objective", "ordinal") != method:
            raise SealedProtocolError(
                "Model objective differs from its selected policy"
            )
        scored = _join_scores(
            validation, pd.read_parquet(run_dir / "validation-scores.parquet")
        )
        if set(policies) != set(arms["sources"]):
            raise SealedProtocolError("Policies must be source-local")
        usage = {}
        expected_budgets = {
            f"{float(value):.4f}"
            for value in config["evaluation"]["relative_cost_budgets"]
        }
        for source, penalties in sorted(policies.items()):
            if set(penalties) != expected_budgets:
                raise SealedProtocolError(
                    "Locked budget grids differ across methods/sources"
                )
            rows = scored.loc[scored["source_dataset"].eq(source)]
            usage[source] = {}
            for budget, penalty in sorted(penalties.items()):
                actions = route_actions(
                    method,
                    rows[[f"score_{tier}" for tier in TIERS]].to_numpy(float),
                    rows[[f"decision_cost_{tier}" for tier in TIERS]].to_numpy(float),
                    float(penalty),
                )
                usage[source][budget] = {
                    tier: float((actions == index).mean())
                    for index, tier in enumerate(TIERS)
                }
        inputs[f"run/{method}/manifest"] = _record(run_manifest_path)
        inputs.update(
            {f"run/{method}/{name}": record for name, record in run_files.items()}
        )
        inputs.update(
            {f"model/{method}/{name}": record for name, record in model_files.items()}
        )
        methods[method] = {
            "model_dir": str(model_dir),
            "model_files": model_files,
            "locked_budget_penalties": policies,
            "validation_usage_ratios": usage,
        }
    protocol = {
        "schema_version": PROTOCOL_VERSION,
        "role": "sealed_test_research_only_not_deployment_gate",
        "winner_method": lock["winner_method"],
        "canonical_training_seed": lock["canonical_seed"],
        "training_seed_scope": lock.get("seed_scope", "canonical_only"),
        "selection_unchanged": True,
        "primary_relative_cost_budget": float(
            config["evaluation"]["default_relative_cost_budget"]
        ),
        "comparison_policy": {
            "primary": "R1_vs_R0_and_R2_vs_R0_at_default_validation_budget",
            "other_budgets_and_source_slices": "secondary_descriptive",
            "intervals": "pointwise_95_percent_not_simultaneous_familywise_intervals",
        },
        "config_path": str(config_path.resolve()),
        "arms": arms,
        "train_best_single": best_single,
        "methods": methods,
        "partitions": {
            name: projection_files[f"{name}.parquet"] for name in TEST_PARTITIONS
        },
        "expected_partition_rows": {
            name: projection["row_counts"][name] for name in TEST_PARTITIONS
        },
        "input_files": inputs,
        "bootstrap": {
            "iterations": 10000,
            "seed": 42,
            "confidence_level": 0.95,
            "unit": "duplicate_group_id",
            "strata": "source_and_group_label_signature",
            "reference_method": "ordinal",
            "uncertainty": "question_group_sampling_only",
        },
        "matched_random": {
            "seeds": list(range(100)),
            "ratio_source": "validation_locked_policy",
            "allocation": "largest_remainder_then_source_local_permutation",
            "remainder_tie_breaker": "model_id_asc",
            "rng_derivation": "SeedSequence([random_seed, first_8_sha256_source_bytes_as_big_endian_integer])",
            "distribution_is_confidence_interval": False,
        },
        "test_policy": "raw_frozen_policy_results_no_test_frontier_or_nauc",
        "oracle_tie_breakers": ["quality_desc", "actual_cost_asc", "model_id_asc"],
        "production_activation_authorized": False,
    }
    write_json(output_path, protocol)
    seal_path.write_text(sha256_file(output_path) + "\n")
    return protocol


def verify_protocol(path: Path) -> dict[str, Any]:
    if path.with_suffix(path.suffix + ".sha256").read_text().strip() != sha256_file(
        path
    ):
        raise SealedProtocolError("Sealed protocol checksum mismatch")
    protocol = load_config(path)
    if (
        protocol["schema_version"] != PROTOCOL_VERSION
        or protocol.get("production_activation_authorized") is not False
    ):
        raise SealedProtocolError("Not a supported research-only sealed protocol")
    for record in protocol["input_files"].values():
        verify_file(Path(record["path"]), record)
    return protocol


def infer_sealed_scores(
    *, protocol_path: Path, output_dir: Path, device: str = "auto"
) -> dict[str, Any]:
    """Explicit model inference, separate from protocol creation and evaluation."""
    protocol = verify_protocol(protocol_path)
    if output_dir.exists():
        raise SealedProtocolError("Inference output already exists")
    from .training import infer_scores

    records = {}
    for method, contract in protocol["methods"].items():
        for partition, data in protocol["partitions"].items():
            target = output_dir / method / f"{partition}-scores.parquet"
            result = infer_scores(
                model_dir=Path(contract["model_dir"]),
                data_path=Path(data["path"]),
                output_path=target,
                experiment_config_path=Path(protocol["config_path"]),
                device=device,
            )
            if result["row_count"] != protocol["expected_partition_rows"][partition]:
                raise SealedProtocolError("Inference changed the sealed row count")
            records[f"{method}/{partition}"] = _record(target)
    # Detect local input changes while inference was in progress, too.
    verify_protocol(protocol_path)
    manifest = {
        "schema_version": "routing-sealed-inference-v1",
        "protocol_sha256": sha256_file(protocol_path),
        "role": "sealed_test",
        "scores": records,
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "_SUCCESS").write_text(
        sha256_file(output_dir / "manifest.json") + "\n"
    )
    return manifest


def _outcome_rows(protocol: Mapping[str, Any]) -> pd.DataFrame:
    frames = []
    for partition, record in protocol["partitions"].items():
        frame = pd.read_parquet(record["path"])
        if len(frame) != protocol["expected_partition_rows"][partition] or set(
            frame["objective_split"]
        ) != {partition}:
            raise SealedProtocolError("Sealed outcome partition changed")
        frames.append(frame)
    rows = (
        pd.concat(frames, ignore_index=True)
        .sort_values("sample_id", kind="stable")
        .reset_index(drop=True)
    )
    required = {
        "sample_id",
        "source_dataset",
        "duplicate_group_id",
        "difficulty_label",
        "objective_split",
    }
    if required - set(rows) or rows.empty or rows["sample_id"].duplicated().any():
        raise SealedProtocolError(
            "Frozen evaluation metadata are incomplete or duplicated"
        )
    if rows[list(required)].isna().any().any():
        raise SealedProtocolError(
            "Frozen evaluation identifiers contain missing values"
        )
    if rows.groupby("duplicate_group_id")["objective_split"].nunique().gt(1).any():
        raise SealedProtocolError("A group crosses sealed partitions")
    statistics = pd.read_parquet(
        protocol["input_files"]["train/candidate-statistics.parquet"]["path"]
    )
    _validate_candidate_rows(rows, protocol["arms"], statistics)
    return rows


def _actions_frame(rows: pd.DataFrame, actions: np.ndarray) -> pd.DataFrame:
    quality, cost, _ = _source_arrays(rows)
    if actions.shape != (len(rows),) or not np.isin(actions, [0, 1, 2]).all():
        raise SealedProtocolError("Invalid routing action vector")
    columns = [
        column
        for column in (
            "sample_id",
            "source_dataset",
            "duplicate_group_id",
            "difficulty_label",
            "objective_split",
            "category",
        )
        if column in rows
    ]
    output = rows[columns].copy().reset_index(drop=True)
    indices = np.arange(len(rows))
    output["selected_tier"] = np.asarray(TIERS)[actions]
    output["selected_model_id"] = rows[[f"model_{tier}" for tier in TIERS]].to_numpy()[
        indices, actions
    ]
    output["selected_quality"] = quality[indices, actions]
    output["selected_actual_cost"] = cost[indices, actions]
    return output


def _policy_predictions(
    rows: pd.DataFrame,
    scored: pd.DataFrame,
    method: str,
    policies: Mapping[str, Mapping[str, float]],
    budget: str,
) -> pd.DataFrame:
    output_parts = []
    for source in sorted(policies):
        mask = rows["source_dataset"].eq(source).to_numpy()
        source_rows = rows.loc[mask].reset_index(drop=True)
        _, _, decision_cost = _source_arrays(source_rows)
        penalty = float(policies[source][budget])
        actions = route_actions(
            method,
            scored.loc[mask, [f"score_{tier}" for tier in TIERS]].to_numpy(float),
            decision_cost,
            penalty,
        )
        predictions = _actions_frame(source_rows, actions)
        predictions["penalty"] = penalty
        predictions["relative_cost_budget"] = float(budget)
        predictions["method"] = method
        output_parts.append(predictions)
    return (
        pd.concat(output_parts, ignore_index=True)
        .sort_values("sample_id", kind="stable")
        .reset_index(drop=True)
    )


def _summary(
    rows: pd.DataFrame, predictions: pd.DataFrame, *, budget: float | None = None
) -> dict[str, Any]:
    if (
        not rows["sample_id"]
        .reset_index(drop=True)
        .equals(predictions["sample_id"].reset_index(drop=True))
    ):
        raise SealedProtocolError("Summary predictions are not paired with outcomes")
    per_source = {}
    for source, source_rows in rows.groupby("source_dataset", sort=True):
        chosen = predictions.loc[predictions["source_dataset"].eq(source)]
        quality, cost, _ = _source_arrays(source_rows)
        mean_quality = float(chosen["selected_quality"].mean())
        mean_cost = float(chosen["selected_actual_cost"].mean())
        low_cost, high_cost = float(cost[:, 0].mean()), float(cost[:, 2].mean())
        low_quality, high_quality = (
            float(quality[:, 0].mean()),
            float(quality[:, 2].mean()),
        )
        cost_span, quality_span = high_cost - low_cost, high_quality - low_quality
        relative_cost = (mean_cost - low_cost) / cost_span if cost_span > 0 else None
        threshold = (
            low_cost + budget * cost_span
            if budget is not None and cost_span > 0
            else None
        )
        per_source[source] = {
            "row_count": len(source_rows),
            "group_count": int(source_rows["duplicate_group_id"].nunique()),
            "mean_quality": mean_quality,
            "mean_actual_cost": mean_cost,
            "always_low_quality": low_quality,
            "always_high_quality": high_quality,
            "always_low_cost": low_cost,
            "always_high_cost": high_cost,
            "relative_cost": relative_cost,
            "normalized_quality_gain_unclipped": (mean_quality - low_quality)
            / quality_span
            if quality_span > 0
            else None,
            "normalization_valid": {"cost": cost_span > 0, "quality": quality_span > 0},
            "relative_cost_budget": budget,
            "actual_cost_budget_threshold": threshold,
            "budget_exceeded": mean_cost > threshold + 1e-12
            if threshold is not None
            else None,
            "actual_cost_budget_excess": max(0.0, mean_cost - threshold)
            if threshold is not None
            else None,
            "usage_ratios": {
                tier: float(chosen["selected_tier"].eq(tier).mean()) for tier in TIERS
            },
        }
    normalized = [item["relative_cost"] for item in per_source.values()]
    return {
        "per_source": per_source,
        "source_macro_mean_quality": float(
            np.mean([item["mean_quality"] for item in per_source.values()])
        ),
        "source_macro_relative_cost": float(np.mean(normalized))
        if all(value is not None for value in normalized)
        else None,
        "row_weighted_mean_quality": float(predictions["selected_quality"].mean()),
        "row_weighted_mean_actual_cost": float(
            predictions["selected_actual_cost"].mean()
        ),
        "row_weighted_actual_cost_interpretation": "pooled_source_cost_not_comparable",
        "cost_semantics": "recorded_source_cost_units_not_current_service_prices",
    }


def _oracle_actions(rows: pd.DataFrame) -> np.ndarray:
    quality, cost, _ = _source_arrays(rows)
    model_ids = rows[[f"model_{tier}" for tier in TIERS]].to_numpy(str)
    return np.lexsort((model_ids, cost, -quality), axis=1)[:, 0]


def _matched_counts(
    probabilities: list[float], count: int, model_ids: list[str]
) -> np.ndarray:
    probabilities_array = np.asarray(probabilities, dtype=float)
    if (
        not np.isfinite(probabilities_array).all()
        or (probabilities_array < 0).any()
        or not np.isclose(probabilities_array.sum(), 1.0)
    ):
        raise SealedProtocolError("Frozen validation usage ratios must sum to one")
    exact = probabilities_array * count
    counts = np.floor(exact).astype(int)
    order = sorted(
        range(3), key=lambda index: (-(exact[index] - counts[index]), model_ids[index])
    )
    for index in order[: count - int(counts.sum())]:
        counts[index] += 1
    return counts


def _matched_random(
    rows: pd.DataFrame, protocol: Mapping[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    records, allocations = [], {}
    for method, contract in protocol["methods"].items():
        allocations[method] = {}
        budgets = sorted(next(iter(contract["validation_usage_ratios"].values())))
        for budget in budgets:
            allocations[method][budget] = {}
            per_source = {}
            for source, source_rows in rows.groupby("source_dataset", sort=True):
                source_rows = source_rows.sort_values("sample_id", kind="stable")
                quality, cost, _ = _source_arrays(source_rows)
                low_cost = float(cost[:, 0].mean())
                cost_span = float(cost[:, 2].mean()) - low_cost
                ratios = contract["validation_usage_ratios"][source][budget]
                selected = protocol["arms"]["sources"][source]["selected_models"]
                counts = _matched_counts(
                    [ratios[tier] for tier in TIERS],
                    len(source_rows),
                    [selected[tier] for tier in TIERS],
                )
                allocations[method][budget][source] = {
                    "validation_ratios": ratios,
                    "test_counts": dict(zip(TIERS, map(int, counts), strict=True)),
                    "row_count": len(source_rows),
                }
                per_source[source] = []
                base_actions = np.repeat(np.arange(3), counts)
                for seed in protocol["matched_random"]["seeds"]:
                    source_seed = int.from_bytes(
                        hashlib.sha256(source.encode()).digest()[:8], "big"
                    )
                    actions = np.random.default_rng(
                        np.random.SeedSequence([seed, source_seed])
                    ).permutation(base_actions)
                    mean_cost = float(cost[np.arange(len(actions)), actions].mean())
                    record = {
                        "method": method,
                        "relative_cost_budget": float(budget),
                        "source_dataset": source,
                        "seed": seed,
                        "mean_quality": float(
                            quality[np.arange(len(actions)), actions].mean()
                        ),
                        "mean_actual_cost": mean_cost,
                        "relative_cost": (mean_cost - low_cost) / cost_span
                        if cost_span > 0
                        else None,
                    }
                    per_source[source].append(record)
                    records.append(record)
            for index, seed in enumerate(protocol["matched_random"]["seeds"]):
                relative_costs = [
                    items[index]["relative_cost"] for items in per_source.values()
                ]
                records.append(
                    {
                        "method": method,
                        "relative_cost_budget": float(budget),
                        "source_dataset": "source_macro",
                        "seed": seed,
                        "mean_quality": float(
                            np.mean(
                                [
                                    items[index]["mean_quality"]
                                    for items in per_source.values()
                                ]
                            )
                        ),
                        "mean_actual_cost": None,
                        "relative_cost": float(np.mean(relative_costs))
                        if all(value is not None for value in relative_costs)
                        else None,
                    }
                )
    frame = pd.DataFrame(records)
    summaries = []
    for (method, budget, source), group in frame.groupby(
        ["method", "relative_cost_budget", "source_dataset"], sort=True
    ):
        summary = {
            "method": method,
            "relative_cost_budget": float(budget),
            "source_dataset": source,
            "seed_count": len(group),
        }
        for metric in ("mean_quality", "mean_actual_cost", "relative_cost"):
            values = group[metric].to_numpy(float)
            if not np.isfinite(values).all():
                summary[metric] = {
                    "available": False,
                    "reason": "incomparable_source_cost_units"
                    if source == "source_macro" and metric == "mean_actual_cost"
                    else "nonpositive_source_high_low_cost_span",
                }
                continue
            summary[metric] = {
                "available": True,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "p025": float(np.quantile(values, 0.025)),
                "p975": float(np.quantile(values, 0.975)),
            }
        summaries.append(summary)
    return {
        "allocations": allocations,
        "summaries": summaries,
        "distribution_is_confidence_interval": False,
        "source_macro_cost_metric": "source_relative_cost",
        "source_macro_actual_cost_unavailable_reason": "incomparable_source_cost_units",
        "cost_semantics": "recorded_source_cost_units_not_current_service_prices",
    }, frame


def paired_group_bootstrap(
    rows: pd.DataFrame,
    differences: np.ndarray,
    *,
    iterations: int = 10000,
    seed: int = 42,
    cost_column_indices: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Same stratified group draws for every paired method/budget/metric column.

    Groups with mixed labels use their complete sorted label signature. A group
    is never split. Source-spanning groups are rejected, since their source
    stratification would require a separate preregistered resampling design.
    Cost columns retain source-local raw units in per-source diagnostics, but
    source-macro cost deltas divide by each source's high-low cost span, computed
    again within the same group draw. No invalid normalization draws are dropped.
    """
    differences = np.asarray(differences, dtype=float)
    if (
        differences.ndim != 2
        or len(rows) != len(differences)
        or not np.isfinite(differences).all()
    ):
        raise SealedProtocolError("Invalid paired bootstrap differences")
    if iterations < 2 or rows.empty:
        raise SealedProtocolError("Bootstrap requires nonempty rows and multiple draws")
    if rows["duplicate_group_id"].isna().any() or rows["sample_id"].duplicated().any():
        raise SealedProtocolError("Invalid group or sample IDs")
    working = rows.reset_index(drop=True)
    if len(set(cost_column_indices)) != len(cost_column_indices) or any(
        index < 0 or index >= differences.shape[1] for index in cost_column_indices
    ):
        raise SealedProtocolError("Invalid bootstrap cost column indices")
    cost_spans = np.zeros(len(working))
    if cost_column_indices:
        cost_fields = ["actual_cost_low", "actual_cost_high"]
        if any(field not in working for field in cost_fields):
            raise SealedProtocolError("Cost bootstrap requires source cost references")
        references = working[cost_fields].to_numpy(float)
        if not np.isfinite(references).all() or (references < 0).any():
            raise SealedProtocolError("Invalid source cost references")
        cost_spans = references[:, 1] - references[:, 0]
    source_names = sorted(working["source_dataset"].unique())
    groups = []
    for group_id, group in working.groupby("duplicate_group_id", sort=True):
        sources = group["source_dataset"].unique()
        if len(sources) != 1:
            raise SealedProtocolError(
                "Source-spanning groups require a preregistered bootstrap design"
            )
        groups.append(
            {
                "id": group_id,
                "source": sources[0],
                "signature": tuple(
                    sorted(group["difficulty_label"].astype(str).unique())
                ),
                "size": len(group),
                "sum": differences[group.index].sum(axis=0),
                "cost_span_sum": float(cost_spans[group.index].sum()),
            }
        )
    strata: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = {}
    for group in groups:
        strata.setdefault((group["source"], group["signature"]), []).append(group)
    samples = {
        source: np.empty((iterations, differences.shape[1])) for source in source_names
    }
    weighted_samples = np.empty((iterations, differences.shape[1]))
    span_samples = {source: np.empty(iterations) for source in source_names}
    rng = np.random.default_rng(seed)
    for start in range(0, iterations, 128):
        size = min(128, iterations - start)
        numerators = {
            source: np.zeros((size, differences.shape[1])) for source in source_names
        }
        denominators = {source: np.zeros(size) for source in source_names}
        span_numerators = {source: np.zeros(size) for source in source_names}
        for (source, _), members in sorted(strata.items()):
            counts = rng.multinomial(
                len(members), np.full(len(members), 1.0 / len(members)), size=size
            )
            numerators[source] += counts @ np.stack(
                [member["sum"] for member in members]
            )
            denominators[source] += counts @ np.array(
                [member["size"] for member in members]
            )
            span_numerators[source] += counts @ np.array(
                [member["cost_span_sum"] for member in members]
            )
        for source in source_names:
            samples[source][start : start + size] = (
                numerators[source] / denominators[source][:, None]
            )
            span_samples[source][start : start + size] = (
                span_numerators[source] / denominators[source]
            )
        weighted_samples[start : start + size] = (
            sum(numerators.values()) / sum(denominators.values())[:, None]
        )

    def summary(values, estimate):
        return {
            "estimate": np.asarray(estimate).tolist(),
            "ci95_lower": np.quantile(values, 0.025, axis=0).tolist(),
            "ci95_upper": np.quantile(values, 0.975, axis=0).tolist(),
        }

    source_estimates = {
        source: differences[working["source_dataset"].eq(source).to_numpy()].mean(
            axis=0
        )
        for source in source_names
    }
    source_spans = {
        source: float(cost_spans[working["source_dataset"].eq(source)].mean())
        for source in source_names
    }
    # Only comparable quantities enter macro aggregation. Raw cost deltas remain
    # in source diagnostics; a shared scalar currency/unit is never assumed.
    macro = {"estimate": [], "ci95_lower": [], "ci95_upper": [], "columns": []}
    for index in range(differences.shape[1]):
        is_cost = index in cost_column_indices
        column_info = {
            "index": index,
            "metric_scale": "source_relative_cost_delta"
            if is_cost
            else "input_difference",
        }
        if is_cost:
            invalid_sources = [s for s in source_names if source_spans[s] <= 0]
            valid_draws = np.logical_and.reduce(
                [span_samples[s] > 0 for s in source_names]
            )
            estimate = (
                float(
                    np.mean(
                        [
                            source_estimates[s][index] / source_spans[s]
                            for s in source_names
                        ]
                    )
                )
                if not invalid_sources
                else None
            )
            column_info.update(
                {
                    "valid_draw_count": int(valid_draws.sum()),
                    "invalid_draw_count": int((~valid_draws).sum()),
                    "nonpositive_original_span_sources": invalid_sources,
                }
            )
            if invalid_sources or not valid_draws.all():
                column_info["unavailable_reason"] = (
                    "nonpositive_source_high_low_cost_span"
                    if invalid_sources
                    else "nonpositive_source_high_low_cost_span_in_bootstrap_draw"
                )
                lower, upper = None, None
            else:
                values = np.mean(
                    [samples[s][:, index] / span_samples[s] for s in source_names],
                    axis=0,
                )
                lower, upper = map(float, np.quantile(values, [0.025, 0.975]))
        else:
            estimate = float(np.mean([v[index] for v in source_estimates.values()]))
            values = np.mean([v[:, index] for v in samples.values()], axis=0)
            lower, upper = map(float, np.quantile(values, [0.025, 0.975]))
        macro["estimate"].append(estimate)
        macro["ci95_lower"].append(lower)
        macro["ci95_upper"].append(upper)
        macro["columns"].append(column_info)
    pooled = summary(weighted_samples, differences.mean(axis=0))
    pooled["cost_columns_interpretation"] = "pooled_source_cost_not_comparable"
    pooled["cost_column_indices"] = list(cost_column_indices)
    return {
        "iterations": iterations,
        "seed": seed,
        "confidence_level": 0.95,
        "unit": "duplicate_group_id",
        "strata": "source_and_group_label_signature",
        "uncertainty": "question_group_sampling_only_not_training_seed_variation",
        "group_count": len(groups),
        "row_count": len(rows),
        "stratum_support": [
            {
                "source_dataset": source,
                "label_signature": list(signature),
                "group_count": len(members),
                "row_count": sum(member["size"] for member in members),
            }
            for (source, signature), members in sorted(strata.items())
        ],
        "per_source": {
            source: summary(samples[source], source_estimates[source])
            for source in source_names
        },
        "source_macro": macro,
        "row_weighted": pooled,
        "cost_normalization": {
            "cost_column_indices": list(cost_column_indices),
            "per_source_scale": "recorded_source_cost_unit_delta",
            "source_macro_scale": "source_relative_cost_delta",
            "source_macro_raw_cost_unavailable_reason": "incomparable_source_cost_units",
            "denominator": "same_source_mean_high_minus_mean_low_actual_cost",
            "denominator_resampled_with_groups": True,
            "original_source_spans": source_spans if cost_column_indices else {},
        },
        "interpretation": "CI containing zero does not establish superiority; random-seed and training-seed variation are separate",
    }


def evaluate_sealed_scores(
    *, protocol_path: Path, inference_dir: Path, output_dir: Path
) -> dict[str, Any]:
    """Evaluate previously sealed scores; never loads a model or chooses a policy."""
    protocol = verify_protocol(protocol_path)
    if output_dir.exists():
        raise SealedProtocolError("Evaluation output already exists")
    inference_manifest_path = inference_dir / "manifest.json"
    inference = load_config(inference_manifest_path)
    if inference["schema_version"] != "routing-sealed-inference-v1" or inference[
        "protocol_sha256"
    ] != sha256_file(protocol_path):
        raise SealedProtocolError("Inference manifest differs from sealed protocol")
    if (inference_dir / "_SUCCESS").read_text().strip() != sha256_file(
        inference_manifest_path
    ):
        raise SealedProtocolError("Inference completion marker is invalid")
    expected_scores = {
        f"{method}/{partition}" for method in METHODS for partition in TEST_PARTITIONS
    }
    if set(inference["scores"]) != expected_scores:
        raise SealedProtocolError(
            "Inference did not cover every method and sealed partition"
        )
    for record in inference["scores"].values():
        verify_file(Path(record["path"]), record)
    rows = _outcome_rows(protocol)
    methods, predictions = {}, {}
    for method in METHODS:
        parts = [
            pd.read_parquet(inference["scores"][f"{method}/{partition}"]["path"])
            for partition in TEST_PARTITIONS
        ]
        scored = _join_scores(rows, pd.concat(parts, ignore_index=True))
        policies = protocol["methods"][method]["locked_budget_penalties"]
        predictions[method], methods[method] = {}, {}
        for budget in sorted(next(iter(policies.values()))):
            chosen = _policy_predictions(rows, scored, method, policies, budget)
            predictions[method][budget] = chosen
            methods[method][budget] = _summary(rows, chosen, budget=float(budget))
    baselines = {
        f"always_{tier}": _summary(
            rows, _actions_frame(rows, np.full(len(rows), index, dtype=int))
        )
        for index, tier in enumerate(TIERS)
    }
    best_actions = np.array(
        [
            TIERS.index(protocol["train_best_single"][source]["tier"])
            for source in rows["source_dataset"]
        ]
    )
    baselines["train_best_single"] = {
        "selection": protocol["train_best_single"],
        **_summary(rows, _actions_frame(rows, best_actions)),
    }
    baselines["quality_oracle"] = {
        "deployable": False,
        "tie_breakers": protocol["oracle_tie_breakers"],
        **_summary(rows, _actions_frame(rows, _oracle_actions(rows))),
    }
    random_summary, random_runs = _matched_random(rows, protocol)
    column_specs, delta_columns = [], []
    for method in METHODS[1:]:
        for budget, candidate in predictions[method].items():
            reference = predictions["ordinal"][budget]
            for metric in ("selected_quality", "selected_actual_cost"):
                column_specs.append(
                    {
                        "candidate": method,
                        "reference": "ordinal",
                        "relative_cost_budget": float(budget),
                        "metric": metric,
                        "source_macro_metric": "source_relative_cost_delta"
                        if metric == "selected_actual_cost"
                        else "selected_quality_delta",
                    }
                )
                delta_columns.append(
                    candidate[metric].to_numpy(float)
                    - reference[metric].to_numpy(float)
                )
    bootstrap = paired_group_bootstrap(
        rows,
        np.stack(delta_columns, axis=1),
        iterations=protocol["bootstrap"]["iterations"],
        seed=protocol["bootstrap"]["seed"],
        cost_column_indices=tuple(
            index
            for index, spec in enumerate(column_specs)
            if spec["metric"] == "selected_actual_cost"
        ),
    )
    bootstrap["column_order"] = column_specs
    result = {
        "schema_version": "routing-sealed-evaluation-v1",
        "role": protocol["role"],
        "protocol_sha256": sha256_file(protocol_path),
        "winner_method_unchanged": protocol["winner_method"],
        "canonical_training_seed": protocol["canonical_training_seed"],
        "training_seed_scope": protocol["training_seed_scope"],
        "primary_relative_cost_budget": protocol["primary_relative_cost_budget"],
        "comparison_policy": protocol["comparison_policy"],
        "test_policy": protocol["test_policy"],
        "row_count": len(rows),
        "methods": methods,
        "baselines": baselines,
        "matched_random": random_summary,
        "paired_group_bootstrap_vs_ordinal": bootstrap,
        "production_activation_authorized": False,
    }
    output_dir.mkdir(parents=True)
    prediction_path = output_dir / "predictions.parquet"
    pd.concat(
        [frame for method in predictions.values() for frame in method.values()],
        ignore_index=True,
    ).to_parquet(prediction_path, index=False)
    random_path = output_dir / "matched-random-runs.parquet"
    random_runs.to_parquet(random_path, index=False)
    write_json(output_dir / "summary.json", result)
    manifest = {
        "schema_version": "routing-sealed-evaluation-artifacts-v1",
        "protocol_sha256": sha256_file(protocol_path),
        "inference_manifest_sha256": sha256_file(inference_manifest_path),
        "artifacts": {
            path.name: _record(path)
            for path in (prediction_path, random_path, output_dir / "summary.json")
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "_SUCCESS").write_text(
        sha256_file(output_dir / "manifest.json") + "\n"
    )
    return result


def _path_mapping(values: list[str] | None) -> dict[str, Path]:
    output = {}
    for value in values or []:
        method, separator, path = value.partition("=")
        if not separator or method not in METHODS or method in output or not path:
            raise SealedProtocolError(
                "Path overrides must be unique METHOD=PATH entries"
            )
        output[method] = Path(path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    lock = commands.add_parser(
        "lock", help="Seal train/validation policies before held-out inference"
    )
    for name in (
        "validation-lock",
        "config",
        "train-outcome-dir",
        "projection-dir",
        "output",
    ):
        lock.add_argument(f"--{name}", type=Path, required=True)
    lock.add_argument(
        "--run-dir",
        action="append",
        help="Optional METHOD=PATH override for downloaded runs",
    )
    lock.add_argument(
        "--model-dir",
        action="append",
        help="Optional METHOD=PATH override for downloaded checkpoints",
    )
    infer = commands.add_parser(
        "infer", help="Explicitly run model inference on the two sealed public tests"
    )
    infer.add_argument("--protocol", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    evaluate = commands.add_parser(
        "evaluate", help="Evaluate sealed scores using fixed policies only"
    )
    evaluate.add_argument("--protocol", type=Path, required=True)
    evaluate.add_argument("--inference-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "lock":
        create_sealed_protocol(
            validation_lock_path=args.validation_lock,
            config_path=args.config,
            train_outcome_dir=args.train_outcome_dir,
            projection_dir=args.projection_dir,
            output_path=args.output,
            run_dirs=_path_mapping(args.run_dir),
            model_dirs=_path_mapping(args.model_dir),
        )
        print(
            f"Research-only protocol sealed: {args.output}; no held-out inference performed"
        )
    elif args.command == "infer":
        infer_sealed_scores(
            protocol_path=args.protocol, output_dir=args.output_dir, device=args.device
        )
        print(
            f"Sealed inference completed: {args.output_dir}; no policy selection performed"
        )
    else:
        evaluate_sealed_scores(
            protocol_path=args.protocol,
            inference_dir=args.inference_dir,
            output_dir=args.output_dir,
        )
        print(
            f"Frozen-policy research evaluation completed: {args.output_dir}; not a deployment gate"
        )


if __name__ == "__main__":
    main()
