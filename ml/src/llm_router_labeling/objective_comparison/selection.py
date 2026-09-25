from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from ..shared.dataset_io import sha256_file, write_json
from .data import TIERS
from .evaluation import predictions_frame, route_actions
from .objectives import METHODS
from .training import infer_scores


class ObjectiveSelectionError(ValueError):
    """Raised when validation locking or sealed evaluation is invalid."""


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ObjectiveSelectionError(f"Expected JSON object: {path}")
    return value


def discover_runs(
    roots: Iterable[Path], *, selection_metric: str = "quality_cost_nauc"
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for root in roots:
        candidates = (
            [root / "manifest.json"]
            if (root / "manifest.json").is_file()
            else list(root.rglob("manifest.json"))
        )
        for manifest_path in sorted(candidates):
            run_dir = manifest_path.parent
            if (
                not (run_dir / "_SUCCESS").is_file()
                or not (run_dir / "metrics.json").is_file()
            ):
                continue
            manifest = _load_json(manifest_path)
            if manifest.get("schema_version") != "routing-objective-run-v1":
                continue
            if any(
                int(manifest.get(key, 0)) > 0
                for key in (
                    "maximum_train_rows",
                    "maximum_validation_rows",
                    "maximum_train_steps",
                )
            ):
                continue
            method = str(manifest["method"])
            seed = int(manifest["seed"])
            key = (method, seed)
            if key in seen:
                raise ObjectiveSelectionError(
                    f"Duplicate complete run for {method} seed {seed}"
                )
            seen.add(key)
            metrics = _load_json(run_dir / "metrics.json")
            metric_value = metrics["summary"].get(selection_metric)
            if metric_value is None or not np.isfinite(metric_value):
                raise ObjectiveSelectionError(
                    f"Selection metric is unavailable for {method} seed {seed}"
                )
            records.append(
                {
                    "method": method,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "quality_cost_nauc": float(metric_value),
                    "budget_penalties": metrics["validation_budget_penalties"],
                    "summary": metrics["summary"],
                    "manifest": manifest,
                    "validation_mean_actual_cost": metrics.get(
                        "validation_mean_actual_cost", float("inf")
                    ),
                    "batch_1_p95_latency_ms": metrics.get("batch_1_latency_ms", {}).get(
                        "p95", float("inf")
                    ),
                    "optimizer_steps": metrics.get("optimizer_steps"),
                    "planned_steps": metrics.get("training_metrics", {})
                    .get("optimization", {})
                    .get("planned_steps"),
                }
            )
    return records


def build_validation_lock(
    *,
    run_roots: Sequence[Path],
    experiment_config_path: Path,
    output_path: Path,
    stage: str,
) -> dict[str, Any]:
    if stage not in {"provisional", "final"}:
        raise ObjectiveSelectionError("stage must be provisional or final")
    config = _load_json(experiment_config_path)
    methods = tuple(config.get("methods", METHODS))
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ObjectiveSelectionError("Invalid configured objective methods")
    metric = config["evaluation"]["selection_metric"]
    records = [
        record
        for record in discover_runs(run_roots, selection_metric=metric)
        if record["method"] in methods
    ]
    if any(not np.isfinite(record["quality_cost_nauc"]) for record in records):
        raise ObjectiveSelectionError("Selection metric is non-finite")
    canonical_seed = int(config["seed"])
    canonical = {
        record["method"]: record
        for record in records
        if record["seed"] == canonical_seed
    }
    missing = sorted(set(methods) - set(canonical))
    if missing:
        raise ObjectiveSelectionError(f"Missing canonical full runs: {missing}")
    if config.get("selection", {}).get("canonical_only") and any(
        record["seed"] != canonical_seed for record in records
    ):
        raise ObjectiveSelectionError(
            "Canonical-only comparison contains extra seed runs"
        )
    if config.get("selection", {}).get("require_contract_match"):
        manifest_hash = sha256_file(experiment_config_path)
        reference = canonical[methods[0]]["manifest"]
        for record in records:
            manifest = record["manifest"]
            if (
                manifest.get("experiment_version") != config["experiment_version"]
                or manifest.get("experiment_config_sha256") != manifest_hash
                or manifest.get("base_training_sha256")
                != config["base_training_sha256"]
            ):
                raise ObjectiveSelectionError(
                    "Run does not match the frozen comparison contract"
                )
            for field in (
                "outcome_manifest_sha256",
                "training_data_sha256",
                "validation_data_sha256",
                "common_training_config",
                "runtime_version",
                "runtime_contract_sha256",
            ):
                if not reference.get(field) or manifest.get(field) != reference[field]:
                    raise ObjectiveSelectionError(f"Comparison runs differ: {field}")
            if (
                record["optimizer_steps"] is None
                or record["optimizer_steps"] != record["planned_steps"]
            ):
                raise ObjectiveSelectionError(
                    "Canonical run did not complete the common step budget"
                )
            if (
                manifest.get("test_data_accessed") is not False
                or manifest.get("routerarena_accessed") is not False
            ):
                raise ObjectiveSelectionError(
                    "Selection run accessed a sealed evaluation set"
                )
            run_dir = Path(record["run_dir"])
            if (run_dir / "_SUCCESS").read_text().strip() != sha256_file(
                run_dir / "manifest.json"
            ):
                raise ObjectiveSelectionError(
                    "Run success marker does not seal its manifest"
                )
            for name in (
                "metrics.json",
                "validation-scores.parquet",
                "validation-curve.parquet",
            ):
                expected = manifest.get("artifacts", {}).get(name, {}).get("sha256")
                if not expected or sha256_file(run_dir / name) != expected:
                    raise ObjectiveSelectionError(
                        f"Run artifact checksum mismatch: {name}"
                    )
            model_artifacts = manifest.get("model_artifacts", {})
            model_dir = Path(manifest.get("model_dir", run_dir / "model"))
            if not model_artifacts:
                raise ObjectiveSelectionError("Run has no sealed model artifacts")
            for name, artifact in model_artifacts.items():
                path = model_dir / name
                if (
                    not path.is_file()
                    or path.stat().st_size != artifact["size_bytes"]
                    or sha256_file(path) != artifact["sha256"]
                ):
                    raise ObjectiveSelectionError(
                        f"Model artifact checksum mismatch: {name}"
                    )
    provisional_ranking = sorted(
        canonical.values(),
        key=lambda record: (
            -record["quality_cost_nauc"],
            record["validation_mean_actual_cost"],
            record["batch_1_p95_latency_ms"],
            record["method"],
        ),
    )
    finalists = [record["method"] for record in provisional_ranking[:2]]
    expected_finalist_seeds = [canonical_seed, *map(int, config["finalist_seeds"])]
    method_statistics: dict[str, Any] = {}
    for method in methods:
        method_records = sorted(
            [
                record
                for record in records
                if record["method"] == method
                and (stage == "final" or record["seed"] == canonical_seed)
            ],
            key=lambda record: record["seed"],
        )
        if stage == "final" and method in finalists:
            present = {record["seed"] for record in method_records}
            missing_seeds = sorted(set(expected_finalist_seeds) - present)
            if missing_seeds:
                raise ObjectiveSelectionError(
                    f"Finalist {method} is missing full seeds: {missing_seeds}"
                )
        values = np.array([record["quality_cost_nauc"] for record in method_records])
        method_statistics[method] = {
            "seeds": [record["seed"] for record in method_records],
            "quality_cost_nauc_mean": float(values.mean()),
            "quality_cost_nauc_std": float(values.std(ddof=1))
            if len(values) > 1
            else 0.0,
            "canonical_quality_cost_nauc": canonical[method]["quality_cost_nauc"],
        }
    if stage == "provisional" or not config["finalist_seeds"]:
        winner = provisional_ranking[0]["method"]
    else:
        winner = min(
            finalists,
            key=lambda method: (
                -method_statistics[method]["quality_cost_nauc_mean"],
                method_statistics[method]["quality_cost_nauc_std"],
                method,
            ),
        )
    validation_outcomes = pd.read_parquet(
        Path(canonical["ordinal"]["run_dir"]) / "validation-scores.parquet"
    )
    tier_quality = {
        tier: float(validation_outcomes[f"quality_{tier}"].mean()) for tier in TIERS
    }
    best_single_tier = min(
        TIERS, key=lambda tier: (-tier_quality[tier], TIERS.index(tier))
    )
    lock = {
        "schema_version": "routing-objective-validation-lock-v1",
        "experiment_version": config["experiment_version"],
        "stage": stage,
        "selection_data": "validation_only",
        "test_data_accessed": False,
        "routerarena_accessed": False,
        "selection_metric": config["evaluation"]["selection_metric"],
        "canonical_seed": canonical_seed,
        "finalists": finalists,
        "winner_method": winner,
        "validation_best_single_tier": best_single_tier,
        "validation_fixed_tier_quality": tier_quality,
        "default_relative_cost_budget": float(
            config["evaluation"]["default_relative_cost_budget"]
        ),
        "methods": {
            method: {
                **method_statistics[method],
                "canonical_run_dir": canonical[method]["run_dir"],
                "locked_budget_penalties": canonical[method]["budget_penalties"],
            }
            for method in methods
        },
        "experiment_config_sha256": sha256_file(experiment_config_path),
        "required_methods": list(methods),
        "seed_scope": "canonical_only"
        if not config["finalist_seeds"]
        else "top_two_repeated",
        "production_activation_authorized": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        existing = _load_json(output_path)
        if existing != lock:
            raise ObjectiveSelectionError(
                f"Refusing to overwrite a different validation lock: {output_path}"
            )
        return existing
    write_json(output_path, lock)
    return lock


def _scores(scored: pd.DataFrame) -> np.ndarray:
    return scored.loc[:, [f"score_{tier}" for tier in TIERS]].to_numpy(float)


def _locked_budget_metrics(
    scored: pd.DataFrame,
    *,
    method: str,
    penalties: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    if penalties and all(isinstance(value, Mapping) for value in penalties.values()):
        sources = set(scored["source_dataset"].astype(str))
        if sources != set(penalties):
            raise ObjectiveSelectionError(
                "Locked policies and evaluated sources differ"
            )
        parts, per_source = [], {}
        for source, locked in sorted(penalties.items()):
            rows = scored.loc[scored["source_dataset"].eq(source)]
            source_metrics, source_predictions = _locked_budget_metrics(
                rows, method=method, penalties=locked
            )
            per_source[source] = source_metrics
            parts.append(source_predictions)
        predictions = pd.concat(parts, ignore_index=True)
        budgets = {tuple(sorted(metrics)) for metrics in per_source.values()}
        if len(budgets) != 1:
            raise ObjectiveSelectionError("Source policies have different budget grids")
        combined = {}
        for budget in next(iter(budgets)):
            rows = predictions.loc[
                predictions["relative_cost_budget"].eq(float(budget))
            ]
            combined[budget] = {
                "mean_quality": float(rows["selected_quality"].mean()),
                "mean_actual_cost": float(rows["selected_actual_cost"].mean()),
                "source_macro_quality": float(
                    np.mean(
                        [
                            metrics[budget]["mean_quality"]
                            for metrics in per_source.values()
                        ]
                    )
                ),
                "per_source": {
                    source: metrics[budget] for source, metrics in per_source.items()
                },
            }
        return combined, predictions
    results: dict[str, Any] = {}
    predictions: list[pd.DataFrame] = []
    for budget, penalty_value in sorted(
        penalties.items(), key=lambda item: float(item[0])
    ):
        penalty = float(penalty_value)
        output = predictions_frame(
            scored, _scores(scored), method=method, penalty=penalty
        )
        output["relative_cost_budget"] = float(budget)
        results[budget] = {
            "penalty": penalty,
            "mean_quality": float(output["selected_quality"].mean()),
            "mean_actual_cost": float(output["selected_actual_cost"].mean()),
            "low_rate": float(output["selected_tier"].eq("low").mean()),
            "medium_rate": float(output["selected_tier"].eq("medium").mean()),
            "high_rate": float(output["selected_tier"].eq("high").mean()),
        }
        predictions.append(output)
    return results, pd.concat(predictions, ignore_index=True)


def evaluate_sealed_tests(
    *,
    validation_lock_path: Path,
    run_roots: Sequence[Path],
    routerbench_test_path: Path,
    xroutebench_test_path: Path,
    experiment_config_path: Path,
    output_dir: Path,
    device: str = "auto",
) -> dict[str, Any]:
    lock = _load_json(validation_lock_path)
    config = _load_json(experiment_config_path)
    if lock.get("stage") != "final":
        raise ObjectiveSelectionError("Sealed tests require a final validation lock")
    if lock.get("experiment_config_sha256") != sha256_file(experiment_config_path):
        raise ObjectiveSelectionError("Evaluation config differs from validation lock")
    methods = tuple(config.get("methods", METHODS))
    if set(lock["methods"]) != set(methods):
        raise ObjectiveSelectionError("Evaluation methods differ from validation lock")
    runs = discover_runs(
        run_roots, selection_metric=config["evaluation"]["selection_metric"]
    )
    lookup = {(record["method"], record["seed"]): record for record in runs}
    canonical_seed = int(lock["canonical_seed"])
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}
    default_predictions: dict[str, pd.DataFrame] = {}
    test_paths = [routerbench_test_path, xroutebench_test_path]
    for method in methods:
        record = lookup.get((method, canonical_seed))
        if record is None:
            raise ObjectiveSelectionError(
                f"Missing canonical run for sealed test: {method}"
            )
        scored_parts: list[pd.DataFrame] = []
        for test_path in test_paths:
            target = output_dir / method / f"{test_path.stem}-scores.parquet"
            infer_scores(
                model_dir=Path(
                    record["manifest"].get(
                        "model_dir", Path(record["run_dir"]) / "model"
                    )
                ),
                data_path=test_path,
                output_path=target,
                experiment_config_path=experiment_config_path,
                device=device,
            )
            scored_parts.append(pd.read_parquet(target))
        scored = pd.concat(scored_parts, ignore_index=True)
        metrics, predictions = _locked_budget_metrics(
            scored,
            method=method,
            penalties=lock["methods"][method]["locked_budget_penalties"],
        )
        method_dir = output_dir / method
        predictions.to_parquet(method_dir / "test-predictions.parquet", index=False)
        write_json(method_dir / "test-metrics.json", metrics)
        all_results[method] = metrics
        default_budget = float(lock["default_relative_cost_budget"])
        default_predictions[method] = (
            predictions.loc[predictions["relative_cost_budget"].eq(default_budget)]
            .sort_values("sample_id", kind="stable")
            .reset_index(drop=True)
        )

    outcome = pd.concat(
        [pd.read_parquet(path) for path in test_paths], ignore_index=True
    )
    quality = outcome.loc[:, [f"quality_{tier}" for tier in TIERS]].to_numpy(float)
    cost = outcome.loc[:, [f"actual_cost_{tier}" for tier in TIERS]].to_numpy(float)
    baselines = {
        tier: {
            "mean_quality": float(quality[:, index].mean()),
            "mean_actual_cost": float(cost[:, index].mean()),
        }
        for index, tier in enumerate(TIERS)
    }
    oracle_action = quality.argmax(axis=1)
    baselines["quality_oracle"] = {
        "mean_quality": float(quality.max(axis=1).mean()),
        "mean_actual_cost": float(cost[np.arange(len(cost)), oracle_action].mean()),
    }
    best_single_tier = str(lock["validation_best_single_tier"])
    best_single_index = TIERS.index(best_single_tier)
    baselines["validation_best_single"] = {
        "tier": best_single_tier,
        "mean_quality": float(quality[:, best_single_index].mean()),
        "mean_actual_cost": float(cost[:, best_single_index].mean()),
    }
    random_generator = np.random.default_rng(int(config["seed"]))
    random_actions = random_generator.integers(0, len(TIERS), size=len(outcome))
    baselines["random"] = {
        "seed": int(config["seed"]),
        "mean_quality": float(quality[np.arange(len(quality)), random_actions].mean()),
        "mean_actual_cost": float(cost[np.arange(len(cost)), random_actions].mean()),
    }
    reference = default_predictions["ordinal"]
    bootstrap_iterations = int(config["evaluation"]["bootstrap_iterations"])
    bootstrap_rng = np.random.default_rng(int(config["seed"]))
    paired_bootstrap: dict[str, Any] = {}
    for method in methods:
        candidate = default_predictions[method]
        if not candidate["sample_id"].equals(reference["sample_id"]):
            raise ObjectiveSelectionError(
                f"Default-budget predictions are not paired for {method}"
            )
        quality_delta = candidate["selected_quality"].to_numpy(float) - reference[
            "selected_quality"
        ].to_numpy(float)
        cost_delta = candidate["selected_actual_cost"].to_numpy(float) - reference[
            "selected_actual_cost"
        ].to_numpy(float)
        quality_samples = np.empty(bootstrap_iterations, dtype=float)
        cost_samples = np.empty(bootstrap_iterations, dtype=float)
        for iteration in range(bootstrap_iterations):
            indices = bootstrap_rng.integers(0, len(reference), size=len(reference))
            quality_samples[iteration] = quality_delta[indices].mean()
            cost_samples[iteration] = cost_delta[indices].mean()
        paired_bootstrap[method] = {
            "reference": "ordinal",
            "relative_cost_budget": float(lock["default_relative_cost_budget"]),
            "mean_quality_delta": float(quality_delta.mean()),
            "quality_delta_ci95": [
                float(np.quantile(quality_samples, 0.025)),
                float(np.quantile(quality_samples, 0.975)),
            ],
            "mean_actual_cost_delta": float(cost_delta.mean()),
            "actual_cost_delta_ci95": [
                float(np.quantile(cost_samples, 0.025)),
                float(np.quantile(cost_samples, 0.975)),
            ],
        }
    result = {
        "schema_version": "routing-objective-sealed-test-v1",
        "validation_lock": str(validation_lock_path),
        "canonical_seed": canonical_seed,
        "routerarena_used": False,
        "methods": all_results,
        "baselines": baselines,
        "paired_bootstrap_vs_ordinal": paired_bootstrap,
    }
    write_json(output_dir / "summary.json", result)
    return result


def evaluate_routerarena_alignment(
    *,
    validation_lock_path: Path,
    run_roots: Sequence[Path],
    routerarena_path: Path,
    experiment_config_path: Path,
    output_dir: Path,
    device: str = "auto",
) -> dict[str, Any]:
    lock = _load_json(validation_lock_path)
    config = _load_json(experiment_config_path)
    if lock.get("stage") != "final":
        raise ObjectiveSelectionError(
            "RouterArena evaluation requires a final validation lock"
        )
    if any(
        isinstance(value, Mapping)
        for method in lock["methods"].values()
        for value in method["locked_budget_penalties"].values()
    ):
        raise ObjectiveSelectionError(
            "Source-local policies need an explicit RouterArena candidate mapping; "
            "R1/R2 action scores are not difficulty probabilities"
        )
    records = discover_runs(
        run_roots, selection_metric=config["evaluation"]["selection_metric"]
    )
    lookup = {(record["method"], record["seed"]): record for record in records}
    seed = int(lock["canonical_seed"])
    arena = pd.read_parquet(routerarena_path)
    labels = (
        arena["difficulty_label"].map({"easy": 0, "medium": 1, "hard": 2}).to_numpy(int)
    )
    abstract_cost = np.tile(np.array([[0.0, 0.5, 1.0]], dtype=float), (len(arena), 1))
    default_budget = f"{float(lock['default_relative_cost_budget']):.4f}"
    results: dict[str, Any] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for method in lock["methods"]:
        record = lookup.get((method, seed))
        if record is None:
            raise ObjectiveSelectionError(
                f"Missing canonical run for RouterArena: {method}"
            )
        score_path = output_dir / method / "routerarena-scores.parquet"
        infer_scores(
            model_dir=Path(
                record["manifest"].get("model_dir", Path(record["run_dir"]) / "model")
            ),
            data_path=routerarena_path,
            output_path=score_path,
            experiment_config_path=experiment_config_path,
            device=device,
        )
        scored = pd.read_parquet(score_path)
        penalty = float(
            lock["methods"][method]["locked_budget_penalties"][default_budget]
        )
        actions = route_actions(method, _scores(scored), abstract_cost, penalty)
        predictions = scored.copy()
        predictions["predicted_tier"] = [TIERS[index] for index in actions]
        predictions["is_under_routed"] = actions < labels
        predictions["is_over_routed"] = actions > labels
        predictions["is_hard_to_low"] = (labels == 2) & (actions == 0)
        predictions["locked_penalty"] = penalty
        predictions.to_parquet(
            output_dir / method / "routerarena-predictions.parquet", index=False
        )
        results[method] = {
            "row_count": len(arena),
            "accuracy": float(accuracy_score(labels, actions)),
            "macro_f1": float(
                f1_score(labels, actions, average="macro", zero_division=0)
            ),
            "balanced_accuracy": float(balanced_accuracy_score(labels, actions)),
            "under_routing_rate": float((actions < labels).mean()),
            "over_routing_rate": float((actions > labels).mean()),
            "hard_to_low_rate": float(
                ((labels == 2) & (actions == 0)).sum() / max((labels == 2).sum(), 1)
            ),
            "low_rate": float((actions == 0).mean()),
            "medium_rate": float((actions == 1).mean()),
            "high_rate": float((actions == 2).mean()),
            "interpretation": "external_label_alignment_only",
        }
    summary = {
        "schema_version": "routing-objective-routerarena-alignment-v1",
        "selection_influence": False,
        "cost_quality_claim_allowed": False,
        "default_relative_cost_budget": float(lock["default_relative_cost_budget"]),
        "abstract_routerarena_decision_costs": [0.0, 0.5, 1.0],
        "methods": results,
    }
    write_json(output_dir / "summary.json", summary)
    return summary
