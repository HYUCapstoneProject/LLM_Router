from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from ..shared.dataset_io import verify_file


REQUIRED_BASE_COLUMNS = ("sample_id", "prompt", "eval_name")
NON_MODEL_COLUMNS = {
    *REQUIRED_BASE_COLUMNS,
    "oracle_model_to_route_to",
    "index",
    "idx",
}


class RouterBenchError(RuntimeError):
    """Raised when RouterBench data violates the labeling contract."""


@dataclass(frozen=True)
class ModelTiers:
    low: tuple[str, ...]
    medium: tuple[str, ...]
    high: tuple[str, ...]
    statistics: Mapping[str, Mapping[str, float | None]]

    @property
    def all_models(self) -> tuple[str, ...]:
        return self.low + self.medium + self.high

    def tier_for_model(self, model: str) -> str:
        for tier_name in ("low", "medium", "high"):
            if model in getattr(self, tier_name):
                return tier_name
        raise KeyError(model)

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": {
                "low": list(self.low),
                "medium": list(self.medium),
                "high": list(self.high),
            },
            "model_statistics": self.statistics,
        }


def load_trusted_pickle(
    path: Path, source_config: Mapping[str, Any], trust_pickle: bool
) -> pd.DataFrame:
    verify_file(path, source_config)
    if not trust_pickle:
        raise RouterBenchError(
            "RouterBench is distributed as Pickle. Re-run with --trust-pickle "
            "after verifying the configured URL and checksum."
        )
    dataframe = pd.read_pickle(path)
    if not isinstance(dataframe, pd.DataFrame):
        raise RouterBenchError(f"Expected pandas.DataFrame in {path}")
    validate_base_schema(dataframe)
    return dataframe


def validate_base_schema(dataframe: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_BASE_COLUMNS if column not in dataframe]
    if missing:
        raise RouterBenchError(f"Missing required RouterBench columns: {missing}")


def detect_model_columns(dataframe: pd.DataFrame) -> list[str]:
    validate_base_schema(dataframe)
    models: list[str] = []
    for column in dataframe.columns:
        if column in NON_MODEL_COLUMNS or "|" in str(column):
            continue
        column_name = str(column)
        has_model_sibling = any(
            f"{column_name}|{suffix}" in dataframe.columns
            for suffix in ("model_response", "total_cost")
        )
        if not has_model_sibling:
            continue
        non_null = dataframe[column].notna()
        if not non_null.any():
            continue
        numeric = pd.to_numeric(dataframe.loc[non_null, column], errors="coerce")
        if numeric.notna().all():
            models.append(column_name)
    if not models:
        raise RouterBenchError("No numeric model performance columns were detected")
    return sorted(models)


def reduce_to_labeling_columns(
    dataframe: pd.DataFrame, models: Iterable[str]
) -> pd.DataFrame:
    selected = list(REQUIRED_BASE_COLUMNS)
    selected.extend(models)
    selected.extend(
        f"{model}|total_cost"
        for model in models
        if f"{model}|total_cost" in dataframe.columns
    )
    if "oracle_model_to_route_to" in dataframe.columns:
        selected.append("oracle_model_to_route_to")
    return dataframe.loc[:, selected].copy()


def _matches(value: str, patterns: Iterable[str]) -> bool:
    lowered = value.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def preferred_source_for_row(
    sample_id: object, eval_name: object, canonical_config: Mapping[str, Any]
) -> str:
    identity = f"{sample_id} {eval_name}".lower()
    if _matches(identity, canonical_config["five_shot_patterns"]):
        return "5shot"
    if _matches(identity, canonical_config["zero_shot_patterns"]):
        return "0shot"
    return str(canonical_config["default_source"])


def extract_query_text(prompt: object) -> str:
    if prompt is None or (isinstance(prompt, float) and math.isnan(prompt)):
        return ""
    text = str(prompt)
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError, TypeError):
        return text.strip()
    if isinstance(parsed, (list, tuple)):
        non_empty = [str(item).strip() for item in parsed if str(item).strip()]
        demonstration_indexes = [
            index
            for index, item in enumerate(non_empty[:-1])
            if "Answer:" in item
        ]
        if demonstration_indexes:
            non_empty = non_empty[demonstration_indexes[-1] + 1 :]
        text = "\n\n".join(non_empty)
    else:
        text = str(parsed).strip()

    final_answer = text.rfind("Answer:")
    if final_answer >= 0:
        prefix = text[:final_answer]
        previous_answer = prefix.rfind("Answer:")
        if previous_answer >= 0:
            boundary = prefix.find("\n\n", previous_answer)
            if boundary >= 0:
                text = prefix[boundary + 2 :]
            else:
                text = prefix
        else:
            text = prefix
    text = re.sub(r"\n?Answer:\s*$", "", text).strip()
    return text


def infer_domain(sample_id: object, eval_name: object) -> str:
    identity = f"{sample_id} {eval_name}".lower()
    mappings = (
        (("gsm8k", "grade-school-math"), "math"),
        (("mbpp", "code", "humaneval"), "code"),
        (("hellaswag", "winogrande", "arc"), "commonsense"),
        (("mmlu",), "knowledge"),
        (("mt_bench", "mt-bench", "mtbench"), "conversation"),
        (("rag",), "rag"),
    )
    for patterns, domain in mappings:
        if any(pattern in identity for pattern in patterns):
            return domain
    return "other"


def filter_eval_scope(
    dataframe: pd.DataFrame, canonical_config: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    patterns = canonical_config.get("included_eval_patterns")
    if not patterns:
        return dataframe.copy(), {
            "scope_excluded_row_count": 0,
            "scope_excluded_eval_names": [],
        }
    identity = (
        dataframe["sample_id"].astype(str) + " " + dataframe["eval_name"].astype(str)
    )
    included = identity.map(lambda value: _matches(value, patterns))
    excluded = dataframe.loc[~included]
    scoped = dataframe.loc[included].copy().reset_index(drop=True)
    return scoped, {
        "scope_excluded_row_count": int(len(excluded)),
        "scope_excluded_eval_names": sorted(
            excluded["eval_name"].astype(str).unique().tolist()
        ),
    }


def canonicalize_sources(
    zero_shot: pd.DataFrame,
    five_shot: pd.DataFrame,
    canonical_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    zero_models = detect_model_columns(zero_shot)
    five_models = detect_model_columns(five_shot)
    if set(zero_models) != set(five_models):
        raise RouterBenchError(
            "0-shot and 5-shot model columns differ: "
            f"only_0shot={sorted(set(zero_models) - set(five_models))}, "
            f"only_5shot={sorted(set(five_models) - set(zero_models))}"
        )
    models = sorted(zero_models)
    frames = []
    for source_name, dataframe in (("0shot", zero_shot), ("5shot", five_shot)):
        reduced = reduce_to_labeling_columns(dataframe, models)
        reduced["shot_source"] = source_name
        frames.append(reduced)
    combined = pd.concat(frames, ignore_index=True)
    combined["preferred_source"] = [
        preferred_source_for_row(sample_id, eval_name, canonical_config)
        for sample_id, eval_name in zip(
            combined["sample_id"], combined["eval_name"], strict=False
        )
    ]
    combined["source_priority"] = (
        combined["shot_source"] != combined["preferred_source"]
    ).astype("int8")
    combined.sort_values(
        ["sample_id", "source_priority", "shot_source"],
        kind="stable",
        inplace=True,
    )
    canonical = combined.drop_duplicates("sample_id", keep="first").copy()
    canonical["query_text"] = canonical["prompt"].map(extract_query_text)
    canonical["domain"] = [
        infer_domain(sample_id, eval_name)
        for sample_id, eval_name in zip(
            canonical["sample_id"], canonical["eval_name"], strict=False
        )
    ]
    canonical["needs_retrieval"] = canonical["domain"].eq("rag")
    canonical.drop(columns=["preferred_source", "source_priority"], inplace=True)
    canonical.sort_values("sample_id", kind="stable", inplace=True)
    canonical.reset_index(drop=True, inplace=True)
    if canonical["sample_id"].duplicated().any():
        raise RouterBenchError("Canonicalization left duplicate sample_id values")
    return canonical, models


def calibrate_model_tiers(
    dataframe: pd.DataFrame, models: Iterable[str]
) -> ModelTiers:
    model_list = list(models)
    performance = dataframe.loc[:, model_list].apply(pd.to_numeric, errors="coerce")
    per_eval = performance.groupby(dataframe["eval_name"], dropna=False).mean()
    percentile = per_eval.rank(axis=1, method="average", pct=True)
    capability = percentile.mean(axis=0, skipna=True)
    mean_cost: dict[str, float | None] = {}
    for model in model_list:
        cost_column = f"{model}|total_cost"
        if cost_column in dataframe:
            value = pd.to_numeric(dataframe[cost_column], errors="coerce").mean()
            mean_cost[model] = None if pd.isna(value) else float(value)
        else:
            mean_cost[model] = None

    ordered = sorted(
        model_list,
        key=lambda model: (
            float(capability.get(model, float("-inf"))),
            mean_cost[model] if mean_cost[model] is not None else float("inf"),
            model,
        ),
    )
    groups = [tuple(group.tolist()) for group in np.array_split(np.array(ordered), 3)]
    if any(not group for group in groups):
        raise RouterBenchError("At least three model columns are required for tiering")
    statistics = {
        model: {
            "capability_score": float(capability[model]),
            "mean_cost": mean_cost[model],
        }
        for model in ordered
    }
    return ModelTiers(
        low=groups[0], medium=groups[1], high=groups[2], statistics=statistics
    )


def acceptance_thresholds(
    dataframe: pd.DataFrame, acceptance_config: Mapping[str, Any]
) -> pd.Series:
    identities = (
        dataframe["sample_id"].astype(str) + " " + dataframe["eval_name"].astype(str)
    ).str.lower()
    thresholds = pd.Series(
        float(acceptance_config["default_threshold"]), index=dataframe.index
    )
    continuous = identities.map(
        lambda identity: _matches(identity, acceptance_config["continuous_patterns"])
    )
    exact = identities.map(
        lambda identity: _matches(identity, acceptance_config["exact_match_patterns"])
    )
    thresholds.loc[continuous] = float(acceptance_config["continuous_threshold"])
    thresholds.loc[exact] = float(acceptance_config["exact_match_threshold"])
    return thresholds


def _rate_and_count(
    accepted: pd.DataFrame, valid: pd.DataFrame, models: Iterable[str]
) -> tuple[pd.Series, pd.Series]:
    columns = list(models)
    valid_count = valid.loc[:, columns].sum(axis=1)
    success_count = accepted.loc[:, columns].sum(axis=1)
    rate = success_count.div(valid_count.replace(0, np.nan))
    return rate, valid_count


def label_questions(
    dataframe: pd.DataFrame,
    tiers: ModelTiers,
    acceptance_config: Mapping[str, Any],
    labeling_config: Mapping[str, Any],
) -> pd.DataFrame:
    output = dataframe.copy()
    models = list(tiers.all_models)
    performance = output.loc[:, models].apply(pd.to_numeric, errors="coerce")
    thresholds = acceptance_thresholds(output, acceptance_config)
    valid = performance.notna()
    accepted = performance.ge(thresholds, axis="index") & valid

    tier_rates: dict[str, pd.Series] = {}
    tier_counts: dict[str, pd.Series] = {}
    for tier_name in ("low", "medium", "high"):
        tier_rates[tier_name], tier_counts[tier_name] = _rate_and_count(
            accepted, valid, getattr(tiers, tier_name)
        )
        output[f"{tier_name}_success_rate"] = tier_rates[tier_name]
        output[f"{tier_name}_valid_model_count"] = tier_counts[tier_name]
    output["overall_solve_rate"] = accepted.sum(axis=1).div(
        valid.sum(axis=1).replace(0, np.nan)
    )

    strategy = str(labeling_config.get("strategy", "tier_consensus"))
    unresolved_label = str(labeling_config.get("unresolved_label", "hard"))
    if unresolved_label not in {"hard", "unresolved"}:
        raise RouterBenchError(
            "labeling.unresolved_label must be 'hard' or 'unresolved'"
        )
    minimum_valid = int(labeling_config["minimum_valid_models_for_confidence"])
    if strategy == "tier_consensus":
        low_pass = tier_rates["low"].ge(
            float(labeling_config["low_consensus_threshold"])
        )
        medium_pass = tier_rates["medium"].ge(
            float(labeling_config["medium_consensus_threshold"])
        )
        high_pass = tier_rates["high"].ge(
            float(labeling_config["high_consensus_threshold"])
        )
        output["difficulty_label"] = np.select(
            [low_pass, medium_pass, high_pass],
            ["easy", "medium", "hard"],
            default=unresolved_label,
        )
        unresolved_reason = (
            "unresolved" if unresolved_label == "unresolved" else "unsolved"
        )
        output["label_reason"] = np.select(
            [low_pass, ~low_pass & medium_pass, ~low_pass & ~medium_pass & high_pass],
            ["low_tier_consensus", "medium_tier_consensus", "strong_only"],
            default=unresolved_reason,
        )
        decisive_count = np.select(
            [low_pass, ~low_pass & medium_pass],
            [tier_counts["low"], tier_counts["medium"]],
            default=tier_counts["high"],
        )
        any_missing = pd.Series(False, index=output.index)
        for tier_name in ("low", "medium", "high"):
            any_missing |= tier_counts[tier_name].lt(len(getattr(tiers, tier_name)))
        non_monotonic = tier_rates["low"].gt(
            tier_rates["medium"].fillna(-1)
        ) | tier_rates["medium"].gt(tier_rates["high"].fillna(-1))
        confidence = np.where(
            decisive_count < minimum_valid,
            "low",
            np.where(any_missing | non_monotonic, "medium", "high"),
        )
    elif strategy == "overall_solve_rate":
        easy_threshold = float(labeling_config["easy_min_solve_rate"])
        medium_threshold = float(labeling_config["medium_min_solve_rate"])
        if not 0.0 <= medium_threshold < easy_threshold <= 1.0:
            raise RouterBenchError(
                "overall solve-rate thresholds must satisfy "
                "0 <= medium_min_solve_rate < easy_min_solve_rate <= 1"
            )
        overall_rate = output["overall_solve_rate"]
        overall_valid_count = valid.sum(axis=1)
        no_valid_models = overall_valid_count.eq(0)
        output["difficulty_label"] = np.select(
            [
                no_valid_models,
                overall_rate.ge(easy_threshold),
                overall_rate.ge(medium_threshold),
            ],
            [unresolved_label, "easy", "medium"],
            default="hard",
        )
        unresolved_reason = (
            "unresolved" if unresolved_label == "unresolved" else "unsolved"
        )
        output["label_reason"] = np.select(
            [
                no_valid_models,
                overall_rate.ge(easy_threshold),
                overall_rate.ge(medium_threshold),
            ],
            [
                unresolved_reason,
                "overall_solve_rate_easy",
                "overall_solve_rate_medium",
            ],
            default="overall_solve_rate_hard",
        )
        any_missing = overall_valid_count.lt(len(models))
        confidence = np.where(
            overall_valid_count.lt(minimum_valid),
            "low",
            np.where(any_missing, "medium", "high"),
        )
    else:
        raise RouterBenchError(
            "labeling.strategy must be 'tier_consensus' or 'overall_solve_rate'"
        )
    output["label_confidence"] = confidence
    output["label_weight"] = output["label_confidence"].map(
        {"high": 1.0, "medium": 0.5, "low": 0.0}
    )

    tier_by_model = {
        model: tier_name
        for tier_name in ("low", "medium", "high")
        for model in getattr(tiers, tier_name)
    }

    def cheapest_tier(row_index: Any) -> str | None:
        correct_models = [model for model in models if accepted.at[row_index, model]]
        if not correct_models:
            return None
        models_with_cost = []
        for model in correct_models:
            cost_column = f"{model}|total_cost"
            if cost_column in output:
                cost = pd.to_numeric(
                    pd.Series([output.at[row_index, cost_column]]), errors="coerce"
                ).iloc[0]
                if not pd.isna(cost):
                    models_with_cost.append((float(cost), model))
        chosen = min(models_with_cost)[1] if models_with_cost else correct_models[0]
        return tier_by_model[chosen]

    output["cheapest_correct_tier"] = [
        cheapest_tier(index) for index in output.index
    ]
    return output


def stratified_smoke_sample(
    dataframe: pd.DataFrame, sample_size: int, seed: int
) -> pd.DataFrame:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if len(dataframe) < sample_size:
        raise RouterBenchError(
            f"Requested {sample_size} smoke rows from only {len(dataframe)} rows"
        )
    groups = sorted(dataframe["eval_name"].astype(str).unique())
    rng = np.random.default_rng(seed)
    if len(groups) > sample_size:
        groups = sorted(rng.choice(groups, size=sample_size, replace=False).tolist())
    covered = (
        dataframe[dataframe["eval_name"].astype(str).isin(groups)]
        .groupby("eval_name", group_keys=False, sort=True)
        .sample(n=1, random_state=seed)
    )
    target_per_domain = max(1, sample_size // 10)
    selected = covered
    for offset, domain in enumerate(sorted(dataframe["domain"].astype(str).unique())):
        current_count = int(selected["domain"].astype(str).eq(domain).sum())
        available_slots = sample_size - len(selected)
        needed = min(target_per_domain - current_count, available_slots)
        if needed <= 0:
            continue
        candidates = dataframe.drop(index=selected.index)
        candidates = candidates[candidates["domain"].astype(str).eq(domain)]
        needed = min(needed, len(candidates))
        if needed:
            selected = pd.concat(
                [
                    selected,
                    candidates.sample(n=needed, random_state=seed + offset + 1),
                ]
            )
    remaining_size = sample_size - len(selected)
    if remaining_size:
        remaining = dataframe.drop(index=selected.index).sample(
            n=remaining_size, random_state=seed
        )
        selected = pd.concat([selected, remaining])
    sampled = selected.sort_values("sample_id", kind="stable").reset_index(drop=True)
    if sampled["sample_id"].duplicated().any():
        raise RouterBenchError("Smoke sample contains duplicate sample_id values")
    return sampled


def build_metrics(
    labeled: pd.DataFrame,
    canonical: pd.DataFrame,
    tiers: ModelTiers,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_version": config["dataset_version"],
        "tier_config_version": config["tier_config_version"],
        "label_rule_version": config["label_rule_version"],
        "labeling_strategy": str(
            config.get("labeling", {}).get("strategy", "tier_consensus")
        ),
        "canonical_row_count": int(len(canonical)),
        "labeled_row_count": int(len(labeled)),
        "model_count": len(tiers.all_models),
        "models": tiers.to_dict()["models"],
        "label_distribution": labeled["difficulty_label"].value_counts().to_dict(),
        "reason_distribution": labeled["label_reason"].value_counts().to_dict(),
        "confidence_distribution": labeled["label_confidence"].value_counts().to_dict(),
        "domain_distribution": labeled["domain"].value_counts().to_dict(),
        "shot_source_distribution": labeled["shot_source"].value_counts().to_dict(),
        "eval_count": int(labeled["eval_name"].nunique()),
        "duplicate_sample_id_count": int(labeled["sample_id"].duplicated().sum()),
        "duplicate_query_text_count": int(labeled["query_text"].duplicated().sum()),
        "empty_query_text_count": int(labeled["query_text"].eq("").sum()),
    }


def validate_labeled_output(
    labeled: pd.DataFrame, expected_rows: int | None = None
) -> None:
    required = {
        *REQUIRED_BASE_COLUMNS,
        "query_text",
        "domain",
        "shot_source",
        "low_success_rate",
        "medium_success_rate",
        "high_success_rate",
        "overall_solve_rate",
        "cheapest_correct_tier",
        "difficulty_label",
        "label_reason",
        "label_confidence",
        "label_weight",
        "needs_retrieval",
    }
    missing = sorted(required - set(labeled.columns))
    if missing:
        raise RouterBenchError(f"Labeled output is missing columns: {missing}")
    if expected_rows is not None and len(labeled) != expected_rows:
        raise RouterBenchError(f"Expected {expected_rows} rows, got {len(labeled)}")
    if labeled.empty:
        raise RouterBenchError("Labeled output is empty")
    if labeled["sample_id"].duplicated().any():
        raise RouterBenchError("Labeled output has duplicate sample_id values")
    if labeled["query_text"].fillna("").astype(str).str.strip().eq("").any():
        raise RouterBenchError("Labeled output has an empty query_text")
    if not labeled["difficulty_label"].isin(
        {"easy", "medium", "hard", "unresolved"}
    ).all():
        raise RouterBenchError("Labeled output has an invalid difficulty label")
    for column in (
        "low_success_rate",
        "medium_success_rate",
        "high_success_rate",
        "overall_solve_rate",
    ):
        values = labeled[column].dropna()
        if not values.between(0.0, 1.0).all():
            raise RouterBenchError(f"{column} contains a value outside [0, 1]")


def validate_smoke_output(labeled: pd.DataFrame, expected_rows: int) -> None:
    """Backward-compatible smoke validator using the common output contract."""

    validate_labeled_output(labeled, expected_rows)
