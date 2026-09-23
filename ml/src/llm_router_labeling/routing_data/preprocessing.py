from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from sklearn.model_selection import GroupKFold

from ..shared.dataset_io import write_json
from ..shared.text_normalization import normalize_for_duplicate_detection
from ..training.routerbench_preprocessing import (
    assign_duplicate_groups,
    clean_model_input_text,
)


LABELS = ("easy", "medium", "hard")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
OUTPUT_COLUMNS = (
    "sample_id",
    "source_dataset",
    "source_split",
    "category",
    "source_task",
    "query_text",
    "model_input_text",
    "valid_model_count",
    "accepted_model_count",
    "overall_solve_rate",
    "difficulty_label",
    "label_confidence",
    "sample_weight",
    "duplicate_group_id",
    "split",
    "dataset_version",
    "label_rule_version",
    "preprocessing_version",
)


class RoutingDataError(ValueError):
    """Raised when an integrated comparison-data contract is violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, expected: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise RoutingDataError(f"Required source file is missing: {path}")
    expected_size = int(expected["size_bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RoutingDataError(
            f"Source size mismatch for {path}: expected {expected_size}, got {actual_size}"
        )
    expected_sha = str(expected["sha256"])
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise RoutingDataError(
            f"Source SHA-256 mismatch for {path}: expected {expected_sha}, got {actual_sha}"
        )


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _labels_from_rates(rates: np.ndarray, hard: float, easy: float) -> np.ndarray:
    if not 0.0 <= hard < easy <= 1.0:
        raise RoutingDataError("Thresholds must satisfy 0 <= hard < easy <= 1")
    return np.where(rates >= easy, "easy", np.where(rates >= hard, "medium", "hard"))


def _strip_xroute_wrapper(value: object) -> str:
    text, _ = clean_model_input_text(value)
    marker = "[User Query]"
    if marker in text:
        text = text.split(marker, 1)[1].strip()
    text = re.sub(r"^\[System Instruction\].*?\n\s*", "", text, flags=re.DOTALL)
    return text.strip()


def _routerbench_models(model_tiers_path: Path) -> list[str]:
    with model_tiers_path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    models: list[str] = []
    for tier in ("low", "medium", "high"):
        models.extend(str(model) for model in payload["models"][tier])
    if len(models) != len(set(models)) or not models:
        raise RoutingDataError("RouterBench model tier artifact has invalid model names")
    return models


def _routerbench_accepted(labeled: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    identities = (
        labeled["sample_id"].astype(str) + " " + labeled["eval_name"].astype(str)
    ).str.casefold()
    continuous = identities.str.contains(
        r"grade-school-math|gsm8k|mbpp|mt_bench|mt-bench|mtbench|rag",
        regex=True,
    )
    thresholds = pd.Series(np.where(continuous, 0.70, 1.0), index=labeled.index)
    values = labeled.loc[:, list(models)].apply(pd.to_numeric, errors="coerce")
    return values.ge(thresholds, axis="index").where(values.notna())


def _load_routerbench(config: Mapping[str, Any], root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rb = config["routerbench"]
    labeled_path = root / str(rb["labeled_path"])
    metadata_path = root / str(rb["metadata_path"])
    tiers_path = root / str(rb["model_tiers_path"])
    for path in (labeled_path, metadata_path, tiers_path):
        if not path.is_file():
            raise RoutingDataError(f"RouterBench prerequisite artifact is missing: {path}")

    labeled = pd.read_parquet(labeled_path)
    metadata = pd.read_parquet(metadata_path)
    models = _routerbench_models(tiers_path)
    missing = sorted(set(models) - set(labeled.columns))
    if missing:
        raise RoutingDataError(f"RouterBench labeled data is missing models: {missing}")
    accepted = _routerbench_accepted(labeled, models)
    accepted.index = labeled["sample_id"].astype(str)

    meta_columns = [
        "sample_id",
        "query_text",
        "model_input_text",
        "eval_name",
        "domain",
        "label_confidence",
        "sample_weight",
        "duplicate_group_id",
        "split",
        "included_in_training",
    ]
    missing_meta = sorted(set(meta_columns) - set(metadata.columns))
    if missing_meta:
        raise RoutingDataError(f"RouterBench metadata is missing columns: {missing_meta}")
    merged = metadata.loc[:, meta_columns].merge(
        labeled.loc[:, ["sample_id", "overall_solve_rate"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    merged = merged.loc[merged["included_in_training"].astype(bool)].copy()
    merged["sample_id"] = "routerbench:" + merged["sample_id"].astype(str)
    merged["source_dataset"] = "routerbench"
    merged["source_split"] = merged["split"].astype(str)
    merged["category"] = merged["domain"].astype(str)
    merged["source_task"] = merged["eval_name"].astype(str)
    merged["valid_model_count"] = len(models)
    aligned_accepted = accepted.loc[
        merged["sample_id"].str.removeprefix("routerbench:")
    ].copy()
    aligned_accepted.index = merged["sample_id"]
    merged["accepted_model_count"] = aligned_accepted.astype("boolean").sum(axis=1).to_numpy()
    merged["label_confidence"] = merged["label_confidence"].fillna("high").astype(str)
    merged["sample_weight"] = pd.to_numeric(merged["sample_weight"], errors="raise")
    return merged, aligned_accepted


def _validate_xroute_rows(
    frame: pd.DataFrame,
    *,
    split: str,
    candidates: Sequence[str],
    exact_metrics: set[str],
    continuous_metrics: set[str],
) -> pd.DataFrame:
    required = {
        "task_name",
        "query",
        "metric",
        "task_id",
        "model_name",
        "performance",
        "embedding_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RoutingDataError(f"xRouteBench {split} is missing columns: {missing}")
    unknown_models = sorted(set(frame["model_name"].astype(str)) - set(candidates))
    if unknown_models:
        raise RoutingDataError(f"xRouteBench {split} has unknown models: {unknown_models}")
    output = frame.copy()
    output["performance"] = pd.to_numeric(output["performance"], errors="coerce")
    invalid_range = output["performance"].notna() & ~output["performance"].between(0, 1)
    if invalid_range.any():
        raise RoutingDataError(f"xRouteBench {split} has performance outside [0, 1]")
    output["normalized_query"] = output["query"].map(normalize_for_duplicate_detection)
    if output["normalized_query"].eq("").any():
        raise RoutingDataError(f"xRouteBench {split} has an empty normalized query")
    source_identifier = output["task_id"].where(
        output["task_id"].notna(), output["embedding_id"]
    )
    output["query_key"] = (
        output["task_name"].astype(str)
        + "|"
        + source_identifier.astype(str)
        + "|"
        + output["normalized_query"]
    )
    duplicate_runs = output.duplicated(["query_key", "model_name"], keep=False)
    if duplicate_runs.any():
        examples = output.loc[duplicate_runs, ["query_key", "model_name"]].head(5)
        raise RoutingDataError(
            "xRouteBench contains duplicate query/model runs: "
            + examples.to_dict("records").__repr__()
        )
    known_metrics = exact_metrics | continuous_metrics
    output["metric_known"] = output["metric"].astype(str).isin(known_metrics)
    output["accepted"] = pd.NA
    exact = output["metric"].astype(str).isin(exact_metrics) & output["performance"].notna()
    continuous = (
        output["metric"].astype(str).isin(continuous_metrics)
        & output["performance"].notna()
    )
    output.loc[exact, "accepted"] = output.loc[exact, "performance"].eq(1.0)
    output.loc[continuous, "accepted"] = output.loc[continuous, "performance"].ge(0.70)
    return output


def _pivot_xroute(
    frame: pd.DataFrame,
    *,
    source_split: str,
    candidates: Sequence[str],
    minimum_valid: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    first = frame.sort_values(["query_key", "model_name"], kind="stable").groupby(
        "query_key", sort=True, as_index=False
    ).first()
    accepted = frame.pivot(index="query_key", columns="model_name", values="accepted")
    accepted = accepted.reindex(columns=list(candidates))
    valid_count = accepted.notna().sum(axis=1)
    accepted_count = accepted.astype("boolean").sum(axis=1)
    metric_unknown = frame.groupby("query_key")["metric_known"].apply(lambda values: not values.all())
    eligible = valid_count.ge(minimum_valid)

    first = first.set_index("query_key").loc[accepted.index].copy()
    first["valid_model_count"] = valid_count
    first["accepted_model_count"] = accepted_count
    first["overall_solve_rate"] = accepted_count.div(valid_count.replace(0, np.nan))
    first["sample_id"] = [
        "xroutebench:"
        + hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:20]
        for key in first.index
    ]
    first["source_dataset"] = "xroutebench"
    first["source_split"] = source_split
    first["category"] = first["task_name"].astype(str)
    first["source_task"] = first["task_id"].where(
        first["task_id"].notna(), first["embedding_id"]
    ).astype(str)
    first["query_text"] = first["query"].astype(str)
    first["model_input_text"] = first["query"].map(_strip_xroute_wrapper)
    first["label_confidence"] = np.where(valid_count.eq(len(candidates)), "high", "medium")
    first["sample_weight"] = np.where(valid_count.eq(len(candidates)), 1.0, 0.5)
    first["split"] = "external_test" if source_split == "test" else "train"
    first["eligible"] = eligible
    accepted.index = first["sample_id"]
    metrics = {
        "raw_run_rows": int(len(frame)),
        "query_rows": int(len(first)),
        "eligible_query_rows": int(eligible.sum()),
        "excluded_too_few_valid_models": int((~eligible).sum()),
        "queries_with_unknown_metric": int(metric_unknown.sum()),
    }
    return first.reset_index(drop=True), accepted, metrics


def _assign_xroute_validation(
    frame: pd.DataFrame,
    *,
    fraction: float,
    seed: int,
    score_boundaries: Sequence[float],
) -> None:
    if not 0.0 < fraction < 1.0:
        raise RoutingDataError("xRouteBench validation fraction must be in (0, 1)")
    official_train = frame["source_split"].eq("train")
    boundaries = np.array(sorted(set(float(value) for value in score_boundaries)))
    if boundaries.size == 0 or (boundaries < 0.0).any() or (boundaries > 1.0).any():
        raise RoutingDataError("Score boundaries must contain values in [0, 1]")
    working = frame.loc[official_train].copy()
    # Every threshold boundary becomes a stratum edge.  This keeps the
    # validation share close to the requested fraction for every possible
    # candidate label, without selecting a threshold on validation data.
    working["_score_stratum"] = np.searchsorted(
        boundaries,
        working["overall_solve_rate"].to_numpy(float),
        side="right",
    )
    groups = working.groupby("duplicate_group_id", sort=True).agg(
        category=("category", "first"),
        score_stratum=("_score_stratum", "first"),
        row_count=("sample_id", "size"),
    )
    validation_groups: set[str] = set()
    for (category, score_stratum), category_groups in groups.groupby(
        ["category", "score_stratum"], sort=True
    ):
        ordered = sorted(
            category_groups.index.astype(str),
            key=lambda group: _stable_fraction(
                f"{category}:{score_stratum}:{group}", seed
            ),
        )
        target = round(int(category_groups["row_count"].sum()) * fraction)
        selected_rows = 0
        for group in ordered:
            if selected_rows >= target:
                break
            validation_groups.add(group)
            selected_rows += int(category_groups.at[group, "row_count"])
    mask = official_train & frame["duplicate_group_id"].astype(str).isin(validation_groups)
    frame.loc[mask, "split"] = "validation"
    frame.loc[official_train & ~mask, "split"] = "train"


def _deduplicate_with_priority(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    priority = {"external_test": 4, "test": 3, "validation": 2, "train": 1}
    working = frame.copy()
    working["_priority"] = working["split"].map(priority).astype(int)
    keep_indices: list[int] = []
    dropped: Counter[str] = Counter()
    cross_source_groups = 0
    for _, group in working.groupby("duplicate_group_id", sort=True):
        if group["source_dataset"].nunique() > 1:
            cross_source_groups += 1
        highest = int(group["_priority"].max())
        winners = group.loc[group["_priority"].eq(highest)].sort_values(
            ["source_dataset", "sample_id"], kind="stable"
        )
        winner = int(winners.index[0])
        keep_indices.append(winner)
        for index, row in group.iterrows():
            if int(index) != winner:
                dropped[f"{row['source_dataset']}|{row['split']}"] += 1
    output = working.loc[sorted(keep_indices)].drop(columns=["_priority"]).reset_index(drop=True)
    return output, {
        "input_rows": int(len(frame)),
        "output_rows": int(len(output)),
        "dropped_rows": int(len(frame) - len(output)),
        "dropped_by_source_split": dict(sorted(dropped.items())),
        "cross_source_duplicate_group_count": cross_source_groups,
    }


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, sources: np.ndarray) -> dict[str, float]:
    ids_true = np.array([LABEL_TO_ID[str(value)] for value in y_true], dtype=int)
    ids_pred = np.array([LABEL_TO_ID[str(value)] for value in y_pred], dtype=int)
    per_source = {
        source: f1_score(
            y_true[sources == source],
            y_pred[sources == source],
            labels=list(LABELS),
            average="macro",
            zero_division=0,
        )
        for source in sorted(set(sources))
    }
    qwk = cohen_kappa_score(ids_true, ids_pred, labels=[0, 1, 2], weights="quadratic")
    hard_mask = y_true == "hard"
    return {
        "source_macro_f1": float(np.mean(list(per_source.values()))),
        "routerbench_macro_f1": float(per_source.get("routerbench", 0.0)),
        "xroutebench_macro_f1": float(per_source.get("xroutebench", 0.0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "qwk": float(qwk),
        "qwk_01": float((qwk + 1.0) / 2.0),
        "ordinal_mae": float(np.abs(ids_true - ids_pred).mean()),
        "hard_to_easy_rate": float((y_pred[hard_mask] == "easy").mean()) if hard_mask.any() else 0.0,
    }


def _bootstrap_label_agreement(
    frame: pd.DataFrame,
    accepted_by_source: Mapping[str, pd.DataFrame],
    *,
    hard: float,
    easy: float,
    iterations: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    agreements: list[float] = []
    train = frame.loc[frame["split"].eq("train")]
    for source_name, source_rows in train.groupby("source_dataset", sort=True):
        matrix_frame = accepted_by_source[source_name].loc[source_rows["sample_id"]]
        matrix = matrix_frame.to_numpy(dtype=float, na_value=np.nan)
        base = _labels_from_rates(source_rows["overall_solve_rate"].to_numpy(float), hard, easy)
        if matrix.shape[1] == 0:
            continue
        source_scores = np.zeros(iterations, dtype=float)
        for iteration in range(iterations):
            columns = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
            sample = matrix[:, columns]
            valid = np.isfinite(sample).sum(axis=1)
            rates = np.nansum(sample, axis=1) / np.maximum(valid, 1)
            labels = _labels_from_rates(rates, hard, easy)
            source_scores[iteration] = float((labels == base).mean())
        agreements.extend(source_scores.tolist())
    return float(np.mean(agreements)) if agreements else 0.0


def _candidate_sanity(
    train: pd.DataFrame,
    labels: np.ndarray,
    agreement: float,
) -> tuple[bool, list[str], dict[str, Any]]:
    working = train.copy()
    working["difficulty_label"] = labels
    counts = working["difficulty_label"].value_counts().reindex(LABELS, fill_value=0)
    reasons: list[str] = []
    for label in LABELS:
        if int(counts[label]) < 500 or float(counts[label] / len(working)) < 0.10:
            reasons.append(f"class_gate:{label}")
    for source in sorted(working["source_dataset"].unique()):
        present = set(working.loc[working["source_dataset"].eq(source), "difficulty_label"])
        if present != set(LABELS):
            reasons.append(f"missing_source_label:{source}")
    means = working.groupby("difficulty_label")["overall_solve_rate"].mean().reindex(LABELS)
    if not (float(means["easy"]) > float(means["medium"]) > float(means["hard"])):
        reasons.append("solve_rate_order")
    if agreement < 0.85:
        reasons.append("bootstrap_agreement")
    return not reasons, reasons, {
        "class_counts": {label: int(counts[label]) for label in LABELS},
        "class_shares": {label: float(counts[label] / len(working)) for label in LABELS},
        "mean_solve_rate": {label: float(means[label]) for label in LABELS},
    }


def _coverage_status(counts: pd.Series, total: int, policy: Mapping[str, Any], *, evaluation: bool) -> tuple[str, str]:
    values = counts.reindex(LABELS, fill_value=0).astype(int)
    if evaluation:
        minimum = int(values.min())
        if minimum >= 100:
            return "confirmatory", "all_label_cells_at_least_100"
        if minimum >= 30:
            return "exploratory", "one_or_more_label_cells_below_100"
        return "insufficient_coverage", "one_or_more_label_cells_below_30"
    if total < int(policy["core_category_minimum_rows"]):
        return "small_category", "category_total_below_300"
    shares = values / max(total, 1)
    if (
        float(shares.min()) >= float(policy["green_minimum_share"])
        and float(shares.max()) <= float(policy["green_maximum_share"])
        and int(values.min()) >= int(policy["green_minimum_cell_rows"])
    ):
        return "green", "recommended_coverage"
    if (
        float(shares.min()) >= float(policy["yellow_minimum_share"])
        and float(shares.max()) <= float(policy["yellow_maximum_share"])
        and int(values.min()) >= int(policy["yellow_minimum_cell_rows"])
    ):
        return "yellow", "allowed_with_caution"
    return "red", "label_share_or_cell_count_below_policy"


def _coverage_report(
    frame: pd.DataFrame,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    phase: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    policy = config["coverage"]
    for candidate in candidates:
        labeled = frame.copy()
        labeled["difficulty_label"] = _labels_from_rates(
            labeled["overall_solve_rate"].to_numpy(float),
            float(candidate["hard_threshold"]),
            float(candidate["easy_threshold"]),
        )
        for (source, split, category), category_rows in labeled.groupby(
            ["source_dataset", "split", "category"], sort=True
        ):
            counts = category_rows["difficulty_label"].value_counts()
            status, reason = _coverage_status(
                counts,
                len(category_rows),
                policy,
                evaluation=str(split) not in {"train"},
            )
            for label in LABELS:
                cell_count = int(counts.get(label, 0))
                rows.append(
                    {
                        "threshold_id": candidate["candidate_id"],
                        "phase": phase,
                        "source": str(source),
                        "split": str(split),
                        "category": str(category),
                        "label": label,
                        "category_total": int(len(category_rows)),
                        "cell_count": cell_count,
                        "cell_share": float(cell_count / len(category_rows)),
                        "coverage_status": status,
                        "reason": reason,
                        "resolution": "",
                        "split_version": config["split_version"],
                        "policy_version": "category-coverage-v1",
                    }
                )
    return pd.DataFrame(rows)


def _candidate_red_count(train: pd.DataFrame, candidate: Mapping[str, Any], config: Mapping[str, Any]) -> int:
    report = _coverage_report(train, [candidate], config, phase="candidate_train")
    return int(
        report.loc[
            report["split"].eq("train") & report["coverage_status"].eq("red"),
            ["source", "category"],
        ].drop_duplicates().shape[0]
    )


def _run_threshold_probe(
    frame: pd.DataFrame,
    accepted_by_source: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    train = frame.loc[frame["split"].eq("train")].reset_index(drop=True)
    if train.empty:
        raise RoutingDataError("No train rows remain after deduplication")
    threshold_config = config["thresholds"]
    candidates = [
        {
            "candidate_id": f"h{hard:.2f}-e{easy:.2f}",
            "hard_threshold": float(hard),
            "easy_threshold": float(easy),
        }
        for hard in threshold_config["hard"]
        for easy in threshold_config["easy"]
    ]
    candidate_labels = {
        candidate["candidate_id"]: _labels_from_rates(
            train["overall_solve_rate"].to_numpy(float),
            candidate["hard_threshold"],
            candidate["easy_threshold"],
        )
        for candidate in candidates
    }
    predictions = {
        candidate["candidate_id"]: np.empty(len(train), dtype=object)
        for candidate in candidates
    }
    folds = list(
        GroupKFold(n_splits=int(threshold_config["oof_folds"])).split(
            train,
            groups=train["duplicate_group_id"].astype(str),
        )
    )
    tfidf_config = config["tfidf"]
    for fit_indices, predict_indices in folds:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=tuple(tfidf_config["ngram_range"]),
            min_df=int(tfidf_config["min_df"]),
            sublinear_tf=bool(tfidf_config["sublinear_tf"]),
            max_features=int(tfidf_config["max_features"]),
        )
        fit_matrix = vectorizer.fit_transform(train.loc[fit_indices, "model_input_text"])
        predict_matrix = vectorizer.transform(train.loc[predict_indices, "model_input_text"])
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            model = LogisticRegression(
                C=float(tfidf_config["C"]),
                class_weight=tfidf_config["class_weight"],
                max_iter=int(tfidf_config["max_iter"]),
                random_state=int(config["seed"]),
                solver="lbfgs",
            )
            model.fit(fit_matrix, candidate_labels[candidate_id][fit_indices])
            predictions[candidate_id][predict_indices] = model.predict(predict_matrix)

    result_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    for offset, candidate in enumerate(candidates):
        candidate_id = candidate["candidate_id"]
        agreement = _bootstrap_label_agreement(
            train,
            accepted_by_source,
            hard=candidate["hard_threshold"],
            easy=candidate["easy_threshold"],
            iterations=int(threshold_config["bootstrap_iterations"]),
            seed=int(config["seed"]) + offset,
        )
        sane, failure_reasons, distribution = _candidate_sanity(
            train, candidate_labels[candidate_id], agreement
        )
        metrics = _classification_metrics(
            candidate_labels[candidate_id],
            predictions[candidate_id],
            train["source_dataset"].to_numpy(str),
        )
        score = (
            0.25 * metrics["source_macro_f1"]
            + 0.15 * metrics["balanced_accuracy"]
            + 0.15 * metrics["qwk_01"]
            + 0.15
            * min(metrics["routerbench_macro_f1"], metrics["xroutebench_macro_f1"])
            + 0.10 * (1.0 - metrics["ordinal_mae"] / 2.0)
            + 0.10 * (1.0 - metrics["hard_to_easy_rate"])
            + 0.10 * agreement
        )
        red_count = _candidate_red_count(train, candidate, config)
        result_rows.append(
            {
                **candidate,
                **metrics,
                "candidate_model_bootstrap_label_agreement": agreement,
                "j_threshold": float(score),
                "sanity_gate_passed": bool(sane),
                "sanity_gate_failures": failure_reasons,
                "core_category_red_count": red_count,
                **distribution,
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "sample_id": train["sample_id"],
                    "source_dataset": train["source_dataset"],
                    "duplicate_group_id": train["duplicate_group_id"],
                    "candidate_id": candidate_id,
                    "difficulty_label": candidate_labels[candidate_id],
                    "oof_prediction": predictions[candidate_id],
                }
            )
        )

    passing = [row for row in result_rows if row["sanity_gate_passed"]]
    if not passing:
        raise RoutingDataError("No threshold candidate passed the label sanity gate")
    best_score = max(float(row["j_threshold"]) for row in passing)
    tied = [
        row
        for row in passing
        if best_score - float(row["j_threshold"]) < float(threshold_config["tie_tolerance"])
    ]
    anchor_id = "h0.27-e0.64"
    t1 = min(
        tied,
        key=lambda row: (
            int(row["core_category_red_count"]),
            0 if row["candidate_id"] == anchor_id else 1,
            -float(row["j_threshold"]),
        ),
    )
    t1_labels = candidate_labels[t1["candidate_id"]]
    remaining = []
    for row in passing:
        if row["candidate_id"] == t1["candidate_id"]:
            continue
        disagreement = float((candidate_labels[row["candidate_id"]] != t1_labels).mean())
        row["t1_label_disagreement"] = disagreement
        remaining.append(row)
    eligible_t2 = [
        row
        for row in remaining
        if row["t1_label_disagreement"] >= float(threshold_config["t2_minimum_disagreement"])
    ] or remaining
    if not eligible_t2:
        raise RoutingDataError("A T2 threshold candidate could not be selected")

    def _grid_distance(row: Mapping[str, Any]) -> int:
        changed = int(float(row["hard_threshold"]) != float(t1["hard_threshold"]))
        changed += int(float(row["easy_threshold"]) != float(t1["easy_threshold"]))
        return changed

    t2_best = max(float(row["j_threshold"]) for row in eligible_t2)
    t2_tied = [
        row
        for row in eligible_t2
        if math.isclose(float(row["j_threshold"]), t2_best, abs_tol=1e-12)
    ]
    t2 = min(t2_tied, key=lambda row: (_grid_distance(row), -float(row["j_threshold"])))
    selection = {
        "t1": {key: t1[key] for key in ("candidate_id", "hard_threshold", "easy_threshold", "j_threshold")},
        "t2": {
            **{key: t2[key] for key in ("candidate_id", "hard_threshold", "easy_threshold", "j_threshold")},
            "t1_label_disagreement": float(t2["t1_label_disagreement"]),
        },
        "selection_data": "train_only_group_oof",
        "routerarena_used_for_selection": False,
    }
    return result_rows, selection, pd.concat(prediction_rows, ignore_index=True)


def _apply_selected_label(frame: pd.DataFrame, selected: Mapping[str, Any], config: Mapping[str, Any]) -> pd.DataFrame:
    output = frame.copy()
    output["difficulty_label"] = _labels_from_rates(
        output["overall_solve_rate"].to_numpy(float),
        float(selected["hard_threshold"]),
        float(selected["easy_threshold"]),
    )
    output["dataset_version"] = str(config["dataset_version"])
    output["label_rule_version"] = str(config["label_rule_version"]) + ":" + str(selected["candidate_id"])
    output["preprocessing_version"] = str(config["preprocessing_version"])
    return output.loc[:, OUTPUT_COLUMNS]


def _write_manifest(output_dir: Path, config: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    artifact_paths = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema_version": "routing-comparison-dataset-v1",
        "experiment_version": config["experiment_version"],
        "dataset_version": config["dataset_version"],
        "label_rule_version": config["label_rule_version"],
        "preprocessing_version": config["preprocessing_version"],
        "split_version": config["split_version"],
        "created_at_unix": int(time.time()),
        "artifacts": {
            str(path.relative_to(output_dir)): {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifact_paths
        },
        **extra,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def build_comparison_dataset(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    xroute_data_dir: Path,
    routerarena_path: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    xconfig = config["xroutebench"]
    source_paths = {
        name: xroute_data_dir / str(source["filename"])
        for name, source in xconfig["sources"].items()
    }
    for name, path in source_paths.items():
        _verify_file(path, xconfig["sources"][name])
    candidates_frame = pd.read_parquet(source_paths["candidates"])
    candidates = candidates_frame["model_name"].astype(str).tolist()
    if len(candidates) != int(xconfig["candidate_count"]) or len(candidates) != len(set(candidates)):
        raise RoutingDataError("xRouteBench candidate pool must contain 18 unique models")

    exact_metrics = set(str(value) for value in xconfig["exact_metrics"])
    continuous_metrics = set(str(value) for value in xconfig["continuous_metrics"])
    xroute_frames: list[pd.DataFrame] = []
    xroute_accepted: list[pd.DataFrame] = []
    xroute_metrics: dict[str, Any] = {}
    for source_split in ("train", "test"):
        raw = _validate_xroute_rows(
            pd.read_parquet(source_paths[source_split]),
            split=source_split,
            candidates=candidates,
            exact_metrics=exact_metrics,
            continuous_metrics=continuous_metrics,
        )
        pivoted, accepted, metrics = _pivot_xroute(
            raw,
            source_split=source_split,
            candidates=candidates,
            minimum_valid=int(xconfig["minimum_valid_models"]),
        )
        xroute_frames.append(pivoted.loc[pivoted["eligible"]].copy())
        xroute_accepted.append(accepted.loc[pivoted.loc[pivoted["eligible"], "sample_id"]])
        xroute_metrics[source_split] = metrics
    xroute = pd.concat(xroute_frames, ignore_index=True)
    xroute_accept = pd.concat(xroute_accepted, axis=0)

    routerbench, routerbench_accept = _load_routerbench(config, project_root)
    combined = pd.concat([routerbench, xroute], ignore_index=True, sort=False)
    dedup = config["deduplication"]
    combined["duplicate_group_id"], duplicate_metrics = assign_duplicate_groups(
        combined["model_input_text"],
        jaccard_threshold=float(dedup["near_duplicate_jaccard_threshold"]),
        minimum_tokens=int(dedup["near_duplicate_min_tokens"]),
        maximum_bucket_size=int(dedup["near_duplicate_max_bucket_size"]),
    )
    xmask = combined["source_dataset"].eq("xroutebench")
    xworking = combined.loc[xmask].copy()
    _assign_xroute_validation(
        xworking,
        fraction=float(xconfig["train_validation_fraction"]),
        seed=int(config["seed"]),
        score_boundaries=[
            *config["thresholds"]["hard"],
            *config["thresholds"]["easy"],
        ],
    )
    combined.loc[xmask, "split"] = xworking["split"].to_numpy()

    pre_dedup = combined.copy()
    contamination: dict[str, Any] = {"routerarena_overlap_rows": 0}
    if routerarena_path is not None and routerarena_path.is_file():
        routerarena = pd.read_parquet(routerarena_path, columns=["sample_id", "model_input_text"])
        augmented = pd.concat(
            [
                combined.loc[:, ["sample_id", "model_input_text"]].assign(_kind="comparison"),
                routerarena.assign(
                    sample_id="routerarena:" + routerarena["sample_id"].astype(str),
                    _kind="routerarena",
                ),
            ],
            ignore_index=True,
        )
        augmented["_group"], _ = assign_duplicate_groups(
            augmented["model_input_text"],
            jaccard_threshold=float(dedup["near_duplicate_jaccard_threshold"]),
            minimum_tokens=int(dedup["near_duplicate_min_tokens"]),
            maximum_bucket_size=int(dedup["near_duplicate_max_bucket_size"]),
        )
        arena_groups = set(augmented.loc[augmented["_kind"].eq("routerarena"), "_group"])
        contaminated_ids = set(
            augmented.loc[
                augmented["_kind"].eq("comparison") & augmented["_group"].isin(arena_groups),
                "sample_id",
            ].astype(str)
        )
        contamination["routerarena_overlap_rows"] = len(contaminated_ids)
        overlap_counts = (
            combined.loc[combined["sample_id"].isin(contaminated_ids)]
            .groupby(["source_dataset", "split"])
            .size()
        )
        contamination["dropped_by_source_split"] = {
            f"{source}|{split}": int(value)
            for (source, split), value in overlap_counts.items()
        }
        combined = combined.loc[~combined["sample_id"].isin(contaminated_ids)].copy()

    combined, dedup_metrics = _deduplicate_with_priority(combined)
    accepted_by_source = {
        "routerbench": routerbench_accept.loc[
            combined.loc[combined["source_dataset"].eq("routerbench"), "sample_id"]
        ],
        "xroutebench": xroute_accept.loc[
            combined.loc[combined["source_dataset"].eq("xroutebench"), "sample_id"]
        ],
    }
    candidate_metrics, selection, oof_predictions = _run_threshold_probe(
        combined, accepted_by_source, config
    )
    t1 = selection["t1"]
    t2 = selection["t2"]
    selected_candidates = [t1, t2]
    pre_for_coverage = pre_dedup.copy()
    xroute_official_train = (
        pre_for_coverage["source_dataset"].eq("xroutebench")
        & pre_for_coverage["source_split"].eq("train")
    )
    pre_for_coverage.loc[xroute_official_train, "split"] = "train"
    coverage = pd.concat(
        [
            _coverage_report(pre_for_coverage, selected_candidates, config, phase="pre_split"),
            _coverage_report(combined, selected_candidates, config, phase="post_split"),
        ],
        ignore_index=True,
    )
    t1_data = _apply_selected_label(combined, t1, config)
    t2_data = _apply_selected_label(combined, t2, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("train", "validation", "test", "threshold", "coverage"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    t1_data.loc[t1_data["split"].eq("train")].to_parquet(
        output_dir / "train" / "training.parquet", index=False
    )
    t1_data.loc[t1_data["split"].eq("validation")].to_parquet(
        output_dir / "validation" / "validation.parquet", index=False
    )
    # T2 sensitivity notebooks receive train/validation-only projections so
    # they cannot inspect either sealed test set before selection lock.
    t2_data.loc[t2_data["split"].eq("train")].to_parquet(
        output_dir / "train" / "training-t2.parquet", index=False
    )
    t2_data.loc[t2_data["split"].eq("validation")].to_parquet(
        output_dir / "validation" / "validation-t2.parquet", index=False
    )
    t1_data.loc[
        t1_data["source_dataset"].eq("routerbench") & t1_data["split"].eq("test")
    ].to_parquet(output_dir / "test" / "routerbench.parquet", index=False)
    t1_data.loc[
        t1_data["source_dataset"].eq("xroutebench")
        & t1_data["split"].eq("external_test")
    ].to_parquet(output_dir / "test" / "xroutebench.parquet", index=False)
    t2_data.to_parquet(output_dir / "threshold" / "t2-all-splits.parquet", index=False)
    candidate_table = pd.DataFrame(candidate_metrics)
    for column in ("sanity_gate_failures", "class_counts", "class_shares", "mean_solve_rate"):
        candidate_table[column] = candidate_table[column].map(
            lambda value: json.dumps(value, sort_keys=True)
        )
    candidate_table.to_parquet(
        output_dir / "threshold" / "candidates.parquet", index=False
    )
    oof_predictions.to_parquet(
        output_dir / "threshold" / "oof-predictions.parquet", index=False
    )
    coverage.to_parquet(output_dir / "coverage" / "coverage-report.parquet", index=False)
    write_json(output_dir / "threshold" / "selected-t1-t2.json", selection)
    post_red = coverage.loc[
        coverage["phase"].eq("post_split")
        & coverage["split"].eq("train")
        & coverage["coverage_status"].eq("red"),
        ["threshold_id", "source", "split", "category", "coverage_status", "reason"],
    ].drop_duplicates()
    resolutions: list[dict[str, Any]] = []
    for red in post_red.to_dict("records"):
        pre_statuses = coverage.loc[
            coverage["phase"].eq("pre_split")
            & coverage["threshold_id"].eq(red["threshold_id"])
            & coverage["source"].eq(red["source"])
            & coverage["split"].eq("train")
            & coverage["category"].eq(red["category"]),
            "coverage_status",
        ].unique()
        pre_status = str(pre_statuses[0]) if len(pre_statuses) == 1 else "missing"
        resolution = "keep-natural" if pre_status == "red" else "resplit"
        resolutions.append(
            {
                **red,
                "pre_split_status": pre_status,
                "resolution": resolution,
                "resolved": resolution == "keep-natural",
            }
        )
    unresolved = [item for item in resolutions if not item["resolved"]]
    for resolution in resolutions:
        mask = (
            coverage["threshold_id"].eq(resolution["threshold_id"])
            & coverage["source"].eq(resolution["source"])
            & coverage["split"].eq("train")
            & coverage["category"].eq(resolution["category"])
        )
        coverage.loc[mask, "resolution"] = resolution["resolution"]
    coverage.to_parquet(output_dir / "coverage" / "coverage-report.parquet", index=False)
    write_json(
        output_dir / "coverage" / "resolution.json",
        {
            "status": "requires_resolution" if unresolved else "complete",
            "resolutions": resolutions,
            "unresolved": unresolved,
            "allowed_resolutions": [
                "keep-natural",
                "resplit",
                "collect-new-version",
                "merge-taxonomy",
            ],
        },
    )
    write_json(output_dir / "contamination-report.json", contamination)
    write_json(
        output_dir / "experiment-config.json",
        {
            **config,
            "selected_thresholds": selection,
            "source_checksums": {
                name: xconfig["sources"][name]["sha256"] for name in xconfig["sources"]
            },
        },
    )
    counts = {
        split: int(t1_data["split"].eq(split).sum())
        for split in ("train", "validation", "test", "external_test")
    }
    manifest = _write_manifest(
        output_dir,
        config,
        {
            "selected_thresholds": selection,
            "row_counts": counts,
            "source_row_counts": {
                source: int(t1_data["source_dataset"].eq(source).sum())
                for source in sorted(t1_data["source_dataset"].unique())
            },
            "coverage_resolution_status": "requires_resolution" if unresolved else "complete",
            "duplicate_metrics": duplicate_metrics,
            "deduplication": dedup_metrics,
            "xroutebench": xroute_metrics,
        },
    )
    return manifest
