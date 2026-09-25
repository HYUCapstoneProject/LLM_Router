"""Label evaluation for frozen RB+xRB partitions, independent of RouterArena."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .routerarena_routing_evaluator import (
    RouterPredictor,
    RoutingEvaluationResult,
    _classification_metrics,
)

LABELS = ("easy", "medium", "hard")
CONTRACT_COLUMNS = (
    "dataset_version",
    "label_rule_version",
    "preprocessing_version",
    "split_version",
)


def validate_continual_evaluation(
    frame: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Fail before inference when a different partition or data contract is used."""
    required = {
        "sample_id",
        "model_input_text",
        "difficulty_label",
        "duplicate_group_id",
        "source_dataset",
        "category",
        "split",
        *CONTRACT_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"Continual evaluation is empty or missing columns: {missing}")
    expected_split = config["evaluation_split"]
    expected_role = {
        "gate": "continual_gate",
        "validation": "development_calibration",
    }.get(expected_split)
    if expected_role is None or config.get("evaluation_role") != expected_role:
        raise ValueError("Continual evaluation role and partition disagree")
    if set(frame["split"].astype(str)) != {expected_split}:
        raise ValueError("Evaluation partition does not match the frozen config")
    for key in CONTRACT_COLUMNS:
        if set(frame[key].astype(str)) != {str(config["dataset_contract"][key])}:
            raise ValueError(f"Evaluation data contract mismatch: {key}")
    if tuple(config["label_order"]) != LABELS:
        raise ValueError("Label order must be easy, medium, hard")
    if set(frame["difficulty_label"].astype(str)) != set(LABELS):
        raise ValueError("Frozen evaluation must contain all three labels")
    if set(frame["source_dataset"].astype(str)) != {"routerbench", "xroutebench"}:
        raise ValueError("Frozen evaluation must contain RouterBench and xRouteBench")
    if frame["sample_id"].duplicated().any() or frame["sample_id"].isna().any():
        raise ValueError("Evaluation sample IDs must be unique and non-null")
    if frame["model_input_text"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Evaluation contains an empty question")
    if frame["duplicate_group_id"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Evaluation contains an empty duplicate group")
    output = (
        frame.loc[
            :,
            [
                "sample_id",
                "model_input_text",
                "difficulty_label",
                "duplicate_group_id",
                "source_dataset",
                "category",
                "split",
            ],
        ]
        .reset_index(drop=True)
        .copy()
    )
    mapping = config["category_to_domain"]
    keys = output["source_dataset"].astype(str) + ":" + output["category"].astype(str)
    output["domain"] = keys.map(mapping)
    if output["domain"].isna().any():
        unknown = sorted(keys[output["domain"].isna()].unique())
        raise ValueError(f"Domain mapping must be fixed before evaluation: {unknown}")
    return output


def _summary(rows: pd.DataFrame, *, include_unsupported: bool = True) -> dict[str, Any]:
    metrics = _classification_metrics(
        rows["difficulty_label"],
        rows["predicted_label"],
        LABELS,
        include_unsupported=include_unsupported,
    )
    rank = {label: index for index, label in enumerate(LABELS)}
    delta = rows["predicted_label"].map(rank) - rows["difficulty_label"].map(rank)
    hard = rows["difficulty_label"].eq("hard")
    metrics.update(
        {
            "balanced_accuracy": metrics["macro_recall"],
            "mean_absolute_label_error": float(delta.abs().mean()),
            "under_label_rate": float(delta.lt(0).mean()),
            "over_label_rate": float(delta.gt(0).mean()),
            "hard_to_easy_count": int(
                (hard & rows["predicted_label"].eq("easy")).sum()
            ),
            "hard_to_easy_rate": float(
                rows.loc[hard, "predicted_label"].eq("easy").mean()
            )
            if hard.any()
            else None,
            "confusion_matrix": {
                actual: {
                    predicted: int(
                        (
                            rows["difficulty_label"].eq(actual)
                            & rows["predicted_label"].eq(predicted)
                        ).sum()
                    )
                    for predicted in LABELS
                }
                for actual in LABELS
            },
        }
    )
    return metrics


def evaluate_continual_predictions(
    evaluation: pd.DataFrame,
    prediction_output: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    model_info: Mapping[str, Any] | None = None,
) -> RoutingEvaluationResult:
    rows = validate_continual_evaluation(evaluation, config)
    if (
        len(prediction_output) != len(rows)
        or "predicted_label" not in prediction_output
    ):
        raise ValueError("Prediction rows do not match evaluation rows")
    if set(prediction_output["predicted_label"].astype(str)) - set(LABELS):
        raise ValueError("Predictions contain an invalid label")
    if set(prediction_output.columns) & set(rows.columns):
        raise ValueError("Prediction output must not overwrite evaluation metadata")
    predictions = pd.concat([rows, prediction_output.reset_index(drop=True)], axis=1)
    metrics = {
        "evaluation_version": str(config["evaluation_version"]),
        "evaluation_role": str(config["evaluation_role"]),
        "evaluation_split": str(config["evaluation_split"]),
        "label_semantics": "overall-solve-rate-T1-label-agreement",
        "dataset_contract": dict(config["dataset_contract"]),
        "dataset_sha256": str(config["evaluation_dataset_sha256"]),
        "label_order": list(LABELS),
        "group_metric_policy": "macro_over_labels_with_true_support",
        **_summary(predictions),
        "per_source_dataset": {
            str(name): _summary(group, include_unsupported=False)
            for name, group in predictions.groupby("source_dataset", sort=True)
        },
        "per_domain": {
            str(name): _summary(group, include_unsupported=False)
            for name, group in predictions.groupby("domain", sort=True)
        },
        "model_info": dict(model_info or {}),
    }
    if "was_truncated" in predictions:
        metrics["truncated_row_count"] = int(predictions["was_truncated"].sum())
    return RoutingEvaluationResult(predictions=predictions, metrics=metrics)


def run_continual_evaluation(
    evaluation: pd.DataFrame, predictor: RouterPredictor, config: Mapping[str, Any]
) -> RoutingEvaluationResult:
    rows = validate_continual_evaluation(evaluation, config)
    output = predictor.predict(rows["model_input_text"].astype(str).tolist())
    return evaluate_continual_predictions(
        evaluation, output, config, model_info=predictor.info()
    )
