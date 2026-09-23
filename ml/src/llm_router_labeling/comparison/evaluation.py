from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

LABELS = ("easy", "medium", "hard")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


class ComparisonEvaluationError(ValueError):
    """Raised when prediction inputs violate the comparison contract."""


def _ece(actual_ids: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == actual_ids
    bin_ids = np.minimum((confidence * bins).astype(int), bins - 1)
    value = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if mask.any():
            value += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return float(value)


def _coverage_status(actual: pd.Series) -> str:
    counts = actual.value_counts().reindex(LABELS, fill_value=0)
    minimum = int(counts.min())
    if minimum >= 100:
        return "confirmatory"
    if minimum >= 30:
        return "exploratory"
    return "insufficient_coverage"


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual_ids = np.array([LABEL_TO_ID[str(value)] for value in actual], dtype=int)
    predicted_ids = np.array([LABEL_TO_ID[str(value)] for value in predicted], dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=list(LABELS),
        zero_division=0,
    )
    qwk = 0.0
    if np.unique(actual_ids).size >= 2:
        qwk = float(
            cohen_kappa_score(
                actual_ids, predicted_ids, labels=[0, 1, 2], weights="quadratic"
            )
        )
        if not np.isfinite(qwk):
            qwk = 0.0
    supported = support > 0
    return {
        "row_count": int(len(actual)),
        "accuracy": float((actual == predicted).mean()),
        "macro_f1": float(f1_score(actual, predicted, labels=list(LABELS), average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(actual, predicted, labels=list(LABELS), average="weighted", zero_division=0)
        ),
        "balanced_accuracy": float(recall[supported].mean()) if supported.any() else 0.0,
        "qwk": qwk,
        "qwk_01": float((qwk + 1.0) / 2.0),
        "ordinal_mae": float(np.abs(actual_ids - predicted_ids).mean()),
        "under_routing_rate": float((predicted_ids < actual_ids).mean()),
        "over_routing_rate": float((predicted_ids > actual_ids).mean()),
        "hard_to_easy_rate": float(
            ((actual == "hard") & (predicted == "easy")).sum()
            / max(int((actual == "hard").sum()), 1)
        ),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=list(LABELS)
        ).astype(int).tolist(),
    }


def evaluate_classification_predictions(
    validation: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    latency_ms: Sequence[float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"sample_id", "difficulty_label", "source_dataset", "category"}
    missing = sorted(required - set(validation.columns))
    if missing:
        raise ComparisonEvaluationError(f"Validation data is missing columns: {missing}")
    if validation.empty or validation["sample_id"].duplicated().any():
        raise ComparisonEvaluationError("Validation rows must be non-empty and unique")
    if not validation["difficulty_label"].isin(LABELS).all():
        raise ComparisonEvaluationError("Validation data contains an invalid label")
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(validation), len(LABELS)):
        raise ComparisonEvaluationError(
            f"Expected probability shape {(len(validation), len(LABELS))}, got {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-5
    ):
        raise ComparisonEvaluationError("Probabilities must be finite and sum to one")

    actual = validation["difficulty_label"].astype(str).to_numpy()
    predicted_ids = probabilities.argmax(axis=1)
    predicted = np.array([LABELS[index] for index in predicted_ids], dtype=object)
    actual_ids = np.array([LABEL_TO_ID[value] for value in actual], dtype=int)
    predictions = validation.loc[
        :, ["sample_id", "source_dataset", "category", "difficulty_label"]
    ].copy()
    predictions.rename(columns={"difficulty_label": "actual_label"}, inplace=True)
    predictions["predicted_label"] = predicted
    for index, label in enumerate(LABELS):
        predictions[f"prob_{label}"] = probabilities[:, index]
    predictions["confidence"] = probabilities.max(axis=1)
    predictions["is_correct"] = predicted_ids == actual_ids
    predictions["is_under_routed"] = predicted_ids < actual_ids
    predictions["is_over_routed"] = predicted_ids > actual_ids
    predictions["is_hard_to_easy"] = (actual == "hard") & (predicted == "easy")

    overall = _metrics(actual, predicted)
    overall["ece"] = _ece(actual_ids, probabilities)
    source_metrics = {
        str(source): _metrics(
            rows["actual_label"].to_numpy(str), rows["predicted_label"].to_numpy(str)
        )
        for source, rows in predictions.groupby("source_dataset", sort=True)
    }
    overall["source_macro_f1"] = float(
        np.mean([metrics["macro_f1"] for metrics in source_metrics.values()])
    )
    overall["q_model"] = float(
        0.45 * overall["source_macro_f1"]
        + 0.15 * overall["balanced_accuracy"]
        + 0.15 * overall["qwk_01"]
        + 0.15 * (1.0 - overall["hard_to_easy_rate"])
        + 0.10 * (1.0 - overall["ece"])
    )
    category_metrics: dict[str, Any] = {}
    for (source, category), rows in predictions.groupby(
        ["source_dataset", "category"], sort=True
    ):
        key = f"{source}|{category}"
        category_metrics[key] = {
            "coverage_status": _coverage_status(rows["actual_label"]),
            **_metrics(
                rows["actual_label"].to_numpy(str),
                rows["predicted_label"].to_numpy(str),
            ),
        }
    metrics: dict[str, Any] = {
        "label_order": list(LABELS),
        "overall": overall,
        "per_source": source_metrics,
        "per_source_category": category_metrics,
    }
    if latency_ms is not None:
        latency = np.asarray(list(latency_ms), dtype=float)
        if latency.size:
            metrics["batch_1_latency_ms"] = {
                "sample_count": int(latency.size),
                "mean": float(latency.mean()),
                "p50": float(np.quantile(latency, 0.50)),
                "p95": float(np.quantile(latency, 0.95)),
            }
    return predictions, metrics
