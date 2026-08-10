from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from .routerarena_preprocessing import RouterArenaError


REQUIRED_EVALUATION_COLUMNS = (
    "sample_id",
    "model_input_text",
    "difficulty_label",
    "evaluation_split",
    "source_sample_id",
)


class RouterPredictor(Protocol):
    def predict(self, texts: Sequence[str]) -> pd.DataFrame: ...

    def info(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class RoutingEvaluationResult:
    predictions: pd.DataFrame
    metrics: dict[str, Any]


def _normalized_label(value: object) -> str:
    return str(value).strip().casefold()


def resolve_model_label_order(
    id2label: Mapping[int | str, str] | None,
    fallback_order: Sequence[str],
    *,
    num_labels: int,
) -> tuple[str, ...]:
    fallback = tuple(_normalized_label(label) for label in fallback_order)
    if len(fallback) != num_labels or len(set(fallback)) != num_labels:
        raise RouterArenaError(
            "Configured label_order must contain one unique label per model output"
        )
    if not id2label:
        return fallback
    normalized_mapping = {
        int(index): _normalized_label(label) for index, label in id2label.items()
    }
    if set(normalized_mapping) != set(range(num_labels)):
        raise RouterArenaError("Model id2label keys must be contiguous from zero")
    model_order = tuple(normalized_mapping[index] for index in range(num_labels))
    generic = tuple(f"label_{index}" for index in range(num_labels))
    if model_order == generic:
        return fallback
    if set(model_order) != set(fallback):
        raise RouterArenaError(
            f"Model labels {model_order} do not match configured labels {fallback}"
        )
    return model_order


class HuggingFaceRouterPredictor:
    """Batched inference for a Hugging Face sequence-classification checkpoint."""

    def __init__(
        self,
        model_path: str,
        *,
        label_order: Sequence[str],
        batch_size: int,
        max_length: int,
        device: str,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Model evaluation requires the optional 'model-evaluation' "
                "dependencies: pip install -e 'ml[model-evaluation]'"
            ) from error
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
        self._torch = torch
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = self._resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.label_order = resolve_model_label_order(
            getattr(self.model.config, "id2label", None),
            label_order,
            num_labels=int(self.model.config.num_labels),
        )

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if self._torch.cuda.is_available():
            return "cuda"
        mps = getattr(self._torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _synchronize(self) -> None:
        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize()
        elif self.device == "mps" and hasattr(self._torch, "mps"):
            self._torch.mps.synchronize()

    def predict(self, texts: Sequence[str]) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = [str(text) for text in texts[start : start + self.batch_size]]
            untruncated = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )
            token_counts = [len(input_ids) for input_ids in untruncated["input_ids"]]
            encoded = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            self._synchronize()
            started = time.perf_counter()
            with self._torch.inference_mode():
                logits = self.model(**encoded).logits
                probabilities = self._torch.softmax(logits, dim=-1).cpu().numpy()
            self._synchronize()
            per_row_latency_ms = (
                (time.perf_counter() - started) * 1000.0 / len(batch_texts)
            )
            for probability, token_count in zip(
                probabilities, token_counts, strict=True
            ):
                predicted_index = int(np.argmax(probability))
                record: dict[str, Any] = {
                    "predicted_label": self.label_order[predicted_index],
                    "prediction_confidence": float(probability[predicted_index]),
                    "input_token_count": int(token_count),
                    "was_truncated": bool(token_count > self.max_length),
                    "inference_latency_ms": per_row_latency_ms,
                }
                for index, label in enumerate(self.label_order):
                    record[f"prob_{label}"] = float(probability[index])
                records.append(record)
        return pd.DataFrame.from_records(records)

    def info(self) -> Mapping[str, Any]:
        return {
            "model_path": self.model_path,
            "model_type": getattr(self.model.config, "model_type", None),
            "num_labels": int(self.model.config.num_labels),
            "label_order": list(self.label_order),
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "device": self.device,
        }


def validate_routerarena_evaluation_input(dataframe: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_EVALUATION_COLUMNS) - set(dataframe.columns))
    if missing:
        raise RouterArenaError(f"RouterArena evaluation input is missing: {missing}")
    if dataframe.empty:
        raise RouterArenaError("RouterArena evaluation input is empty")
    if dataframe["sample_id"].duplicated().any():
        raise RouterArenaError("RouterArena evaluation input has duplicate sample_id")
    if dataframe["model_input_text"].fillna("").astype(str).str.strip().eq("").any():
        raise RouterArenaError("RouterArena evaluation input has an empty question")


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _classification_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    labels: Sequence[str],
    *,
    include_unsupported: bool,
) -> dict[str, Any]:
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    for label in labels:
        true_positive = int(((actual == label) & (predicted == label)).sum())
        false_positive = int(((actual != label) & (predicted == label)).sum())
        false_negative = int(((actual == label) & (predicted != label)).sum())
        support = int((actual == label).sum())
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        if include_unsupported or support:
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)
    row_count = len(actual)
    return {
        "row_count": row_count,
        "accuracy": float((actual == predicted).mean()) if row_count else 0.0,
        "macro_precision": float(np.mean(precision_values)) if precision_values else 0.0,
        "macro_recall": float(np.mean(recall_values)) if recall_values else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_class": per_class,
    }


def _group_metrics(
    predictions: pd.DataFrame, group_column: str, labels: Sequence[str]
) -> dict[str, Any]:
    if group_column not in predictions:
        return {}
    output: dict[str, Any] = {}
    for group, rows in predictions.groupby(group_column, dropna=False, sort=True):
        summary = _classification_metrics(
            rows["routerarena_label"],
            rows["predicted_label"],
            labels,
            include_unsupported=False,
        )
        summary["under_routing_rate"] = float(rows["is_under_routed"].mean())
        summary["over_routing_rate"] = float(rows["is_over_routed"].mean())
        output[str(group)] = summary
    return output


def _conditional_label_distribution(
    predictions: pd.DataFrame,
    *,
    group_column: str,
    label_column: str,
    groups: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    overall_counts = predictions[label_column].value_counts()
    overall_total = len(predictions)
    overall_ratios = {
        label: _safe_divide(int(overall_counts.get(label, 0)), overall_total)
        for label in labels
    }
    output: dict[str, Any] = {}
    for group in groups:
        rows = predictions[predictions[group_column].eq(group)]
        counts = rows[label_column].value_counts()
        ratios = {
            label: _safe_divide(int(counts.get(label, 0)), len(rows))
            for label in labels
        }
        output[group] = {
            "row_count": len(rows),
            "counts": {label: int(counts.get(label, 0)) for label in labels},
            "ratios": ratios,
            "overall_percentage_point_difference": {
                label: (ratios[label] - overall_ratios[label]) * 100.0
                for label in labels
            },
        }
    return {
        "overall_counts": {
            label: int(overall_counts.get(label, 0)) for label in labels
        },
        "overall_ratios": overall_ratios,
        "groups": output,
    }


def evaluate_routerarena_predictions(
    evaluation: pd.DataFrame,
    prediction_output: pd.DataFrame,
    *,
    label_order: Sequence[str],
    route_mapping: Mapping[str, str],
    evaluation_version: str,
    label_semantics: str,
    metadata: pd.DataFrame | None = None,
    model_info: Mapping[str, Any] | None = None,
) -> RoutingEvaluationResult:
    validate_routerarena_evaluation_input(evaluation)
    labels = tuple(_normalized_label(label) for label in label_order)
    if len(labels) != 3 or len(set(labels)) != 3:
        raise RouterArenaError("Router evaluation requires three unique ordered labels")
    if len(prediction_output) != len(evaluation):
        raise RouterArenaError(
            "Prediction row count does not match RouterArena evaluation input"
        )
    if "predicted_label" not in prediction_output:
        raise RouterArenaError("Prediction output is missing predicted_label")

    predictions = evaluation.reset_index(drop=True).copy()
    predictions.rename(
        columns={"difficulty_label": "routerarena_label"}, inplace=True
    )
    prediction_output = prediction_output.reset_index(drop=True).copy()
    predictions = pd.concat([predictions, prediction_output], axis=1)
    predictions["routerarena_label"] = predictions["routerarena_label"].map(
        _normalized_label
    )
    predictions["predicted_label"] = predictions["predicted_label"].map(
        _normalized_label
    )
    invalid_routerarena = sorted(
        set(predictions["routerarena_label"]) - set(labels)
    )
    invalid_predicted = sorted(set(predictions["predicted_label"]) - set(labels))
    if invalid_routerarena or invalid_predicted:
        raise RouterArenaError(
            "Invalid labels: "
            f"routerarena={invalid_routerarena}, predicted={invalid_predicted}"
        )

    label_rank = {label: index for index, label in enumerate(labels)}
    predictions["routerarena_rank"] = predictions["routerarena_label"].map(
        label_rank
    )
    predictions["predicted_rank"] = predictions["predicted_label"].map(label_rank)
    predictions["routing_delta"] = (
        predictions["predicted_rank"] - predictions["routerarena_rank"]
    )
    predictions["tier_distance"] = predictions["routing_delta"].abs()
    predictions["is_label_match"] = predictions["routing_delta"].eq(0)
    predictions["is_under_routed"] = predictions["routing_delta"].lt(0)
    predictions["is_over_routed"] = predictions["routing_delta"].gt(0)
    predictions["routing_result"] = np.select(
        [predictions["is_under_routed"], predictions["is_over_routed"]],
        ["under_routing", "over_routing"],
        default="aligned",
    )
    predictions["predicted_route"] = predictions["predicted_label"].map(
        {_normalized_label(key): value for key, value in route_mapping.items()}
    )
    if predictions["predicted_route"].isna().any():
        raise RouterArenaError("route_mapping does not cover every predicted label")

    if metadata is not None:
        metadata_columns = [
            column
            for column in (
                "sample_id",
                "domain",
                "category",
                "source_dataset",
                "label_source",
            )
            if column in metadata.columns
        ]
        if "sample_id" not in metadata_columns:
            raise RouterArenaError("RouterArena metadata is missing sample_id")
        if metadata["sample_id"].duplicated().any():
            raise RouterArenaError("RouterArena metadata has duplicate sample_id")
        metadata_projection = metadata.loc[:, metadata_columns].drop_duplicates(
            "sample_id"
        )
        predictions = predictions.merge(
            metadata_projection, on="sample_id", how="left", validate="one_to_one"
        )

    summary = _classification_metrics(
        predictions["routerarena_label"],
        predictions["predicted_label"],
        labels,
        include_unsupported=True,
    )
    confusion_matrix = {
        routerarena_label: {
            predicted_label: int(
                (
                    (predictions["routerarena_label"] == routerarena_label)
                    & (predictions["predicted_label"] == predicted_label)
                ).sum()
            )
            for predicted_label in labels
        }
        for routerarena_label in labels
    }
    row_count = len(predictions)
    metrics: dict[str, Any] = {
        "evaluation_version": evaluation_version,
        "label_semantics": label_semantics,
        "label_order": list(labels),
        "route_mapping": dict(route_mapping),
        **summary,
        "confusion_matrix": confusion_matrix,
        "under_routing_count": int(predictions["is_under_routed"].sum()),
        "under_routing_rate": float(predictions["is_under_routed"].mean()),
        "over_routing_count": int(predictions["is_over_routed"].sum()),
        "over_routing_rate": float(predictions["is_over_routed"].mean()),
        "severe_under_routing_count": int(predictions["routing_delta"].eq(-2).sum()),
        "mean_absolute_tier_error": float(predictions["tier_distance"].mean()),
        "routerarena_label_distribution": predictions["routerarena_label"]
        .value_counts()
        .to_dict(),
        "predicted_label_distribution": predictions["predicted_label"]
        .value_counts()
        .to_dict(),
        "predicted_route_distribution": predictions["predicted_route"]
        .value_counts()
        .to_dict(),
        "routerarena_labels_within_predicted_groups": (
            _conditional_label_distribution(
                predictions,
                group_column="predicted_label",
                label_column="routerarena_label",
                groups=labels,
                labels=labels,
            )
        ),
        "predicted_groups_within_routerarena_labels": (
            _conditional_label_distribution(
                predictions,
                group_column="routerarena_label",
                label_column="predicted_label",
                groups=labels,
                labels=labels,
            )
        ),
        "per_domain": _group_metrics(predictions, "domain", labels),
        "per_source_dataset": _group_metrics(
            predictions, "source_dataset", labels
        ),
        "model": dict(model_info or {}),
    }
    if "was_truncated" in predictions:
        metrics["truncated_count"] = int(predictions["was_truncated"].sum())
        metrics["truncated_rate"] = _safe_divide(
            int(predictions["was_truncated"].sum()), row_count
        )
    if "inference_latency_ms" in predictions:
        latency = predictions["inference_latency_ms"].astype(float)
        metrics["inference_latency_ms"] = {
            "mean": float(latency.mean()),
            "p50": float(latency.quantile(0.50)),
            "p95": float(latency.quantile(0.95)),
        }
    return RoutingEvaluationResult(predictions=predictions, metrics=metrics)


def run_routerarena_routing_evaluation(
    evaluation: pd.DataFrame,
    predictor: RouterPredictor,
    *,
    label_order: Sequence[str],
    route_mapping: Mapping[str, str],
    evaluation_version: str,
    label_semantics: str,
    metadata: pd.DataFrame | None = None,
) -> RoutingEvaluationResult:
    prediction_output = predictor.predict(
        evaluation["model_input_text"].astype(str).tolist()
    )
    return evaluate_routerarena_predictions(
        evaluation,
        prediction_output,
        label_order=label_order,
        route_mapping=route_mapping,
        evaluation_version=evaluation_version,
        label_semantics=label_semantics,
        metadata=metadata,
        model_info=predictor.info(),
    )
