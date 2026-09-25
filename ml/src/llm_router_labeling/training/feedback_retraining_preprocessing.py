from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..shared.dataset_io import sha256_file
from ..shared.text_normalization import normalize_for_duplicate_detection
from .routerbench_preprocessing import TRAINING_COLUMNS

REQUIRED_MANIFEST_FIELDS = (
    "schema_version",
    "batch_id",
    "created_at",
    "window_start",
    "window_end",
    "event_count",
    "data_format",
    "data_uri",
    "sha256",
    "producer_version",
    "watermark",
)

REQUIRED_EVENT_COLUMNS = (
    "event_id",
    "request_id",
    "question",
    "initial_label",
    "initial_model",
    "final_label",
    "final_model",
    "user_satisfied",
    "escalation_count",
    "created_at",
    "router_model_version",
    "pii_redacted",
)


class FeedbackDataError(RuntimeError):
    """Raised when an escalation feedback artifact violates its contract."""


@dataclass(frozen=True)
class FeedbackPreparationResult:
    training: pd.DataFrame
    metadata: pd.DataFrame
    metrics: dict[str, Any]


@dataclass(frozen=True)
class FeedbackMergeResult:
    training: pd.DataFrame
    metadata: pd.DataFrame
    metrics: dict[str, Any]


def load_feedback_events(path: Path, data_format: str) -> pd.DataFrame:
    if data_format == "parquet":
        return pd.read_parquet(path)
    if data_format == "jsonl":
        return pd.read_json(path, lines=True)
    raise FeedbackDataError(f"Unsupported feedback data format: {data_format!r}")


def validate_feedback_manifest(
    manifest: Mapping[str, Any],
    events_path: Path,
    *,
    expected_schema_version: str,
) -> None:
    missing = sorted(set(REQUIRED_MANIFEST_FIELDS) - set(manifest))
    if missing:
        raise FeedbackDataError(f"Feedback manifest is missing fields: {missing}")
    if str(manifest["schema_version"]) != expected_schema_version:
        raise FeedbackDataError(
            "Unsupported feedback schema version: "
            f"{manifest['schema_version']!r}; expected {expected_schema_version!r}"
        )
    if str(manifest["data_format"]) not in {"parquet", "jsonl"}:
        raise FeedbackDataError("Feedback data_format must be parquet or jsonl")
    for field in ("batch_id", "data_uri", "producer_version", "watermark"):
        if not str(manifest[field]).strip():
            raise FeedbackDataError(f"Feedback manifest {field} must not be empty")
    try:
        event_count = int(manifest["event_count"])
    except (TypeError, ValueError) as error:
        raise FeedbackDataError(
            "Feedback manifest event_count must be an integer"
        ) from error
    if event_count < 0:
        raise FeedbackDataError("Feedback manifest event_count must be non-negative")
    timestamps = {
        field: pd.to_datetime(manifest[field], utc=True, errors="coerce")
        for field in ("created_at", "window_start", "window_end")
    }
    invalid_timestamps = [
        field for field, value in timestamps.items() if pd.isna(value)
    ]
    if invalid_timestamps:
        raise FeedbackDataError(
            f"Feedback manifest has invalid UTC timestamps: {invalid_timestamps}"
        )
    if timestamps["window_start"] > timestamps["window_end"]:
        raise FeedbackDataError("Feedback manifest window_start exceeds window_end")
    expected_hash = str(manifest["sha256"]).removeprefix("sha256:").lower()
    actual_hash = sha256_file(events_path)
    if actual_hash != expected_hash:
        raise FeedbackDataError(
            f"Feedback events SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )


def _string(value: object) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    return str(value).strip()


def _is_boolean(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _question_hash(normalized_question: str) -> str:
    return hashlib.sha256(normalized_question.encode("utf-8")).hexdigest()


def _question_normalizer(input_contract: str):
    if input_contract == "legacy-word-only-v1":
        return normalize_for_duplicate_detection
    if input_contract == "structure-preserving-v2":
        from ..shared.structured_text import structured_duplicate_key

        return structured_duplicate_key
    raise FeedbackDataError(f"Unsupported feedback input contract: {input_contract}")


def _question_text(value: object, input_contract: str) -> str:
    if input_contract == "structure-preserving-v2" and _string(value):
        from ..shared.structured_text import clean_text

        return clean_text(value)
    return _string(value)


def canonical_training_digest(
    training: pd.DataFrame, *, input_contract: str = "legacy-word-only-v1"
) -> str:
    """Hash only the canonical learning content, independent of event identity/order."""
    normalize = _question_normalizer(input_contract)
    rows = [
        {
            "normalized_question": normalize(str(row["model_input_text"])),
            "final_label": str(row["difficulty_label"]),
            "sample_weight": float(row["sample_weight"]),
        }
        for _, row in training.iterrows()
    ]
    rows.sort(
        key=lambda row: (
            row["normalized_question"],
            row["final_label"],
            row["sample_weight"],
        )
    )
    payload = (
        rows
        if input_contract == "legacy-word-only-v1"
        else {"input_contract": input_contract, "rows": rows}
    )
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def prepare_feedback_training_dataset(
    events: pd.DataFrame,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> FeedbackPreparationResult:
    missing = sorted(set(REQUIRED_EVENT_COLUMNS) - set(events.columns))
    if missing:
        raise FeedbackDataError(f"Feedback events are missing columns: {missing}")
    expected_count = int(manifest["event_count"])
    if len(events) != expected_count:
        raise FeedbackDataError(
            f"Feedback event_count mismatch: manifest={expected_count}, rows={len(events)}"
        )

    label_order = tuple(str(label) for label in config["label_order"])
    label_rank = {label: index for index, label in enumerate(label_order)}
    if label_order != ("easy", "medium", "hard"):
        raise FeedbackDataError("Feedback label_order must be easy, medium, hard")
    acceptance = config["acceptance"]
    input_contract = str(config.get("input_contract", "legacy-word-only-v1"))
    normalize = _question_normalizer(input_contract)

    records: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    for _, row in events.reset_index(drop=True).iterrows():
        event_id = _string(row["event_id"])
        question = _question_text(row["question"], input_contract)
        normalized_question = normalize(question)
        initial_label = _string(row["initial_label"]).casefold()
        final_label = _string(row["final_label"]).casefold()
        reason = ""

        if not event_id or not _string(row["request_id"]):
            reason = "missing_event_or_request_id"
        elif event_id in seen_event_ids:
            reason = "duplicate_event_id"
        elif not question.strip() or not normalized_question.strip():
            reason = "empty_question"
        elif initial_label not in label_rank or final_label not in label_rank:
            reason = "invalid_label"
        elif not _string(row["initial_model"]) or not _string(row["final_model"]):
            reason = "missing_model"
        elif not _string(row["router_model_version"]):
            reason = "missing_router_model_version"
        elif not _is_boolean(row["user_satisfied"]):
            reason = "invalid_user_satisfied"
        elif acceptance["require_user_satisfied"] and not bool(row["user_satisfied"]):
            reason = "user_not_satisfied"
        elif not _is_boolean(row["pii_redacted"]):
            reason = "invalid_pii_redacted"
        elif acceptance["require_pii_redacted"] and not bool(row["pii_redacted"]):
            reason = "pii_not_redacted"
        else:
            escalation_count = pd.to_numeric(row["escalation_count"], errors="coerce")
            if pd.isna(escalation_count) or float(escalation_count) % 1:
                reason = "invalid_escalation_count"
            elif int(escalation_count) < int(acceptance["minimum_escalation_count"]):
                reason = "insufficient_escalation"
            elif pd.isna(pd.to_datetime(row["created_at"], utc=True, errors="coerce")):
                reason = "invalid_created_at"
            elif (
                acceptance["require_strict_tier_increase"]
                and label_rank[final_label] <= label_rank[initial_label]
            ):
                reason = "not_strictly_escalated"

        if event_id:
            seen_event_ids.add(event_id)
        records.append(
            {
                **{column: row.get(column) for column in events.columns},
                "batch_id": str(manifest["batch_id"]),
                "adapter_version": str(config["adapter_version"]),
                "normalized_question": normalized_question,
                "normalized_question_hash": (
                    _question_hash(normalized_question) if normalized_question else ""
                ),
                "normalized_initial_label": initial_label,
                "normalized_final_label": final_label,
                "acceptance_status": "accepted" if not reason else "excluded",
                "exclusion_reason": reason,
            }
        )

    metadata = pd.DataFrame.from_records(records)
    candidates = metadata[metadata["acceptance_status"].eq("accepted")].copy()
    conflict_hashes = set(
        candidates.groupby("normalized_question_hash")["normalized_final_label"]
        .nunique()
        .loc[lambda counts: counts > 1]
        .index
    )
    if conflict_hashes:
        conflict_mask = metadata["normalized_question_hash"].isin(conflict_hashes)
        metadata.loc[conflict_mask, "acceptance_status"] = "excluded"
        metadata.loc[conflict_mask, "exclusion_reason"] = "conflicting_final_labels"
        candidates = metadata[metadata["acceptance_status"].eq("accepted")].copy()

    weight_config = config["sample_weight"]
    training_records: list[dict[str, Any]] = []
    for question_hash, group in candidates.groupby(
        "normalized_question_hash", sort=True
    ):
        first = group.iloc[0]
        repeat_count = len(group)
        sample_weight = min(
            float(weight_config["maximum"]),
            float(weight_config["base"])
            + (repeat_count - 1) * float(weight_config["repeat_increment"]),
        )
        training_records.append(
            {
                "sample_id": f"feedback-{question_hash[:24]}",
                "model_input_text": _question_text(first["question"], input_contract),
                "difficulty_label": str(first["normalized_final_label"]),
                "sample_weight": sample_weight,
                "split": "train",
                "duplicate_group_id": f"feedback-{question_hash[:24]}",
            }
        )
    training = pd.DataFrame.from_records(training_records, columns=TRAINING_COLUMNS)
    exclusion_distribution = Counter(
        metadata.loc[
            metadata["acceptance_status"].eq("excluded"), "exclusion_reason"
        ].astype(str)
    )
    metrics = {
        "schema_version": str(manifest["schema_version"]),
        "adapter_version": str(config["adapter_version"]),
        "batch_id": str(manifest["batch_id"]),
        "input_event_count": len(events),
        "accepted_event_count": int(metadata["acceptance_status"].eq("accepted").sum()),
        "excluded_event_count": int(metadata["acceptance_status"].eq("excluded").sum()),
        "training_row_count": len(training),
        "input_contract": input_contract,
        "canonical_batch_digest": canonical_training_digest(
            training, input_contract=input_contract
        ),
        "exclusion_distribution": dict(sorted(exclusion_distribution.items())),
        "label_distribution": {
            str(label): int(count)
            for label, count in training["difficulty_label"]
            .value_counts()
            .sort_index()
            .items()
        },
    }
    return FeedbackPreparationResult(
        training=training, metadata=metadata, metrics=metrics
    )


def _validate_training_projection(dataframe: pd.DataFrame, name: str) -> None:
    missing = sorted(set(TRAINING_COLUMNS) - set(dataframe.columns))
    if missing:
        raise FeedbackDataError(
            f"{name} training artifact is missing columns: {missing}"
        )
    invalid_labels = sorted(
        set(dataframe["difficulty_label"].astype(str)) - {"easy", "medium", "hard"}
    )
    if invalid_labels:
        raise FeedbackDataError(
            f"{name} has invalid difficulty labels: {invalid_labels}"
        )
    invalid_splits = sorted(
        set(dataframe["split"].astype(str)) - {"train", "validation", "test"}
    )
    if invalid_splits:
        raise FeedbackDataError(f"{name} has invalid splits: {invalid_splits}")
    if dataframe["model_input_text"].fillna("").astype(str).str.strip().eq("").any():
        raise FeedbackDataError(f"{name} has an empty model_input_text")
    if dataframe["sample_id"].fillna("").astype(str).str.strip().eq("").any():
        raise FeedbackDataError(f"{name} has an empty sample_id")
    if dataframe["duplicate_group_id"].fillna("").astype(str).str.strip().eq("").any():
        raise FeedbackDataError(f"{name} has an empty duplicate_group_id")
    weights = pd.to_numeric(dataframe["sample_weight"], errors="coerce")
    if weights.isna().any() or not np.isfinite(weights).all() or weights.le(0).any():
        raise FeedbackDataError(f"{name} sample_weight must be finite and positive")


def merge_feedback_with_base_training(
    base_training: pd.DataFrame,
    feedback_training: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    training_data_strategy: str = "full_base",
) -> FeedbackMergeResult:
    merge_config = config["merge"]
    normalize = _question_normalizer(
        str(config.get("input_contract", "legacy-word-only-v1"))
    )
    if merge_config["base_train_conflict_policy"] != "replace_exact_train_rows":
        raise FeedbackDataError("Unsupported base_train_conflict_policy")
    if merge_config["base_holdout_overlap_policy"] != "exclude_feedback":
        raise FeedbackDataError("Unsupported base_holdout_overlap_policy")
    _validate_training_projection(base_training, "Base")
    _validate_training_projection(feedback_training, "Feedback")
    if set(feedback_training["split"].astype(str)) - {"train"}:
        raise FeedbackDataError("Feedback training rows must use split=train")
    group_split_counts = base_training.groupby("duplicate_group_id")["split"].nunique()
    if group_split_counts.gt(1).any():
        raise FeedbackDataError("Base duplicate groups must not cross data splits")

    base = base_training.loc[:, TRAINING_COLUMNS].copy()
    feedback = feedback_training.loc[:, TRAINING_COLUMNS].copy()
    base["_normalized_question"] = base["model_input_text"].map(normalize)
    replace_base_indices: set[int] = set()
    accepted_rows: list[dict[str, Any]] = []
    merge_records: list[dict[str, Any]] = []

    for _, row in feedback.iterrows():
        normalized = normalize(str(row["model_input_text"]))
        overlaps = base[base["_normalized_question"].eq(normalized)]
        decision = "append_new_question"
        include = True
        output_group = str(row["duplicate_group_id"])

        if not overlaps.empty:
            overlap_splits = set(overlaps["split"].astype(str))
            if overlap_splits - {"train"}:
                include = False
                decision = "excluded_base_holdout_overlap"
            elif (
                overlaps["difficulty_label"]
                .astype(str)
                .eq(str(row["difficulty_label"]))
                .any()
            ):
                include = False
                decision = "excluded_same_label_already_in_base"
            else:
                groups = sorted(overlaps["duplicate_group_id"].astype(str).unique())
                replace_base_indices.update(int(index) for index in overlaps.index)
                output_group = groups[0]
                decision = "replace_exact_base_train_rows"

        output_row = row.to_dict()
        output_row.pop("_normalized_question", None)
        output_row["duplicate_group_id"] = output_group
        if include:
            accepted_rows.append(output_row)
        merge_records.append(
            {
                "sample_id": str(row["sample_id"]),
                "difficulty_label": str(row["difficulty_label"]),
                "merge_status": "accepted" if include else "excluded",
                "merge_decision": decision,
                "output_duplicate_group_id": output_group if include else "",
            }
        )

    retained_base = base[~base.index.isin(replace_base_indices)]
    retained_base = retained_base.drop(columns=["_normalized_question"])
    replaced_base_row_count = len(base_training) - len(retained_base)
    rehearsal_input_row_count = len(retained_base)
    if training_data_strategy == "compact_rehearsal":
        retained_base = _compact_rehearsal(retained_base, merge_config)
    elif training_data_strategy != "full_base":
        raise FeedbackDataError(
            f"Unsupported training_data_strategy: {training_data_strategy!r}"
        )
    accepted_feedback = pd.DataFrame.from_records(
        accepted_rows, columns=TRAINING_COLUMNS
    )
    merged = pd.concat([retained_base, accepted_feedback], ignore_index=True)
    metadata = pd.DataFrame.from_records(merge_records)
    decision_distribution = Counter(metadata["merge_decision"].astype(str))
    metrics = {
        "merge_version": str(config["merge_version"]),
        "base_dataset_contract": config.get("base_dataset_contract", {}),
        "base_input_row_count": len(base_training),
        "feedback_input_row_count": len(feedback_training),
        "replaced_base_row_count": replaced_base_row_count,
        "training_data_strategy": training_data_strategy,
        "rehearsal_input_row_count": rehearsal_input_row_count,
        "rehearsal_output_row_count": len(retained_base),
        "accepted_feedback_row_count": len(accepted_feedback),
        "excluded_feedback_row_count": len(feedback_training) - len(accepted_feedback),
        "merged_training_row_count": len(merged),
        "merge_decision_distribution": dict(sorted(decision_distribution.items())),
        "split_distribution": {
            str(split): int(count)
            for split, count in merged["split"].value_counts().sort_index().items()
        },
    }
    return FeedbackMergeResult(training=merged, metadata=metadata, metrics=metrics)


def _compact_rehearsal(
    retained_base: pd.DataFrame,
    merge_config: Mapping[str, Any],
) -> pd.DataFrame:
    settings = merge_config.get("compact_rehearsal")
    if not isinstance(settings, Mapping):
        raise FeedbackDataError("merge.compact_rehearsal configuration is required")
    seed = int(settings["seed"])
    limits = {
        "train": int(settings["train_rows_per_label"]),
        "validation": int(settings["validation_rows_per_label"]),
    }
    if any(value < 1 for value in limits.values()):
        raise FeedbackDataError("Compact rehearsal limits must be positive")

    selected: list[pd.DataFrame] = []
    for split, limit in limits.items():
        split_rows = retained_base[retained_base["split"].eq(split)]
        for label in ("easy", "medium", "hard"):
            candidates = split_rows[split_rows["difficulty_label"].eq(label)].copy()
            if candidates.empty:
                raise FeedbackDataError(
                    f"Compact rehearsal has no {split}/{label} base rows"
                )
            candidates["_rehearsal_rank"] = candidates["sample_id"].map(
                lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
            )
            selected.append(
                candidates.sort_values("_rehearsal_rank", kind="stable")
                .head(limit)
                .drop(columns=["_rehearsal_rank"])
            )
    return pd.concat(selected, ignore_index=True).loc[:, TRAINING_COLUMNS]
