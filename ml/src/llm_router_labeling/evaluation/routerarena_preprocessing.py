from __future__ import annotations

import ast
import hashlib
import html
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..shared.dataset_io import verify_file
from ..shared.text_normalization import normalize_for_duplicate_detection


class RouterArenaError(RuntimeError):
    """Raised when RouterArena data violates the evaluation contract."""


REQUIRED_ROUTERARENA_COLUMNS = (
    "Category",
    "Domain",
    "Dataset name",
    "Global Index",
    "Context",
    "Question",
    "Options",
    "Answer",
    "Metadata",
    "Keywords",
    "Difficulty",
)

EVALUATION_COLUMNS = (
    "sample_id",
    "model_input_text",
    "difficulty_label",
    "evaluation_split",
    "source_sample_id",
)
ROUTERARENA_DIFFICULTY_LABELS = frozenset({"easy", "medium", "hard"})

_REFUSAL_PLACEHOLDER = re.compile(
    r"^i(?:'m| am) sorry,?(?: but)? i (?:can't|cannot) assist with that request\.?$",
    flags=re.IGNORECASE,
)
_APPENDED_BOXED_ANSWER = re.compile(
    r"\n\s*\n\\boxed\{\{[^}]",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class PreparedEvaluationDataset:
    evaluation: pd.DataFrame
    metadata: pd.DataFrame
    metrics: dict[str, Any]


def validate_routerarena_schema(dataframe: pd.DataFrame) -> None:
    missing = [
        column for column in REQUIRED_ROUTERARENA_COLUMNS if column not in dataframe
    ]
    if missing:
        raise RouterArenaError(f"Missing required RouterArena columns: {missing}")


def load_routerarena_source(
    path: Path, source_config: Mapping[str, Any]
) -> pd.DataFrame:
    verify_file(path, source_config)
    dataframe = pd.read_parquet(path)
    validate_routerarena_schema(dataframe)
    expected_rows = int(source_config["row_count"])
    if len(dataframe) != expected_rows:
        raise RouterArenaError(
            f"Unexpected RouterArena row count: expected {expected_rows}, "
            f"got {len(dataframe)}"
        )
    return dataframe


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = html.unescape(unicodedata.normalize("NFKC", str(value)))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        character
        for character in text
        if character in "\n\t" or not unicodedata.category(character).startswith("C")
    )
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    compact: list[str] = []
    for line in lines:
        if not line and (not compact or not compact[-1]):
            continue
        compact.append(line)
    while compact and not compact[-1]:
        compact.pop()
    return "\n".join(compact)


def parse_options(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    parsed: object = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError, TypeError):
                parsed = [stripped]
    if isinstance(parsed, np.ndarray):
        parsed = parsed.tolist()
    if not isinstance(parsed, (list, tuple)):
        parsed = [parsed]
    return [cleaned for item in parsed if (cleaned := _clean_text(item))]


def build_routerarena_model_input(
    context: object, question: object, options: object
) -> str:
    cleaned_context = _clean_text(context)
    cleaned_question = _clean_text(question)
    if not cleaned_question:
        return ""
    parts = []
    if cleaned_context:
        parts.append(f"Context:\n{cleaned_context}")
    parts.append(cleaned_question)
    cleaned_options = parse_options(options)
    if cleaned_options:
        option_lines = []
        for index, option in enumerate(cleaned_options):
            label = chr(ord("A") + index) if index < 26 else str(index + 1)
            option_lines.append(f"{label}) {option}")
        parts.append("\n".join(option_lines))
    return "\n\n".join(parts)


def _validated_difficulty_labels(values: pd.Series) -> pd.Series:
    labels = values.fillna("").astype(str).str.strip().str.casefold()
    invalid = ~labels.isin(ROUTERARENA_DIFFICULTY_LABELS)
    if invalid.any():
        unknown = sorted(labels[invalid].unique().tolist())
        raise RouterArenaError(f"Unknown RouterArena difficulty labels: {unknown}")
    return labels


def _content_hash(text: str) -> str:
    normalized = normalize_for_duplicate_detection(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _canonical_metadata(
    dataframe: pd.DataFrame,
    *,
    source_split: str,
    reference_full: pd.DataFrame | None,
) -> pd.DataFrame:
    validate_routerarena_schema(dataframe)
    source = dataframe.copy()
    source["Global Index"] = source["Global Index"].astype(str)
    if source["Global Index"].duplicated().any():
        raise RouterArenaError(f"RouterArena {source_split} has duplicate Global Index")

    reference_inputs: pd.Series | None = None
    if source_split == "robustness":
        if reference_full is None:
            raise RouterArenaError("RouterArena robustness requires the full split")
        validate_routerarena_schema(reference_full)
        reference = reference_full.copy()
        reference["Global Index"] = reference["Global Index"].astype(str)
        if reference["Global Index"].duplicated().any():
            raise RouterArenaError("RouterArena full split has duplicate Global Index")
        reference = reference.set_index("Global Index")
        missing_reference = sorted(set(source["Global Index"]) - set(reference.index))
        if missing_reference:
            raise RouterArenaError(
                "Robustness rows missing from full split: "
                f"{missing_reference[:5]}"
            )
        source_index = source["Global Index"]
        for column in (
            "Category",
            "Domain",
            "Dataset name",
            "Answer",
            "Metadata",
            "Keywords",
            "Difficulty",
        ):
            source[column] = source_index.map(reference[column])
        reference_inputs = source_index.map(
            reference.apply(
                lambda row: build_routerarena_model_input(
                    row["Context"], row["Question"], row["Options"]
                ),
                axis=1,
            )
        )

    metadata = pd.DataFrame(
        {
            "sample_id": source["Global Index"].map(
                lambda value: f"routerarena:{source_split}:{value}"
            ),
            "source_sample_id": source["Global Index"],
            "evaluation_split": source_split,
            "category": source["Category"].map(_clean_text),
            "domain": source["Domain"].map(_clean_text),
            "source_dataset": source["Dataset name"].map(_clean_text),
            "context": source["Context"].map(_clean_text),
            "question": source["Question"].map(_clean_text),
            "options": source["Options"].map(parse_options),
            "answer": source["Answer"].map(_clean_text),
            "source_metadata": source["Metadata"].map(_clean_text),
            "keywords": source["Keywords"].map(_clean_text),
            "difficulty_label": _validated_difficulty_labels(source["Difficulty"]),
        }
    )
    metadata["model_input_text"] = [
        build_routerarena_model_input(context, question, options)
        for context, question, options in zip(
            source["Context"], source["Question"], source["Options"], strict=True
        )
    ]
    metadata["reference_model_input_text"] = (
        reference_inputs.values if reference_inputs is not None else pd.NA
    )
    metadata["label_source"] = (
        "routerarena_official_difficulty_full_join"
        if source_split == "robustness"
        else "routerarena_official_difficulty"
    )
    metadata["content_hash"] = metadata["model_input_text"].map(_content_hash)
    metadata["character_count"] = metadata["model_input_text"].str.len()
    metadata["word_count"] = metadata["model_input_text"].map(
        lambda text: len(re.findall(r"\w+", text, flags=re.UNICODE))
    )
    metadata["exclusion_reason"] = ""
    metadata.loc[metadata["question"].eq(""), "exclusion_reason"] = (
        "missing_question"
    )
    refusal = metadata["question"].map(
        lambda question: bool(_REFUSAL_PLACEHOLDER.fullmatch(question.strip()))
    )
    metadata.loc[
        refusal & metadata["exclusion_reason"].eq(""), "exclusion_reason"
    ] = "refusal_placeholder"
    appended_answer = metadata["question"].map(
        lambda question: bool(_APPENDED_BOXED_ANSWER.search(question))
    )
    metadata.loc[
        appended_answer & metadata["exclusion_reason"].eq(""), "exclusion_reason"
    ] = "embedded_answer"
    return metadata


def _training_hashes(training: pd.DataFrame) -> set[str]:
    if "model_input_text" in training:
        texts = training["model_input_text"]
    elif "query_text" in training:
        texts = training["query_text"]
    else:
        raise RouterArenaError(
            "Training overlap input needs model_input_text or query_text"
        )
    return {
        _content_hash(_clean_text(value))
        for value in texts
        if _clean_text(value)
    }


def stratified_evaluation_sample(
    dataframe: pd.DataFrame, sample_size: int, seed: int
) -> pd.Index:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size >= len(dataframe):
        return dataframe.index
    groups = ["difficulty_label", "domain"]
    covered = (
        dataframe.groupby(groups, group_keys=False, sort=True)
        .sample(n=1, random_state=seed)
        .drop_duplicates("sample_id")
    )
    if len(covered) > sample_size:
        covered = covered.sample(n=sample_size, random_state=seed)
    remaining_size = sample_size - len(covered)
    if remaining_size:
        remaining = dataframe.drop(index=covered.index).sample(
            n=remaining_size, random_state=seed
        )
        covered = pd.concat([covered, remaining])
    return covered.sort_values("sample_id", kind="stable").index


def _distribution(dataframe: pd.DataFrame, columns: Sequence[str]) -> dict[str, int]:
    if dataframe.empty:
        return {}
    counts = dataframe.groupby(list(columns), dropna=False).size()
    return {
        "|".join(str(part) for part in key)
        if isinstance(key, tuple)
        else str(key): int(value)
        for key, value in counts.items()
    }


def prepare_routerarena_evaluation(
    dataframe: pd.DataFrame,
    preprocessing_config: Mapping[str, Any],
    *,
    source_split: str,
    dataset_version: str,
    preprocessing_version: str,
    reference_full: pd.DataFrame | None = None,
    training: pd.DataFrame | None = None,
    sample_size: int | None = None,
    seed: int = 42,
) -> PreparedEvaluationDataset:
    if source_split not in {"full", "sub_10", "robustness"}:
        raise ValueError(f"Unsupported RouterArena split: {source_split}")
    metadata = _canonical_metadata(
        dataframe,
        source_split=source_split,
        reference_full=reference_full,
    )
    if bool(preprocessing_config.get("exclude_exact_source_duplicates", True)):
        candidates = metadata.loc[metadata["exclusion_reason"].eq("")].sort_values(
            "sample_id", kind="stable"
        )
        duplicate_indexes = candidates.index[
            candidates.duplicated("content_hash", keep="first")
        ]
        metadata.loc[duplicate_indexes, "exclusion_reason"] = "duplicate_content"
    overlap_check = "not_run"
    metadata["training_overlap"] = False
    if training is not None:
        overlap_check = "exact_normalized_text_hash"
        hashes = _training_hashes(training)
        metadata["training_overlap"] = metadata["content_hash"].isin(hashes)
        if bool(preprocessing_config.get("exclude_exact_training_overlap", True)):
            overlap = metadata["training_overlap"] & metadata["exclusion_reason"].eq("")
            metadata.loc[overlap, "exclusion_reason"] = "exact_training_overlap"

    eligible = metadata.loc[metadata["exclusion_reason"].eq("")]
    selected_index = (
        eligible.index
        if sample_size is None
        else stratified_evaluation_sample(eligible, sample_size, seed)
    )
    metadata["selected_for_evaluation"] = metadata.index.isin(selected_index)
    metadata["dataset_version"] = dataset_version
    metadata["preprocessing_version"] = preprocessing_version
    selected = metadata.loc[selected_index].sort_values("sample_id", kind="stable")
    evaluation = selected.loc[:, list(EVALUATION_COLUMNS)].reset_index(drop=True)
    metadata = metadata.reset_index(drop=True)
    validate_routerarena_output(evaluation)

    duplicate_counts = eligible["content_hash"].value_counts()
    metrics: dict[str, Any] = {
        "dataset_version": dataset_version,
        "preprocessing_version": preprocessing_version,
        "source_split": source_split,
        "seed": seed,
        "requested_sample_size": sample_size,
        "input_row_count": int(len(metadata)),
        "eligible_row_count": int(len(eligible)),
        "evaluation_row_count": int(len(evaluation)),
        "excluded_row_count": int(metadata["exclusion_reason"].ne("").sum()),
        "exclusion_distribution": _distribution(
            metadata.loc[metadata["exclusion_reason"].ne("")],
            ["exclusion_reason"],
        ),
        "label_distribution": _distribution(selected, ["difficulty_label"]),
        "domain_distribution": _distribution(selected, ["domain"]),
        "source_dataset_count": int(selected["source_dataset"].nunique()),
        "exact_duplicate_row_count": int(
            (duplicate_counts - 1).clip(lower=0).sum()
        ),
        "training_overlap_check": overlap_check,
        "training_overlap_count": int(metadata["training_overlap"].sum()),
        "empty_model_input_count": int(selected["model_input_text"].eq("").sum()),
        "character_count": {
            "min": int(selected["character_count"].min()),
            "median": float(selected["character_count"].median()),
            "max": int(selected["character_count"].max()),
        },
    }
    return PreparedEvaluationDataset(
        evaluation=evaluation, metadata=metadata, metrics=metrics
    )


def validate_routerarena_output(evaluation: pd.DataFrame) -> None:
    if tuple(evaluation.columns) != EVALUATION_COLUMNS:
        raise RouterArenaError(
            f"RouterArena evaluation projection must be {list(EVALUATION_COLUMNS)}"
        )
    if evaluation["sample_id"].duplicated().any():
        raise RouterArenaError("RouterArena evaluation has duplicate sample_id")
    if evaluation["model_input_text"].eq("").any():
        raise RouterArenaError("RouterArena evaluation has empty model_input_text")
    if not evaluation["difficulty_label"].isin({"easy", "medium", "hard"}).all():
        raise RouterArenaError("RouterArena evaluation has an invalid label")
