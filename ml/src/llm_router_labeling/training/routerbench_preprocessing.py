from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from ..shared.text_normalization import normalize_for_duplicate_detection
from .routerbench_labeling import RouterBenchError


TRAINING_COLUMNS = (
    "sample_id",
    "model_input_text",
    "difficulty_label",
    "sample_weight",
    "split",
    "duplicate_group_id",
)

_LEADING_BENCHMARK_INSTRUCTIONS = (
    "please answer with the letter of the correct answer.",
    "please answer with the letter of the correct answer",
)
_TRAILING_CHOICE_INSTRUCTION = re.compile(
    r"^print\s+only\s+a\s+single\s+choice\s+from\s+.+"
    r"without\s+explanation\.?$",
    flags=re.IGNORECASE,
)
_WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class PreparedDataset:
    training: pd.DataFrame
    metadata: pd.DataFrame
    metrics: dict[str, Any]


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def clean_model_input_text(query_text: object) -> tuple[str, bool]:
    """Remove RouterBench-only wrappers while preserving task content."""
    if query_text is None or (
        isinstance(query_text, float) and math.isnan(query_text)
    ):
        return "", False
    text = unicodedata.normalize("NFKC", str(query_text))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        character
        for character in text
        if character in "\n\t" or not unicodedata.category(character).startswith("C")
    )
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    removed = False
    if lines and lines[0].casefold() in _LEADING_BENCHMARK_INSTRUCTIONS:
        lines.pop(0)
        removed = True
        while lines and not lines[0]:
            lines.pop(0)
    if lines and _TRAILING_CHOICE_INSTRUCTION.fullmatch(lines[-1]):
        lines.pop()
        removed = True
        while lines and not lines[-1]:
            lines.pop()

    normalized_lines: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and previous_blank:
            continue
        normalized_lines.append(line)
        previous_blank = is_blank
    return "\n".join(normalized_lines).strip(), removed


def _content_tokens(normalized_text: str) -> frozenset[str]:
    words = normalized_text.split()
    if len(words) < 3:
        return frozenset(words)
    return frozenset(
        " ".join(words[index : index + 3]) for index in range(len(words) - 2)
    )


def _stable_u64(value: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big"
    )


def _simhash(features: frozenset[str]) -> int:
    if not features:
        return 0
    scores = [0] * 64
    for feature in features:
        hashed = _stable_u64(feature)
        for bit in range(64):
            scores[bit] += 1 if hashed & (1 << bit) else -1
    fingerprint = 0
    for bit, score in enumerate(scores):
        if score >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def assign_duplicate_groups(
    texts: pd.Series,
    *,
    jaccard_threshold: float,
    minimum_tokens: int,
    maximum_bucket_size: int,
) -> tuple[pd.Series, dict[str, int]]:
    if not 0.0 <= jaccard_threshold <= 1.0:
        raise ValueError("jaccard_threshold must be in [0, 1]")
    normalized = texts.astype(str).map(normalize_for_duplicate_detection)
    if normalized.eq("").any():
        raise RouterBenchError("Preprocessing produced an empty duplicate key")

    unique_texts = list(dict.fromkeys(normalized.tolist()))
    unique_index = {text: index for index, text in enumerate(unique_texts)}
    features = [_content_tokens(text) for text in unique_texts]
    token_counts = [len(text.split()) for text in unique_texts]
    fingerprints = [_simhash(feature_set) for feature_set in features]
    disjoint = _DisjointSet(len(unique_texts))
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, (fingerprint, token_count) in enumerate(
        zip(fingerprints, token_counts, strict=True)
    ):
        length_bucket = token_count // 10
        for band in range(4):
            band_value = (fingerprint >> (band * 16)) & 0xFFFF
            for adjacent_bucket in range(max(0, length_bucket - 1), length_bucket + 2):
                buckets[(band, band_value, adjacent_bucket)].append(index)

    candidate_pairs: set[tuple[int, int]] = set()
    skipped_large_bucket_count = 0
    for members in buckets.values():
        distinct_members = sorted(set(members))
        if len(distinct_members) > maximum_bucket_size:
            skipped_large_bucket_count += 1
            continue
        for left_position, left in enumerate(distinct_members):
            for right in distinct_members[left_position + 1 :]:
                candidate_pairs.add((left, right))

    near_duplicate_pair_count = 0
    for left, right in sorted(candidate_pairs):
        if min(token_counts[left], token_counts[right]) < minimum_tokens:
            continue
        length_ratio = min(token_counts[left], token_counts[right]) / max(
            token_counts[left], token_counts[right]
        )
        if length_ratio < jaccard_threshold:
            continue
        if _jaccard(features[left], features[right]) >= jaccard_threshold:
            disjoint.union(left, right)
            near_duplicate_pair_count += 1

    cluster_members: dict[int, list[str]] = defaultdict(list)
    for index, text in enumerate(unique_texts):
        cluster_members[disjoint.find(index)].append(text)
    cluster_ids = {
        root: "dup-" + hashlib.sha256(min(members).encode("utf-8")).hexdigest()[:16]
        for root, members in cluster_members.items()
    }
    groups = normalized.map(
        lambda text: cluster_ids[disjoint.find(unique_index[text])]
    )
    group_sizes = groups.value_counts()
    return groups, {
        "duplicate_group_count": int(group_sizes.size),
        "duplicate_row_count": int((group_sizes - 1).clip(lower=0).sum()),
        "near_duplicate_pair_count": near_duplicate_pair_count,
        "near_duplicate_candidate_pair_count": len(candidate_pairs),
        "near_duplicate_skipped_large_bucket_count": skipped_large_bucket_count,
    }


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_group_splits(
    dataframe: pd.DataFrame,
    split_fractions: Mapping[str, float],
    seed: int,
) -> pd.Series:
    split_names = tuple(split_fractions)
    fractions = {name: float(split_fractions[name]) for name in split_names}
    if len(split_names) < 2 or any(value <= 0 for value in fractions.values()):
        raise ValueError("At least two positive split fractions are required")
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Split fractions must sum to 1.0")

    strata = (
        dataframe["difficulty_label"].astype(str)
        + "|"
        + dataframe["domain"].astype(str)
    )
    working = dataframe.loc[:, ["duplicate_group_id"]].copy()
    working["stratum"] = strata
    group_strata = (
        working.groupby(["duplicate_group_id", "stratum"], sort=True)
        .size()
        .unstack(fill_value=0)
    )
    stratum_totals = group_strata.sum(axis=0).to_dict()
    total_rows = int(len(dataframe))
    target_by_split = {
        split: {
            stratum: float(total) * fractions[split]
            for stratum, total in stratum_totals.items()
        }
        for split in split_names
    }
    total_target = {split: total_rows * fractions[split] for split in split_names}
    assigned_counts = {
        split: Counter({stratum: 0 for stratum in stratum_totals})
        for split in split_names
    }
    assigned_totals = Counter({split: 0 for split in split_names})
    group_assignment: dict[str, str] = {}

    ordered_groups = sorted(
        group_strata.index,
        key=lambda group: (
            -int(group_strata.loc[group].sum()),
            _stable_fraction(str(group), seed),
        ),
    )
    for group in ordered_groups:
        counts = group_strata.loc[group]
        group_size = int(counts.sum())
        scored_splits: list[tuple[float, float, str]] = []
        for split in split_names:
            stratum_need = sum(
                int(count)
                * max(
                    target_by_split[split][stratum]
                    - assigned_counts[split][stratum],
                    0.0,
                )
                / max(target_by_split[split][stratum], 1.0)
                for stratum, count in counts.items()
                if count
            )
            total_need = group_size * max(
                total_target[split] - assigned_totals[split], 0.0
            ) / max(total_target[split], 1.0)
            overflow = max(
                assigned_totals[split] + group_size - total_target[split], 0.0
            ) / max(total_target[split], 1.0)
            score = stratum_need + 0.25 * total_need - 2.0 * overflow
            scored_splits.append(
                (score, _stable_fraction(f"{group}:{split}", seed), split)
            )
        chosen = max(scored_splits)[2]
        group_assignment[str(group)] = chosen
        assigned_totals[chosen] += group_size
        for stratum, count in counts.items():
            assigned_counts[chosen][stratum] += int(count)

    return dataframe["duplicate_group_id"].astype(str).map(group_assignment)


def _distribution(dataframe: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    if dataframe.empty:
        return {}
    counts = dataframe.groupby(columns, dropna=False).size()
    return {
        "|".join(str(part) for part in key if part is not None)
        if isinstance(key, tuple)
        else str(key): int(value)
        for key, value in counts.items()
    }


def prepare_training_dataset(
    labeled: pd.DataFrame,
    preprocessing_config: Mapping[str, Any],
    *,
    preprocessing_version: str,
    seed: int | None = None,
) -> PreparedDataset:
    required = {
        "sample_id",
        "query_text",
        "eval_name",
        "domain",
        "difficulty_label",
        "label_reason",
        "label_confidence",
        "needs_retrieval",
    }
    missing = sorted(required - set(labeled.columns))
    if missing:
        raise RouterBenchError(f"Labeled input is missing preprocessing columns: {missing}")
    if labeled["sample_id"].duplicated().any():
        raise RouterBenchError("Labeled input contains duplicate sample_id values")

    metadata_columns = (
        "sample_id",
        "query_text",
        "eval_name",
        "domain",
        "difficulty_label",
        "label_reason",
        "label_confidence",
        "label_weight",
        "needs_retrieval",
        "shot_source",
        "low_success_rate",
        "low_valid_model_count",
        "medium_success_rate",
        "medium_valid_model_count",
        "high_success_rate",
        "high_valid_model_count",
        "overall_solve_rate",
        "cheapest_correct_tier",
        "dataset_version",
        "tier_config_version",
        "label_rule_version",
    )
    metadata = labeled.loc[
        :, [column for column in metadata_columns if column in labeled.columns]
    ].copy()
    cleaned = metadata["query_text"].map(clean_model_input_text)
    metadata["model_input_text"] = cleaned.map(lambda result: result[0])
    metadata["benchmark_wrapper_removed"] = cleaned.map(lambda result: result[1])
    if metadata["model_input_text"].eq("").any():
        empty_ids = metadata.loc[
            metadata["model_input_text"].eq(""), "sample_id"
        ].astype(str).head(5).tolist()
        raise RouterBenchError(f"Empty model_input_text after cleaning: {empty_ids}")
    metadata["character_count"] = metadata["model_input_text"].str.len()
    metadata["word_count"] = metadata["model_input_text"].map(
        lambda text: len(_WORD_PATTERN.findall(text))
    )

    metadata["duplicate_group_id"], duplicate_metrics = assign_duplicate_groups(
        metadata["model_input_text"],
        jaccard_threshold=float(
            preprocessing_config["near_duplicate_jaccard_threshold"]
        ),
        minimum_tokens=int(preprocessing_config["near_duplicate_min_tokens"]),
        maximum_bucket_size=int(
            preprocessing_config["near_duplicate_max_bucket_size"]
        ),
    )
    confidence_weights = {
        str(key): float(value)
        for key, value in preprocessing_config["confidence_weights"].items()
    }
    metadata["sample_weight"] = metadata["label_confidence"].map(confidence_weights)
    if metadata["sample_weight"].isna().any():
        unknown = sorted(
            metadata.loc[metadata["sample_weight"].isna(), "label_confidence"]
            .astype(str)
            .unique()
            .tolist()
        )
        raise RouterBenchError(f"No preprocessing weight for confidence: {unknown}")

    metadata["exclusion_reason"] = ""
    metadata.loc[metadata["sample_weight"].le(0), "exclusion_reason"] = (
        "non_positive_confidence_weight"
    )
    if not bool(preprocessing_config.get("include_retrieval", True)):
        retrieval = metadata["needs_retrieval"].astype(bool)
        metadata.loc[retrieval, "exclusion_reason"] = "retrieval_excluded"
    metadata["included_in_training"] = metadata["exclusion_reason"].eq("")

    included = metadata.loc[metadata["included_in_training"]].copy()
    if included.empty:
        raise RouterBenchError("Preprocessing excluded every labeled row")
    effective_seed = int(preprocessing_config["seed"]) if seed is None else int(seed)
    included["split"] = assign_group_splits(
        included,
        preprocessing_config["split_fractions"],
        effective_seed,
    )
    metadata["split"] = metadata["sample_id"].map(
        included.set_index("sample_id")["split"]
    )
    metadata["preprocessing_version"] = preprocessing_version

    training = included.loc[:, list(TRAINING_COLUMNS)].reset_index(drop=True)
    metadata = metadata.reset_index(drop=True)
    validate_preprocessed_output(training)
    metrics: dict[str, Any] = {
        "preprocessing_version": preprocessing_version,
        "seed": effective_seed,
        "input_row_count": int(len(metadata)),
        "training_row_count": int(len(training)),
        "excluded_row_count": int((~metadata["included_in_training"]).sum()),
        "benchmark_wrapper_removed_count": int(
            metadata["benchmark_wrapper_removed"].sum()
        ),
        "split_distribution": _distribution(training, ["split"]),
        "split_label_distribution": _distribution(
            training, ["split", "difficulty_label"]
        ),
        "split_domain_distribution": _distribution(
            metadata.loc[metadata["included_in_training"]], ["split", "domain"]
        ),
        "confidence_distribution": _distribution(metadata, ["label_confidence"]),
        "exclusion_distribution": _distribution(
            metadata.loc[~metadata["included_in_training"]], ["exclusion_reason"]
        ),
        "character_count": {
            "min": int(metadata["character_count"].min()),
            "median": float(metadata["character_count"].median()),
            "max": int(metadata["character_count"].max()),
        },
        "word_count": {
            "min": int(metadata["word_count"].min()),
            "median": float(metadata["word_count"].median()),
            "max": int(metadata["word_count"].max()),
        },
        **duplicate_metrics,
    }
    return PreparedDataset(training=training, metadata=metadata, metrics=metrics)


def validate_preprocessed_output(training: pd.DataFrame) -> None:
    if tuple(training.columns) != TRAINING_COLUMNS:
        raise RouterBenchError(
            f"Training projection must contain only {list(TRAINING_COLUMNS)}"
        )
    if training["sample_id"].duplicated().any():
        raise RouterBenchError("Training projection contains duplicate sample_id values")
    if training["model_input_text"].eq("").any():
        raise RouterBenchError("Training projection contains empty model_input_text")
    if not training["difficulty_label"].isin({"easy", "medium", "hard"}).all():
        raise RouterBenchError("Training projection contains an invalid label")
    if not training["sample_weight"].gt(0).all():
        raise RouterBenchError("Training projection contains a non-positive weight")
    split_counts = training.groupby("duplicate_group_id")["split"].nunique()
    if split_counts.gt(1).any():
        raise RouterBenchError("A duplicate group crosses data splits")
