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


def _take_groups_up_to(
    groups: pd.DataFrame,
    row_target: int,
    *,
    seed: int,
    label: str,
    hard_strong_threshold: float,
) -> tuple[list[str], int]:
    if row_target <= 0 or groups.empty:
        return [], 0

    def priority(group_id: object) -> tuple[float, float, float]:
        row = groups.loc[group_id]
        success_rate = float(row["high_success_rate"])
        strong_priority = (
            0.0
            if label == "hard" and success_rate >= hard_strong_threshold
            else 1.0
        )
        success_priority = -success_rate if label == "hard" else 0.0
        return (
            strong_priority,
            success_priority,
            _stable_fraction(str(group_id), seed),
        )

    selected: list[str] = []
    selected_rows = 0
    for group_id in sorted(groups.index, key=priority):
        group_size = int(groups.at[group_id, "row_count"])
        if selected_rows + group_size > row_target:
            continue
        selected.append(str(group_id))
        selected_rows += group_size
        if selected_rows == row_target:
            break
    return selected, selected_rows


def select_stratified_balanced_train_groups(
    train: pd.DataFrame,
    balance_config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[pd.Series, dict[str, Any]]:
    """Balance labels inside each stratum instead of only across the whole train split.

    Global balancing leaves each topic internally skewed, so the model can use topic
    as a proxy for difficulty. Balancing within a stratum breaks that correlation.
    Duplicate groups stay atomic, so a stratum reaches at most its per-label target.
    """
    labels = tuple(str(label) for label in balance_config["label_order"])
    if set(labels) != {"easy", "medium", "hard"} or len(labels) != 3:
        raise RouterBenchError(
            "balanced_train.label_order must contain easy, medium, hard exactly once"
        )
    stratify_by = str(balance_config.get("stratify_by", "eval_name"))
    if stratify_by not in train.columns:
        raise RouterBenchError(
            f"balanced_train.stratify_by column is missing: {stratify_by}"
        )
    min_stratum_rows = int(balance_config.get("min_stratum_rows", 0))
    if min_stratum_rows < 0:
        raise RouterBenchError("balanced_train.min_stratum_rows must not be negative")
    hard_strong_threshold = float(
        balance_config.get("hard_strong_success_rate_threshold", 2.0 / 3.0)
    )

    working = train.copy()
    if "high_success_rate" not in working:
        working["high_success_rate"] = 0.0
    working["_stratum"] = working[stratify_by].astype(str)
    group_summary = working.groupby("duplicate_group_id", sort=True).agg(
        row_count=("sample_id", "size"),
        label_count=("difficulty_label", "nunique"),
        difficulty_label=("difficulty_label", "first"),
        stratum_count=("_stratum", "nunique"),
        stratum=("_stratum", "first"),
        high_success_rate=("high_success_rate", "min"),
    )
    mixed_label_groups = int(group_summary["label_count"].gt(1).sum())
    mixed_stratum_groups = int(group_summary["stratum_count"].gt(1).sum())
    homogeneous = group_summary.loc[
        group_summary["label_count"].eq(1) & group_summary["stratum_count"].eq(1)
    ].copy()

    selected_groups: set[str] = set()
    per_stratum: dict[str, Any] = {}
    dropped_small: list[str] = []
    for stratum in sorted(homogeneous["stratum"].astype(str).unique()):
        candidates = homogeneous.loc[homogeneous["stratum"].astype(str).eq(stratum)]
        available = {
            label: int(
                candidates.loc[
                    candidates["difficulty_label"].astype(str).eq(label), "row_count"
                ].sum()
            )
            for label in labels
        }
        target = min(available.values())
        if target == 0 or sum(available.values()) < min_stratum_rows:
            dropped_small.append(stratum)
            continue
        achieved: dict[str, int] = {}
        for label_offset, label in enumerate(labels):
            label_groups = candidates.loc[
                candidates["difficulty_label"].astype(str).eq(label)
            ]
            group_ids, row_count = _take_groups_up_to(
                label_groups,
                target,
                seed=seed + label_offset,
                label=label,
                hard_strong_threshold=hard_strong_threshold,
            )
            selected_groups.update(group_ids)
            achieved[label] = row_count
        per_stratum[stratum] = {
            "available": available,
            "target_per_class": target,
            "selected": achieved,
        }

    selected = pd.Series(False, index=train.index, dtype=bool)
    selected.loc[train["duplicate_group_id"].astype(str).isin(selected_groups)] = True
    selected_frame = train.loc[selected]
    group_selection_counts = selected_frame.groupby("duplicate_group_id").size()
    original_group_counts = train.groupby("duplicate_group_id").size()
    if not group_selection_counts.eq(
        original_group_counts.loc[group_selection_counts.index]
    ).all():
        raise RouterBenchError("Stratified selection split a duplicate group")

    class_counts = {
        label: int(selected_frame["difficulty_label"].astype(str).eq(label).sum())
        for label in labels
    }
    return selected, {
        "enabled": True,
        "strategy": "stratified",
        "stratify_by": stratify_by,
        "min_stratum_rows": min_stratum_rows,
        "selected_row_count": int(len(selected_frame)),
        "selected_by_label": class_counts,
        "selected_by_label_domain": _distribution(
            selected_frame, ["difficulty_label", "domain"]
        ),
        "stratum_count_used": len(per_stratum),
        "stratum_count_dropped": len(dropped_small),
        "strata_dropped": sorted(dropped_small),
        "mixed_label_group_count_excluded_from_selection": mixed_label_groups,
        "mixed_stratum_group_count_excluded_from_selection": mixed_stratum_groups,
        "hard_strong_success_rate_threshold": hard_strong_threshold,
        "per_stratum": per_stratum,
    }


def select_balanced_train_groups(
    train: pd.DataFrame,
    balance_config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[pd.Series, dict[str, Any]]:
    """Select an exact, deterministic class target without splitting duplicate groups."""
    selected = pd.Series(False, index=train.index, dtype=bool)
    if not bool(balance_config.get("enabled", False)):
        selected.loc[:] = True
        return selected, {"enabled": False}

    if str(balance_config.get("strategy", "global")) == "stratified":
        return select_stratified_balanced_train_groups(
            train, balance_config, seed=seed
        )

    labels = tuple(str(label) for label in balance_config["label_order"])
    if set(labels) != {"easy", "medium", "hard"} or len(labels) != 3:
        raise RouterBenchError(
            "balanced_train.label_order must contain easy, medium, hard exactly once"
        )
    target_per_class = int(balance_config["target_per_class"])
    if target_per_class <= 0:
        raise RouterBenchError("balanced_train.target_per_class must be positive")
    rare_domains = {
        str(domain) for domain in balance_config.get("rare_domains", [])
    }
    hard_strong_threshold = float(
        balance_config.get("hard_strong_success_rate_threshold", 2.0 / 3.0)
    )

    working = train.copy()
    if "high_success_rate" not in working:
        working["high_success_rate"] = 0.0
    group_summary = working.groupby("duplicate_group_id", sort=True).agg(
        row_count=("sample_id", "size"),
        label_count=("difficulty_label", "nunique"),
        difficulty_label=("difficulty_label", "first"),
        domain_count=("domain", "nunique"),
        domain=("domain", "first"),
        high_success_rate=("high_success_rate", "min"),
    )
    mixed_label_groups = group_summary.index[group_summary["label_count"].gt(1)]
    homogeneous = group_summary.loc[group_summary["label_count"].eq(1)].copy()
    homogeneous.loc[homogeneous["domain_count"].gt(1), "domain"] = "__mixed__"

    mixed_rare_groups = {
        str(group_id)
        for group_id in mixed_label_groups
        if set(
            working.loc[
                working["duplicate_group_id"].astype(str).eq(str(group_id)),
                "domain",
            ].astype(str)
        ).issubset(rare_domains)
    }
    mixed_rare_rows = working.loc[
        working["duplicate_group_id"].astype(str).isin(mixed_rare_groups)
    ]
    mixed_rare_label_counts = Counter(
        mixed_rare_rows["difficulty_label"].astype(str).tolist()
    )
    if any(mixed_rare_label_counts[label] > target_per_class for label in labels):
        raise RouterBenchError(
            "Rare mixed-label duplicate groups exceed the per-class target"
        )

    selected_groups: set[str] = set(mixed_rare_groups)
    selected_by_label_domain: dict[str, int] = {}
    available_by_label: dict[str, int] = {}
    for label_offset, label in enumerate(labels):
        candidates = homogeneous.loc[
            homogeneous["difficulty_label"].astype(str).eq(label)
        ].copy()
        available_rows = int(
            working["difficulty_label"].astype(str).eq(label).sum()
        )
        available_by_label[label] = available_rows
        selectable_rows = int(candidates["row_count"].sum()) + int(
            mixed_rare_label_counts[label]
        )
        if selectable_rows < target_per_class:
            raise RouterBenchError(
                f"balanced_train needs {target_per_class} {label} rows, "
                f"but only {selectable_rows} selectable group-atomic rows are available"
            )

        label_selected: set[str] = set()
        selected_rows = int(mixed_rare_label_counts[label])
        rare = candidates.loc[candidates["domain"].astype(str).isin(rare_domains)]
        rare_rows = int(rare["row_count"].sum())
        if rare_rows > target_per_class:
            rare_ids, rare_selected_rows = _take_groups_up_to(
                rare,
                target_per_class,
                seed=seed + label_offset,
                label=label,
                hard_strong_threshold=hard_strong_threshold,
            )
        else:
            rare_ids = [str(group_id) for group_id in rare.index]
            rare_selected_rows = rare_rows
        label_selected.update(rare_ids)
        selected_rows += rare_selected_rows

        non_rare = candidates.loc[~candidates.index.astype(str).isin(label_selected)]
        remaining_target = target_per_class - selected_rows
        domain_counts = non_rare.groupby("domain")["row_count"].sum().astype(int)
        quotas: dict[str, int] = {}
        if remaining_target and int(domain_counts.sum()):
            raw_quotas = remaining_target * domain_counts / int(domain_counts.sum())
            quotas = raw_quotas.map(math.floor).astype(int).to_dict()
            leftover = remaining_target - sum(quotas.values())
            remainder_order = sorted(
                domain_counts.index.astype(str),
                key=lambda domain: (
                    -(float(raw_quotas.loc[domain]) - quotas[domain]),
                    _stable_fraction(f"{label}:{domain}", seed),
                ),
            )
            for domain in remainder_order[:leftover]:
                quotas[domain] += 1

        for domain in sorted(quotas):
            domain_groups = non_rare.loc[
                non_rare["domain"].astype(str).eq(domain)
                & ~non_rare.index.astype(str).isin(label_selected)
            ]
            group_ids, row_count = _take_groups_up_to(
                domain_groups,
                quotas[domain],
                seed=seed + label_offset,
                label=label,
                hard_strong_threshold=hard_strong_threshold,
            )
            label_selected.update(group_ids)
            selected_rows += row_count

        if selected_rows < target_per_class:
            remaining = candidates.loc[
                ~candidates.index.astype(str).isin(label_selected)
            ]
            group_ids, row_count = _take_groups_up_to(
                remaining,
                target_per_class - selected_rows,
                seed=seed + label_offset,
                label=label,
                hard_strong_threshold=hard_strong_threshold,
            )
            label_selected.update(group_ids)
            selected_rows += row_count
        if selected_rows != target_per_class:
            raise RouterBenchError(
                f"Could not select exactly {target_per_class} {label} rows "
                "without splitting a duplicate group"
            )
        selected_groups.update(label_selected)

    selected.loc[
        train["duplicate_group_id"].astype(str).isin(selected_groups)
    ] = True
    selected_frame = train.loc[selected]
    class_counts = selected_frame["difficulty_label"].value_counts().to_dict()
    expected_counts = {label: target_per_class for label in labels}
    if class_counts != expected_counts:
        raise RouterBenchError(
            f"Balanced train class counts must be {expected_counts}, got {class_counts}"
        )
    group_selection_counts = selected_frame.groupby("duplicate_group_id").size()
    original_group_counts = train.groupby("duplicate_group_id").size()
    if not group_selection_counts.eq(
        original_group_counts.loc[group_selection_counts.index]
    ).all():
        raise RouterBenchError("Balanced selection split a duplicate group")
    selected_by_label_domain = _distribution(
        selected_frame, ["difficulty_label", "domain"]
    )
    return selected, {
        "enabled": True,
        "target_per_class": target_per_class,
        "available_by_label": available_by_label,
        "selected_by_label": class_counts,
        "selected_by_label_domain": selected_by_label_domain,
        "mixed_label_group_count_selected": len(mixed_rare_groups),
        "mixed_label_group_count_excluded_from_selection": int(
            len(mixed_label_groups) - len(mixed_rare_groups)
        ),
        "rare_domains_preserved": sorted(rare_domains),
        "hard_strong_success_rate_threshold": hard_strong_threshold,
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
    excluded_labels = {
        str(label)
        for label in preprocessing_config.get("excluded_difficulty_labels", [])
    }
    if excluded_labels:
        excluded_label_rows = metadata["difficulty_label"].astype(str).isin(
            excluded_labels
        )
        metadata.loc[excluded_label_rows, "exclusion_reason"] = (
            "excluded_difficulty_label"
        )
    if not bool(preprocessing_config.get("include_retrieval", True)):
        retrieval = metadata["needs_retrieval"].astype(bool) & metadata[
            "exclusion_reason"
        ].eq("")
        metadata.loc[retrieval, "exclusion_reason"] = "retrieval_excluded"

    eligible = metadata.loc[metadata["exclusion_reason"].eq("")].copy()
    if eligible.empty:
        raise RouterBenchError("Preprocessing excluded every labeled row")
    effective_seed = int(preprocessing_config["seed"]) if seed is None else int(seed)
    eligible["split"] = assign_group_splits(
        eligible,
        preprocessing_config["split_fractions"],
        effective_seed,
    )
    metadata["split"] = metadata["sample_id"].map(
        eligible.set_index("sample_id")["split"]
    )
    natural_split_distribution = _distribution(eligible, ["split"])
    natural_split_label_distribution = _distribution(
        eligible, ["split", "difficulty_label"]
    )

    selected = pd.Series(True, index=eligible.index, dtype=bool)
    train_mask = eligible["split"].eq("train")
    train_selected, balance_metrics = select_balanced_train_groups(
        eligible.loc[train_mask],
        preprocessing_config.get("balanced_train", {"enabled": False}),
        seed=effective_seed,
    )
    selected.loc[train_mask] = train_selected
    unselected_train_ids = eligible.loc[train_mask & ~selected, "sample_id"]
    metadata.loc[
        metadata["sample_id"].isin(unselected_train_ids), "exclusion_reason"
    ] = "train_balance_not_selected"
    metadata["included_in_training"] = metadata["exclusion_reason"].eq("")
    metadata["preprocessing_version"] = preprocessing_version

    training = eligible.loc[selected, list(TRAINING_COLUMNS)].reset_index(drop=True)
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
        "natural_split_distribution_before_train_balance": (
            natural_split_distribution
        ),
        "natural_split_label_distribution_before_train_balance": (
            natural_split_label_distribution
        ),
        "balanced_train": balance_metrics,
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
