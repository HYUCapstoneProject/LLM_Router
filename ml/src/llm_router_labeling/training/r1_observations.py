"""Project trusted service outcomes to R1 targets without changing HTTP contracts.

The input is an internal immutable export. Chain completion is never a success
observation. Export/import authorization belongs to the caller; a JSON field is
not proof that its author is a trusted assessor.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import pandas as pd

from ..shared.structured_text import clean_text
from ..training.routing_policy import ACTION_ORDER, RoutingPolicy

BATCH_SCHEMA = "r1-observation-batch-v1"
EVENT_SCHEMA = "r1-observation-v1"
TARGET_CONTRACT = "service-r1-observed-binary-v1"
FIELDS = {
    "schema_version",
    "event_id",
    "request_id",
    "question",
    "router_model_version",
    "policy_sha256",
    "candidate_pool_version",
    "candidate_model_id",
    "action",
    "success",
    "assessment_source",
    "assessment_id",
    "assessed_at",
    "pii_redacted",
}
SOURCES = {"explicit_user", "reference_evaluator", "human_review"}
UTC_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|\+00:00)$"
)


def question_sha256(question: str) -> str:
    return hashlib.sha256(clean_text(question).encode()).hexdigest()


def project_observations(
    batch: Mapping[str, Any],
    policy: RoutingPolicy,
    *,
    protected_question_hashes: Iterable[str],
    allowed_router_model_versions: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Only create train rows; protected split membership is supplied explicitly."""
    if policy.objective != "pointwise":
        raise ValueError("R1 observations require a pointwise policy")
    if (
        set(batch) != {"schema_version", "observations"}
        or batch["schema_version"] != BATCH_SCHEMA
    ):
        raise ValueError("Unexpected R1 observation batch schema")
    observations = batch["observations"]
    if not isinstance(observations, list) or not observations:
        raise ValueError("R1 observations must be a nonempty list")
    versions = set(allowed_router_model_versions)
    if not versions or any(
        not isinstance(value, str) or not value for value in versions
    ):
        raise ValueError("Pin the router model versions allowed to supply observations")
    protected = set(protected_question_hashes)
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in protected
    ):
        raise ValueError("Protected question hashes must be explicit SHA256 strings")
    seen_events: set[str] = set()
    grouped: dict[str, dict[str, Any]] = {}
    excluded = 0
    for event in observations:
        if (
            not isinstance(event, dict)
            or set(event) != FIELDS
            or event["schema_version"] != EVENT_SCHEMA
        ):
            raise ValueError("Unexpected R1 observation fields/schema")
        for key in FIELDS - {"success", "pii_redacted"}:
            if not isinstance(event[key], str) or not event[key].strip():
                raise ValueError(f"{key} must be a nonblank string")
        if re.fullmatch(r"[A-Za-z0-9._-]{1,200}", event["event_id"]) is None:
            raise ValueError("Invalid event_id")
        request_id = str(UUID(event["request_id"]))
        for key in (
            "router_model_version",
            "candidate_pool_version",
            "candidate_model_id",
            "assessment_id",
        ):
            if any(character.isspace() for character in event[key]):
                raise ValueError(f"{key} must contain no whitespace")
        if event["event_id"] in seen_events:
            raise ValueError("Duplicate event_id in immutable export")
        seen_events.add(event["event_id"])
        if type(event["success"]) is not bool or event["pii_redacted"] is not True:
            raise ValueError(
                "Explicit boolean success and PII declaration are required"
            )
        if event["assessment_source"] not in SOURCES:
            raise ValueError("Chain completion/inferred success is not an assessment")
        if UTC_PATTERN.fullmatch(event["assessed_at"]) is None:
            raise ValueError("Assessment time must be an explicit ISO8601 UTC string")
        time = datetime.fromisoformat(event["assessed_at"].replace("Z", "+00:00"))
        if time.tzinfo is None or time.utcoffset() != timedelta(0):
            raise ValueError("Assessment time must be UTC")
        if (
            event["policy_sha256"] != policy.policy_sha256
            or event["candidate_pool_version"] != policy.candidate_pool_version
            or event["router_model_version"] not in versions
        ):
            raise ValueError("Observation policy/pool/model provenance mismatch")
        if event["action"] not in ACTION_ORDER:
            raise ValueError("Unknown observed action")
        candidate = policy.candidates[ACTION_ORDER.index(event["action"])]
        if event["candidate_model_id"] != candidate.model_id:
            raise ValueError("Observed model does not match its frozen action")
        question = clean_text(event["question"])
        if not question.strip() or len(question) > 100_000:
            raise ValueError("Invalid redacted question")
        question_hash = question_sha256(question)
        if question_hash in protected:
            excluded += 1
            continue
        row = grouped.setdefault(
            request_id,
            {
                "sample_id": "service-r1:"
                + hashlib.sha256(request_id.encode()).hexdigest(),
                "request_id": request_id,
                "model_input_text": question,
                "duplicate_group_id": question_hash,
                "sample_weight": 1.0,
                "split": "train",
                "policy_sha256": policy.policy_sha256,
                "candidate_pool_version": policy.candidate_pool_version,
                "target_contract_version": TARGET_CONTRACT,
                **{f"success_{action}": None for action in ACTION_ORDER},
                **{f"observed_{action}": 0 for action in ACTION_ORDER},
                "observation_ids": [],
            },
        )
        if row["model_input_text"] != question:
            raise ValueError("One request_id contains conflicting questions")
        action = event["action"]
        target = int(event["success"])
        if row[f"observed_{action}"] and row[f"success_{action}"] != target:
            raise ValueError("Conflicting outcomes for one request/candidate")
        row[f"success_{action}"] = target
        row[f"observed_{action}"] = 1
        row["observation_ids"].append(event["event_id"])
    rows = [grouped[key] for key in sorted(grouped)]
    if not rows:
        raise ValueError(
            "No eligible observations remain after protected-split exclusion"
        )
    for row in rows:
        row["observation_ids"] = json.dumps(
            sorted(row["observation_ids"]), separators=(",", ":")
        )
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return pd.DataFrame(rows), {
        "schema_version": "r1-target-projection-v1",
        "target_contract_version": TARGET_CONTRACT,
        "policy_sha256": policy.policy_sha256,
        "candidate_pool_version": policy.candidate_pool_version,
        "allowed_router_model_versions": sorted(versions),
        "input_observation_count": len(observations),
        "protected_observations_excluded": excluded,
        "training_row_count": len(rows),
        "canonical_rows_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "observed_counts": {
            action: sum(row[f"observed_{action}"] for row in rows)
            for action in ACTION_ORDER
        },
        "selection_bias_note": "Only recorded candidates are supervised; this is not counterfactual or unbiased policy evaluation.",
    }


def merge_observed_training(base: pd.DataFrame, feedback: pd.DataFrame) -> pd.DataFrame:
    """Preserve targets/masks and base validation; reject cross-policy rehearsal."""
    required = {
        "sample_id",
        "model_input_text",
        "split",
        "duplicate_group_id",
        "policy_sha256",
        "candidate_pool_version",
        "target_contract_version",
    }
    required |= {
        f"{prefix}_{action}"
        for prefix in ("success", "observed")
        for action in ACTION_ORDER
    }
    for frame in (base, feedback):
        if frame.empty or required - set(frame):
            raise ValueError("R1 merge requires explicit targets, masks and provenance")
        if frame["sample_id"].duplicated().any():
            raise ValueError("Duplicate sample IDs")
        if (
            not frame["model_input_text"]
            .map(lambda value: isinstance(value, str) and bool(clean_text(value)))
            .all()
        ):
            raise ValueError("R1 merge requires nonempty canonical question text")
        if not frame["target_contract_version"].eq(TARGET_CONTRACT).all():
            raise ValueError("R1 merge requires the observed binary target contract")
    for field in ("policy_sha256", "candidate_pool_version", "target_contract_version"):
        values = set(base[field]) | set(feedback[field])
        if len(values) != 1 or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ValueError(f"R1 rehearsal {field} mismatch")
    if not feedback["split"].eq("train").all():
        raise ValueError("Feedback may only join the training split")
    protected = set(base.loc[~base["split"].eq("train"), "duplicate_group_id"])
    if protected & set(feedback["duplicate_group_id"]):
        raise ValueError("Feedback overlaps protected base evaluation groups")
    protected_texts = set(
        base.loc[~base["split"].eq("train"), "model_input_text"].map(question_sha256)
    )
    if protected_texts & set(feedback["model_input_text"].map(question_sha256)):
        raise ValueError("Feedback overlaps protected base evaluation questions")
    if set(base["sample_id"]) & set(feedback["sample_id"]):
        raise ValueError("An existing observation cannot be appended a second time")
    result = pd.concat([base, feedback], ignore_index=True)
    if result.groupby("duplicate_group_id")["split"].nunique().gt(1).any():
        raise ValueError("Duplicate group crosses training splits")
    canonical_splits = result.groupby(result["model_input_text"].map(question_sha256))[
        "split"
    ].nunique()
    if canonical_splits.gt(1).any():
        raise ValueError("Canonical question crosses protected training splits")
    return result
