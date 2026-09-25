"""Frozen RouterArena alignment diagnostics; never selects or deploys a model.

prepare runs without model inference, infer needs local frozen checkpoints, and
report consumes only sealed predictions. R1/R2 action-slot projection is not a
trained difficulty classifier, an Arena Score, or a service quality/cost claim.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from llm_router_labeling.evaluation.routerarena_preprocessing import (
    load_routerarena_source,
)
from llm_router_labeling.routing_data.text import compatible_near_duplicate
from llm_router_labeling.shared.structured_text import (
    clean_text,
    structured_duplicate_key,
)
from llm_router_labeling.training.routerbench_preprocessing import (
    assign_duplicate_groups,
)
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

METHODS = ("ordinal", "pointwise", "pairwise")
LABELS = ("easy", "medium", "hard")
ACTIONS = ("low", "medium", "high")
DEVELOPMENT_SPLITS = ("train", "validation", "gate", "stream", "retention_test")
VERSION = "routing-arena-diagnostics-v1"
REFUSAL = re.compile(
    r"^i(?:'m| am) sorry,?(?: but)? i (?:can't|cannot) assist with that request\.?$",
    re.IGNORECASE,
)
BOXED = re.compile(r"\n\s*\n\\boxed\{\{[^}]", re.IGNORECASE)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict:
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise TypeError(f"Expected object: {path}")
    return result


def write(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    )


def record(path: Path) -> dict:
    return {"sha256": sha(path), "size_bytes": path.stat().st_size}


def fresh(path: Path) -> None:
    if path.exists():
        raise ValueError(f"Output must not exist: {path}")
    path.mkdir(parents=True)


def seal(root: Path, manifest: dict) -> dict:
    manifest["files"] = {
        str(p.relative_to(root)): record(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name not in {"manifest.json", "_SUCCESS"}
    }
    write(root / "manifest.json", manifest)
    (root / "_SUCCESS").write_text(sha(root / "manifest.json") + "\n")
    return manifest


def verify(root: Path, role: str) -> dict:
    manifest = read(root / "manifest.json")
    if (root / "_SUCCESS").read_text().strip() != sha(root / "manifest.json"):
        raise ValueError("Manifest seal mismatch")
    if manifest.get("schema_version") != VERSION or manifest.get("role") != role:
        raise ValueError("Unexpected diagnostic artifact role/schema")
    for name, expected in manifest["files"].items():
        path = root / name
        if path.resolve().parent != root.resolve() or record(path) != expected:
            raise ValueError(f"Artifact checksum/path mismatch: {name}")
    return manifest


def clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return clean_text(html.unescape(str(value)))


def options(value: Any) -> list[str]:
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                value = [value]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [text for item in value if (text := clean(item))]


def model_input(context: Any, question: Any, answer_options: Any) -> str:
    question = clean(question)
    if not question.strip():
        return ""
    parts = [f"Context:\n{clean(context)}"] if clean(context).strip() else []
    parts.append(question)
    choices = options(answer_options)
    if choices:
        parts.append(
            "\n".join(
                f"{chr(65 + i) if i < 26 else i + 1}) {text}"
                for i, text in enumerate(choices)
            )
        )
    return clean_text("\n\n".join(parts))


def arena_rows(full: pd.DataFrame, robustness: pd.DataFrame) -> pd.DataFrame:
    if full["Global Index"].astype(str).duplicated().any():
        raise ValueError("Duplicate full Global Index")
    lookup = full.set_index(full["Global Index"].astype(str))
    rows = []
    for split, source in (("full", full), ("robustness", robustness)):
        if source["Global Index"].astype(str).duplicated().any():
            raise ValueError(f"Duplicate {split} Global Index")
        for _, row in source.iterrows():
            source_id = str(row["Global Index"])
            if source_id not in lookup.index:
                raise ValueError(f"Missing full reference: {source_id}")
            reference = lookup.loc[source_id]
            label = str(reference["Difficulty"]).strip().casefold()
            if label not in LABELS:
                raise ValueError(f"Unknown full difficulty: {label}")
            question = clean(row["Question"])
            text = model_input(row["Context"], row["Question"], row["Options"])
            reason = ""
            if not text.strip():
                reason = "missing_question"
            elif REFUSAL.fullmatch(question.strip()):
                reason = "refusal_placeholder"
            elif BOXED.search(question):
                reason = "embedded_answer"
            dataset = clean(reference["Dataset name"])
            category = clean(reference["Category"])
            unseen = (
                dataset.startswith(("WMT19", "ChessInstruct"))
                or dataset == "NarrativeQA"
                or category.startswith("78 ")
            )
            rows.append(
                {
                    "sample_id": f"routerarena:{split}:{source_id}",
                    "source_sample_id": source_id,
                    "evaluation_split": split,
                    "source_dataset": "routerarena",
                    "arena_source_dataset": dataset,
                    "category": category,
                    "domain": clean(reference["Domain"]),
                    "difficulty_label": label,
                    "model_input_text": text,
                    "slice": "rb_unseen" if unseen else "rb_service_aligned",
                    "exclusion_reason": reason,
                    "label_source": "official_full_join"
                    if split == "robustness"
                    else "official_full",
                }
            )
    return pd.DataFrame(rows)


def audit_overlap(
    arena: pd.DataFrame, development: pd.DataFrame, dedup: dict
) -> tuple[pd.DataFrame, dict]:
    arena = arena.copy()
    valid = arena["exclusion_reason"].eq("")
    combined = pd.concat(
        [
            development[["sample_id", "model_input_text"]].assign(kind="development"),
            arena.loc[valid, ["sample_id", "model_input_text"]].assign(kind="arena"),
        ],
        ignore_index=True,
    )
    groups, metrics = assign_duplicate_groups(
        combined["model_input_text"],
        jaccard_threshold=float(dedup["near_duplicate_jaccard_threshold"]),
        minimum_tokens=int(dedup["near_duplicate_min_tokens"]),
        maximum_bucket_size=int(dedup["near_duplicate_max_bucket_size"]),
        normalizer=structured_duplicate_key,
        near_duplicate_guard=compatible_near_duplicate,
    )
    combined["duplicate_group_id"] = groups
    dev_groups = set(
        combined.loc[combined["kind"].eq("development"), "duplicate_group_id"]
    )
    arena_groups = combined.loc[combined["kind"].eq("arena")].set_index("sample_id")[
        "duplicate_group_id"
    ]
    arena["duplicate_group_id"] = arena["sample_id"].map(arena_groups)
    dev_keys = set(development["model_input_text"].map(structured_duplicate_key))
    arena["exact_development_overlap"] = (
        arena["model_input_text"].map(structured_duplicate_key).isin(dev_keys)
    )
    arena["near_or_exact_development_overlap"] = arena["duplicate_group_id"].isin(
        dev_groups
    )
    arena.loc[
        valid & arena["near_or_exact_development_overlap"], "exclusion_reason"
    ] = "development_overlap"
    candidates = arena["exclusion_reason"].eq("")
    arena["content_sha256"] = arena["model_input_text"].map(
        lambda text: hashlib.sha256(structured_duplicate_key(text).encode()).hexdigest()
    )
    duplicates = arena.loc[candidates].duplicated(
        ["evaluation_split", "content_sha256"], keep="first"
    )
    arena.loc[duplicates.index[duplicates], "exclusion_reason"] = (
        "duplicate_content_within_split"
    )
    full_kept = set(
        arena.loc[
            arena["evaluation_split"].eq("full") & arena["exclusion_reason"].eq(""),
            "source_sample_id",
        ]
    )
    missing_pair = arena["evaluation_split"].eq("robustness") & ~arena[
        "source_sample_id"
    ].isin(full_kept)
    arena.loc[missing_pair & arena["exclusion_reason"].eq(""), "exclusion_reason"] = (
        "full_reference_excluded"
    )
    overlapping = combined.loc[
        combined["duplicate_group_id"].isin(
            set(
                arena.loc[
                    arena["near_or_exact_development_overlap"], "duplicate_group_id"
                ]
            )
        )
    ]
    return arena, {
        "development_rows": len(development),
        "development_splits": development["split"].value_counts().to_dict(),
        "development_metadata_role": "all_development_roles_excluding_original_tests",
        "algorithm": "structured_exact_plus_simhash_candidates_jaccard_with_operator_negation_indent_guard",
        "deduplication": dedup,
        "group_metrics": metrics,
        "exact_arena_overlap_rows": int(arena["exact_development_overlap"].sum()),
        "exact_or_near_arena_overlap_rows": int(
            arena["near_or_exact_development_overlap"].sum()
        ),
        "overlap_group_members": overlapping[
            ["sample_id", "kind", "duplicate_group_id"]
        ].to_dict("records"),
        "limitations": [
            "Near-duplicate candidate generation is approximate; skipped large buckets are counted.",
            "Shared benchmark names alone do not prove contamination; cross-dataset source IDs are not assumed equivalent.",
            "Historical RouterArena observations exist; this is not a claim that the benchmark has never been viewed.",
        ],
    }


def distribution(frame: pd.DataFrame, column: str) -> dict:
    return {str(k): int(v) for k, v in frame[column].value_counts().items()}


def prepare(args: argparse.Namespace) -> dict:
    arena_config = read(args.arena_config)
    development_config = read(args.development_config)
    lock = read(args.validation_lock)
    if lock.get("stage") != "final" or set(lock["methods"]) != set(METHODS):
        raise ValueError("All three methods must be frozen by a final validation lock")
    if development_config.get("input_contract") != "structure-preserving-v2":
        raise ValueError("Requires the current structured development contract")
    sources = {
        name: load_routerarena_source(
            args.raw_dir / arena_config["sources"][name]["filename"],
            arena_config["sources"][name],
        )
        for name in ("full", "robustness", "sub_10")
    }
    if not set(sources["sub_10"]["Global Index"].astype(str)) <= set(
        sources["full"]["Global Index"].astype(str)
    ):
        raise ValueError("sub_10 is not a full subset")
    all_metadata = pd.read_parquet(args.development_metadata)
    if not set(all_metadata["split"]).issubset(
        set(DEVELOPMENT_SPLITS) | {"test", "external_test"}
    ):
        raise ValueError("Unknown development split")
    development = all_metadata.loc[
        all_metadata["split"].isin(DEVELOPMENT_SPLITS)
    ].copy()
    if development.empty or development["sample_id"].duplicated().any():
        raise ValueError("Development rows must be nonempty and unique")
    arena, audit = audit_overlap(
        arena_rows(sources["full"], sources["robustness"]),
        development,
        development_config["deduplication"],
    )
    selected = (
        arena.loc[arena["exclusion_reason"].eq("")]
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    if selected.empty:
        raise ValueError("No rows survive preparation")
    manifests = {}
    for method in METHODS:
        path = args.run_root / method / "manifest.json"
        manifest = read(path)
        if (
            manifest.get("method") != method
            or manifest.get("seed") != lock["canonical_seed"]
            or manifest.get("experiment_config_sha256")
            != lock["experiment_config_sha256"]
        ):
            raise ValueError(f"Run is not the frozen canonical method: {method}")
        if any(
            manifest.get(k, 0)
            for k in (
                "maximum_train_rows",
                "maximum_train_steps",
                "maximum_validation_rows",
            )
        ):
            raise ValueError("Smoke checkpoint is not allowed")
        if (
            manifest["common_training_config"]["max_length"] != 512
            or "model.safetensors" not in manifest["model_artifacts"]
        ):
            raise ValueError("A frozen 512-token final checkpoint is required")
        manifests[method] = manifest
    fresh(args.output_dir)
    for method in METHODS:
        shutil.copyfile(
            args.run_root / method / "manifest.json",
            args.output_dir / f"{method}-run-manifest.json",
        )
    shutil.copyfile(args.validation_lock, args.output_dir / "validation-lock.json")
    arena.to_parquet(args.output_dir / "metadata.parquet", index=False)
    selected.to_parquet(args.output_dir / "evaluation.parquet", index=False)
    write(args.output_dir / "overlap-audit.json", audit)
    protocol = {
        "schema_version": VERSION,
        "arena_dataset_version": arena_config.get("dataset_version", "SYNTHETIC"),
        "arena_revision": arena_config.get("revision", "SYNTHETIC"),
        "preprocessing_version": "routerarena-structure-preserving-diagnostic-v1",
        "projection_policy": "fixed_lambda_zero_action_slot_argmax",
        "lambda_cost": 0.0,
        "action_order": list(ACTIONS),
        "label_projection": list(LABELS),
        "max_length": 512,
        "input_contract": "structure-preserving-v2",
        "attention_implementation": "eager",
        "dtype": "float32",
        "batch_size": 16,
        "tie_breaker": "low_then_medium_then_high",
        "selection_influence": False,
        "production_activation_authorized": False,
        "cost_quality_claim_allowed": False,
        "r1_r2_interpretation": "action_slot_projection_alignment_not_difficulty_classification",
        "robustness": "paired_full_reference_with_inherited_official_label_no_independent_count_addition",
        "source_policy_transfer": "none_no_source_lambda_or_service_policy_inferred",
        "validation_winner_unchanged": lock.get("winner_method"),
        "source_files": {
            "development_metadata": {
                "original_path": str(args.development_metadata),
                **record(args.development_metadata),
            },
            "development_config": record(args.development_config),
            "arena_config": record(args.arena_config),
            **{
                name: record(args.raw_dir / arena_config["sources"][name]["filename"])
                for name in sources
            },
        },
        "script_sha256": sha(Path(__file__)),
        "models": {
            method: {
                "model_dir_recorded": m["model_dir"],
                "model_artifacts": m["model_artifacts"],
            }
            for method, m in manifests.items()
        },
        "counts": {
            split: {
                "raw": int(arena["evaluation_split"].eq(split).sum()),
                "kept": int(selected["evaluation_split"].eq(split).sum()),
                "excluded_by_reason": distribution(
                    arena.loc[
                        arena["evaluation_split"].eq(split)
                        & arena["exclusion_reason"].ne("")
                    ],
                    "exclusion_reason",
                ),
                "labels": distribution(
                    selected.loc[selected["evaluation_split"].eq(split)],
                    "difficulty_label",
                ),
            }
            for split in ("full", "robustness")
        },
        "full_slices": distribution(
            selected.loc[selected["evaluation_split"].eq("full")], "slice"
        ),
        "raw_sub10_rows": len(sources["sub_10"]),
        "sub10_additional_inference": False,
    }
    write(args.output_dir / "protocol.json", protocol)
    return seal(
        args.output_dir,
        {
            "schema_version": VERSION,
            "role": "prepared",
            "protocol_sha256": sha(args.output_dir / "protocol.json"),
            "row_count": len(selected),
        },
    )


def verify_checkpoint(path: Path, expected: dict, method: str) -> None:
    for name, digest in expected.items():
        if Path(name).name != name or record(path / name) != digest:
            raise ValueError(f"Frozen checkpoint mismatch: {method}/{name}")
    config = read(path / "config.json")
    labels = LABELS if method == "ordinal" else ACTIONS
    if config.get("router_objective", "ordinal") != method or config.get(
        "id2label"
    ) != {str(i): label for i, label in enumerate(labels)}:
        raise ValueError(f"Checkpoint output contract mismatch: {method}")


def infer(args: argparse.Namespace) -> dict:
    prepared = verify(args.prepared_dir, "prepared")
    protocol = read(args.prepared_dir / "protocol.json")
    if protocol["script_sha256"] != sha(Path(__file__)):
        raise ValueError("Diagnostic source changed after protocol freeze")
    models = {}
    for value in args.model:
        method, path = value.split("=", 1)
        if method in models or method not in METHODS:
            raise ValueError("Expected exactly one local path for each method")
        models[method] = Path(path)
    if set(models) != set(METHODS):
        raise ValueError("All R0/R1/R2 model paths are required")
    for method, path in models.items():
        verify_checkpoint(path, protocol["models"][method]["model_artifacts"], method)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    frame = pd.read_parquet(args.prepared_dir / "evaluation.parquet")
    fresh(args.output_dir)
    runtimes = {}
    for method in METHODS:
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(models[method], local_files_only=True)
        model = (
            AutoModelForSequenceClassification.from_pretrained(
                models[method], local_files_only=True, attn_implementation="eager"
            )
            .to(device)
            .float()
            .eval()
        )
        scores, lengths = [], []
        with torch.inference_mode():
            for start in range(0, len(frame), protocol["batch_size"]):
                texts = frame.iloc[start : start + protocol["batch_size"]][
                    "model_input_text"
                ].tolist()
                lengths.extend(
                    len(ids)
                    for ids in tokenizer(
                        texts, truncation=False, padding=False, verbose=False
                    )["input_ids"]
                )
                encoded = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                logits = (
                    model(**{key: value.to(device) for key, value in encoded.items()})
                    .logits.detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                if logits.shape != (len(texts), 3) or not np.isfinite(logits).all():
                    raise ValueError("Inference returned invalid logits")
                scores.append(logits)
        raw = np.concatenate(scores)
        output = frame[["sample_id"]].copy()
        for i, action in enumerate(ACTIONS):
            output[f"score_{action}"] = raw[:, i]
        output["input_token_count"] = lengths
        output["was_truncated"] = output["input_token_count"].gt(512)
        output.to_parquet(args.output_dir / f"{method}-scores.parquet", index=False)
        runtimes[method] = {
            "seconds_including_model_load_and_tokenization": time.perf_counter()
            - started,
            "rows": len(output),
            "device": device,
            "truncated_rows": int(output["was_truncated"].sum()),
        }
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    verify(args.prepared_dir, "prepared")
    for method, path in models.items():
        verify_checkpoint(path, protocol["models"][method]["model_artifacts"], method)
    return seal(
        args.output_dir,
        {
            "schema_version": VERSION,
            "role": "scores",
            "prepared_manifest_sha256": sha(args.prepared_dir / "manifest.json"),
            "protocol_sha256": prepared["protocol_sha256"],
            "runtime": runtimes,
        },
    )


def project(method: str, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if logits.ndim != 2 or logits.shape[1] != 3 or not np.isfinite(logits).all():
        raise ValueError("Invalid raw score matrix")
    if method == "ordinal":
        scores = np.exp(logits - logits.max(axis=1, keepdims=True))
        scores /= scores.sum(axis=1, keepdims=True)
    elif method == "pointwise":
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    elif method == "pairwise":
        scores = logits
    else:
        raise ValueError("Unknown method")
    return scores.argmax(axis=1), scores


def metrics(rows: pd.DataFrame) -> dict:
    truth = rows["difficulty_label"].map(dict(zip(LABELS, range(3)))).to_numpy(int)
    pred = rows["action_index"].to_numpy(int)
    matrix = confusion_matrix(truth, pred, labels=[0, 1, 2])
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    recall = np.divide(matrix.diagonal(), support, out=np.zeros(3), where=support > 0)
    qwk = (
        cohen_kappa_score(truth, pred, labels=[0, 1, 2], weights="quadratic")
        if len(np.unique(np.concatenate([truth, pred]))) > 1
        else None
    )
    return {
        "rows": len(rows),
        "accuracy": float((truth == pred).mean()),
        "macro_f1_fixed_three_labels": float(
            f1_score(truth, pred, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "balanced_accuracy_present_labels": float(recall[support > 0].mean()),
        "qwk": float(qwk) if qwk is not None and np.isfinite(qwk) else None,
        "ordinal_mae": float(np.abs(truth - pred).mean()),
        "hard_to_easy_rate_among_hard": float(
            ((truth == 2) & (pred == 0)).sum() / support[2]
        )
        if support[2]
        else None,
        "under_label_rate": float((pred < truth).mean()),
        "over_label_rate": float((pred > truth).mean()),
        "label_support": dict(zip(LABELS, map(int, support))),
        "predicted_support": dict(zip(LABELS, map(int, predicted))),
        "confusion_rows_true_columns_projected": matrix.tolist(),
        "counts_rows_projected_columns_true": matrix.T.tolist(),
        "p_true_given_projected": np.divide(
            matrix.T,
            predicted[:, None],
            out=np.zeros((3, 3)),
            where=predicted[:, None] > 0,
        ).tolist(),
        "p_projected_given_true": np.divide(
            matrix, support[:, None], out=np.zeros((3, 3)), where=support[:, None] > 0
        ).tolist(),
        "conditional_empty_row_policy": "zero_vector_with_explicit_zero_support_not_a_measured_distribution",
        "truncated_rows": int(rows["was_truncated"].sum()),
    }


def report(args: argparse.Namespace) -> dict:
    prepared = verify(args.prepared_dir, "prepared")
    score_manifest = verify(args.scores_dir, "scores")
    if (
        score_manifest["prepared_manifest_sha256"]
        != sha(args.prepared_dir / "manifest.json")
        or score_manifest["protocol_sha256"] != prepared["protocol_sha256"]
    ):
        raise ValueError("Scores do not belong to this frozen preparation")
    frame = pd.read_parquet(args.prepared_dir / "evaluation.parquet")
    protocol = read(args.prepared_dir / "protocol.json")
    if protocol["script_sha256"] != sha(Path(__file__)):
        raise ValueError("Diagnostic source changed after protocol freeze")
    summaries, predictions, category_rows = {}, [], []
    for method in METHODS:
        scores = pd.read_parquet(args.scores_dir / f"{method}-scores.parquet")
        if scores["sample_id"].duplicated().any() or set(scores["sample_id"]) != set(
            frame["sample_id"]
        ):
            raise ValueError(f"Prediction IDs do not match: {method}")
        rows = frame.merge(scores, on="sample_id", how="left", validate="one_to_one")
        raw = rows[[f"score_{action}" for action in ACTIONS]].to_numpy(float)
        actions, action_scores = project(method, raw)
        rows["action_index"] = actions
        rows["projected_label"] = np.array(LABELS)[actions]
        rows["method"] = method
        rows["confidence"] = (
            np.nan
            if method == "pairwise"
            else action_scores[np.arange(len(rows)), actions]
        )
        for i, action in enumerate(ACTIONS):
            rows[f"action_score_{action}"] = action_scores[:, i]
        full = rows.loc[rows["evaluation_split"].eq("full")]
        slices = {
            "full_clean": full,
            **{
                name: full.loc[full["slice"].eq(name)]
                for name in ("rb_service_aligned", "rb_unseen")
            },
        }
        per_slice = {
            name: metrics(part) if len(part) else {"rows": 0, "metrics": None}
            for name, part in slices.items()
        }
        detailed = {}
        for column in ("category", "domain", "arena_source_dataset"):
            detailed[column] = {}
            for name, part in full.groupby(column, sort=True):
                result = metrics(part)
                detailed[column][str(name)] = result
                category_rows.append(
                    {
                        "method": method,
                        "dimension": column,
                        "value": str(name),
                        **{
                            k: v
                            for k, v in result.items()
                            if not isinstance(v, (dict, list))
                        },
                    }
                )
        variants = rows.loc[rows["evaluation_split"].eq("robustness")]
        pairs = variants.merge(
            full[["source_sample_id", "action_index"]],
            on="source_sample_id",
            suffixes=("_variant", "_full"),
            validate="one_to_one",
        )
        delta = pairs["action_index_variant"] - pairs["action_index_full"]
        summaries[method] = {
            "interpretation": "difficulty_prediction_alignment"
            if method == "ordinal"
            else "action_slot_projection_alignment_not_difficulty_classification",
            "slices": per_slice,
            "breakdown": detailed,
            "robustness": {
                "paired_rows": len(pairs),
                "label_alignment": metrics(variants) if len(variants) else None,
                "same_action_rate": float(delta.eq(0).mean()) if len(pairs) else None,
                "upward_action_rate": float(delta.gt(0).mean()) if len(pairs) else None,
                "downward_action_rate": float(delta.lt(0).mean())
                if len(pairs)
                else None,
            },
        }
        predictions.append(rows.drop(columns=["model_input_text"]))
    fresh(args.output_dir)
    pd.concat(predictions, ignore_index=True).to_parquet(
        args.output_dir / "predictions.parquet", index=False
    )
    pd.DataFrame(category_rows).to_csv(
        args.output_dir / "category-domain-source-metrics.csv", index=False
    )
    summary = {
        "schema_version": VERSION,
        "projection_policy": protocol["projection_policy"],
        "lambda_cost": 0.0,
        "selection_influence": False,
        "cost_quality_claim_allowed": False,
        "production_activated": False,
        "validation_winner_unchanged": protocol["validation_winner_unchanged"],
        "preparation_counts": protocol["counts"],
        "methods": summaries,
    }
    write(args.output_dir / "summary.json", summary)
    lines = [
        "# RouterArena difficulty-label / action-slot alignment",
        "",
        "Frozen λ=0 diagnostic; 512 tokens; single seed 42. R0 predicts difficulty; R1/R2 project candidate action slots and are **not difficulty classifiers**. These results do not select a new winner, estimate Claude quality/cost, provide an official Arena Score, or approve deployment.",
        "",
        "Full and robustness are paired overlapping data; their counts must not be added as independent samples. Historical RouterArena observations are disclosed. Near-duplicate detection is approximate; consult overlap-audit.json.",
        "",
        "| Method | Slice | n | Alignment accuracy | Macro-F1 (3 labels) | Balanced accuracy | QWK | Hard→easy |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, result in summaries.items():
        for name, values in result["slices"].items():
            if values["rows"]:
                values_fmt = [
                    "N/A" if values[k] is None else f"{values[k]:.4f}"
                    for k in (
                        "accuracy",
                        "macro_f1_fixed_three_labels",
                        "balanced_accuracy_present_labels",
                        "qwk",
                        "hard_to_easy_rate_among_hard",
                    )
                ]
                lines.append(
                    f"| {method} | {name} | {values['rows']} | "
                    + " | ".join(values_fmt)
                    + " |"
                )
    lines += [
        "",
        "Raw scores and token/truncation data remain in the sealed inference artifact. Per-question projections are in predictions.parquet; category/domain/source metrics are in category-domain-source-metrics.csv. Both directions of the conditional confusion table, label and predicted support, and paired robustness are in summary.json. Missing strata or undefined statistics are N/A, not zero performance.",
        "",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines))
    return seal(
        args.output_dir,
        {
            "schema_version": VERSION,
            "role": "report",
            "prepared_manifest_sha256": sha(args.prepared_dir / "manifest.json"),
            "score_manifest_sha256": sha(args.scores_dir / "manifest.json"),
            "protocol_sha256": prepared["protocol_sha256"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--raw-dir", type=Path, required=True)
    prep.add_argument(
        "--arena-config",
        type=Path,
        default=Path("ml/config/routerarena_preprocessing.json"),
    )
    prep.add_argument("--development-metadata", type=Path, required=True)
    prep.add_argument("--development-config", type=Path, required=True)
    prep.add_argument("--validation-lock", type=Path, required=True)
    prep.add_argument("--run-root", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    inference = sub.add_parser("infer")
    inference.add_argument("--prepared-dir", type=Path, required=True)
    inference.add_argument(
        "--model", action="append", required=True, help="method=/local/final/checkpoint"
    )
    inference.add_argument("--device", default="auto")
    inference.add_argument("--output-dir", type=Path, required=True)
    result = sub.add_parser("report")
    result.add_argument("--prepared-dir", type=Path, required=True)
    result.add_argument("--scores-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = {"prepare": prepare, "infer": infer, "report": report}[args.stage](args)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
