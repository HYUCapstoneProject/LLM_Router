#!/usr/bin/env python3
"""Cross-encoder 후보를 시맨틱 캐시용 질문 쌍에서 평가한다.

평가 지표 계산은 표준 라이브러리만 사용한다. 실제 모델 추론을 실행할 때만
sentence-transformers가 필요하다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


VALID_LABELS = {"positive", "negative"}


def compute_metrics(labels: Sequence[str], scores: Sequence[float], threshold: float) -> dict:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")

    tp = fp = tn = fn = 0
    for index, (label, score) in enumerate(zip(labels, scores)):
        if label not in VALID_LABELS:
            raise ValueError(f"unknown label at index {index}: {label!r}")
        if not math.isfinite(float(score)):
            raise ValueError(f"score at index {index} must be finite")

        predicted_positive = float(score) >= threshold
        if label == "positive" and predicted_positive:
            tp += 1
        elif label == "positive":
            fn += 1
        elif predicted_positive:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    false_hit_rate = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "false_hit_rate": false_hit_rate,
        "f1": f1,
    }


def threshold_sweep(labels: Sequence[str], scores: Sequence[float]) -> list[dict]:
    if not scores:
        return []
    unique = sorted({float(score) for score in scores}, reverse=True)
    return [compute_metrics(labels, scores, threshold) for threshold in unique]


def select_threshold(
    labels: Sequence[str],
    scores: Sequence[float],
    *,
    max_false_hits: int,
    min_recall: float = 0.0,
) -> dict | None:
    candidates = [
        result
        for result in threshold_sweep(labels, scores)
        if result["fp"] <= max_false_hits and result["recall"] >= min_recall
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda result: (
            result["recall"],
            result["precision"],
            result["f1"],
            -result["threshold"],
        ),
    )


def select_min_false_hits(
    labels: Sequence[str],
    scores: Sequence[float],
    *,
    min_recall: float,
) -> dict | None:
    candidates = [
        result
        for result in threshold_sweep(labels, scores)
        if result["recall"] >= min_recall
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda result: (
            result["fp"],
            -result["recall"],
            -result["precision"],
            result["threshold"],
        ),
    )


def compute_joint_metrics(
    labels: Sequence[str],
    similarities: Sequence[float],
    reranker_scores: Sequence[float],
    similarity_threshold: float,
    reranker_threshold: float,
) -> dict:
    if len(labels) != len(similarities) or len(labels) != len(reranker_scores):
        raise ValueError("labels, similarities and reranker_scores must have the same length")
    combined = []
    for index, (similarity, reranker_score) in enumerate(zip(similarities, reranker_scores)):
        if not math.isfinite(float(similarity)) or not math.isfinite(float(reranker_score)):
            raise ValueError(f"scores at index {index} must be finite")
        combined.append(
            1.0
            if float(similarity) >= similarity_threshold
            and float(reranker_score) >= reranker_threshold
            else 0.0
        )
    result = compute_metrics(labels, combined, threshold=0.5)
    result.pop("threshold")
    return {
        "similarity_threshold": float(similarity_threshold),
        "reranker_threshold": float(reranker_threshold),
        **result,
    }


def select_joint_thresholds(
    labels: Sequence[str],
    similarities: Sequence[float],
    reranker_scores: Sequence[float],
    *,
    max_false_hits: int,
    min_recall: float,
) -> dict | None:
    candidates = []
    for similarity_threshold in sorted({float(value) for value in similarities}, reverse=True):
        for reranker_threshold in sorted({float(value) for value in reranker_scores}, reverse=True):
            result = compute_joint_metrics(
                labels,
                similarities,
                reranker_scores,
                similarity_threshold,
                reranker_threshold,
            )
            if result["fp"] <= max_false_hits and result["recall"] >= min_recall:
                candidates.append(result)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda result: (
            result["recall"],
            result["precision"],
            result["f1"],
            result["similarity_threshold"],
            result["reranker_threshold"],
        ),
    )


def select_joint_min_false_hits(
    labels: Sequence[str],
    similarities: Sequence[float],
    reranker_scores: Sequence[float],
    *,
    min_recall: float,
) -> dict | None:
    candidates = []
    for similarity_threshold in sorted({float(value) for value in similarities}, reverse=True):
        for reranker_threshold in sorted({float(value) for value in reranker_scores}, reverse=True):
            result = compute_joint_metrics(
                labels,
                similarities,
                reranker_scores,
                similarity_threshold,
                reranker_threshold,
            )
            if result["recall"] >= min_recall:
                candidates.append(result)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda result: (
            result["fp"],
            -result["recall"],
            -result["precision"],
            result["similarity_threshold"],
            result["reranker_threshold"],
        ),
    )


def percentile(values: Iterable[float], quantile: float) -> float:
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    rank = max(1, math.ceil(quantile * len(values)))
    return values[rank - 1]


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("evaluation CSV is empty")
    required = {"pair_id", "label", "neg_type", "tier", "query", "candidate"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"evaluation CSV is missing columns: {sorted(missing)}")
    for row in rows:
        if row["label"] not in VALID_LABELS:
            raise ValueError(f"unknown label for {row['pair_id']}: {row['label']!r}")
    return rows


def model_slug(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "--", model_name).strip("-")


def revision_from_snapshot_path(path: str | Path) -> str | None:
    parts = str(path).replace("\\", "/").split("/")
    try:
        return parts[parts.index("snapshots") + 1]
    except (ValueError, IndexError):
        return None


def encode_text_pair(tokenizer, query: str, candidate: str):
    kwargs = {"padding": "longest", "return_token_type_ids": True}
    encode_plus = getattr(tokenizer, "encode_plus", None)
    if callable(encode_plus):
        return encode_plus(query, candidate, **kwargs)
    return tokenizer(query, candidate, **kwargs)


def current_rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return None


def predict_scores(
    model_name: str,
    rows: Sequence[dict[str, str]],
    *,
    batch_size: int,
    device: str,
    cache_dir: Path,
    backend: str = "sentence-transformers",
) -> tuple[list[float], dict]:
    if backend == "gptcache-onnx":
        return predict_gptcache_onnx_scores(
            model_name,
            rows,
            cache_dir=cache_dir,
        )
    if backend != "sentence-transformers":
        raise ValueError(f"unknown inference backend: {backend}")

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise RuntimeError(
            "sentence-transformers가 필요합니다. 평가용 가상환경에 의존성을 설치하세요."
        ) from error

    rss_before = current_rss_mb()
    load_start = time.perf_counter()
    model = CrossEncoder(model_name, device=device, cache_folder=str(cache_dir))
    load_seconds = time.perf_counter() - load_start
    rss_after_load = current_rss_mb()

    pairs = [(row["query"], row["candidate"]) for row in rows]
    # 첫 추론의 초기화 비용을 latency 분포에서 분리한다.
    model.predict([pairs[0]], batch_size=1, show_progress_bar=False)

    scores: list[float] = []
    per_pair_ms: list[float] = []
    inference_start = time.perf_counter()
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        batch_start = time.perf_counter()
        output = model.predict(batch, batch_size=len(batch), show_progress_bar=False)
        elapsed_ms = (time.perf_counter() - batch_start) * 1000
        per_pair_ms.extend([elapsed_ms / len(batch)] * len(batch))
        for value in output:
            if hasattr(value, "item"):
                value = value.item()
            score = float(value)
            if not math.isfinite(score):
                raise ValueError(f"model returned a non-finite score: {score}")
            scores.append(score)
    inference_seconds = time.perf_counter() - inference_start

    config = getattr(getattr(model, "model", None), "config", None)
    revision = getattr(config, "_commit_hash", None)
    rss_after_inference = current_rss_mb()
    return scores, {
        "model": model_name,
        "revision": revision,
        "device": device,
        "batch_size": batch_size,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "latency_ms": {
            "mean_per_pair": statistics.fmean(per_pair_ms),
            "p50_per_pair": percentile(per_pair_ms, 0.50),
            "p95_per_pair": percentile(per_pair_ms, 0.95),
        },
        "rss_mb": {
            "before_load": rss_before,
            "after_load": rss_after_load,
            "after_inference": rss_after_inference,
        },
    }


def predict_gptcache_onnx_scores(
    model_name: str,
    rows: Sequence[dict[str, str]],
    *,
    cache_dir: Path,
) -> tuple[list[float], dict]:
    """GPTCache OnnxModelEvaluation과 같은 tokenizer/model 출력을 측정한다."""
    try:
        import numpy as np
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "GPTCache ONNX 평가에는 transformers, huggingface-hub, onnxruntime가 필요합니다."
        ) from error

    rss_before = current_rss_mb()
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2", cache_dir=str(cache_dir))
    model_path = hf_hub_download(
        repo_id=model_name,
        filename="model.onnx",
        cache_dir=str(cache_dir),
    )
    session = onnxruntime.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )
    load_seconds = time.perf_counter() - load_start
    rss_after_load = current_rss_mb()

    def score_pair(query: str, candidate: str) -> float:
        if query.lower() == candidate.lower():
            return 1.0
        encoded = encode_text_pair(tokenizer, query, candidate)
        inputs = {
            "input_ids": np.asarray(encoded["input_ids"], dtype="int64").reshape(1, -1),
            "attention_mask": np.asarray(
                encoded["attention_mask"], dtype="int64"
            ).reshape(1, -1),
            "token_type_ids": np.asarray(
                encoded["token_type_ids"], dtype="int64"
            ).reshape(1, -1),
        }
        return float(session.run(None, inputs)[0][0, 1])

    score_pair(rows[0]["query"], rows[0]["candidate"])
    scores: list[float] = []
    per_pair_ms: list[float] = []
    inference_start = time.perf_counter()
    for row in rows:
        pair_start = time.perf_counter()
        score = score_pair(row["query"], row["candidate"])
        per_pair_ms.append((time.perf_counter() - pair_start) * 1000)
        if not math.isfinite(score):
            raise ValueError(f"model returned a non-finite score: {score}")
        scores.append(score)
    inference_seconds = time.perf_counter() - inference_start

    return scores, {
        "model": model_name,
        "revision": revision_from_snapshot_path(model_path),
        "device": "cpu",
        "batch_size": 1,
        "backend": "gptcache-onnx",
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "latency_ms": {
            "mean_per_pair": statistics.fmean(per_pair_ms),
            "p50_per_pair": percentile(per_pair_ms, 0.50),
            "p95_per_pair": percentile(per_pair_ms, 0.95),
        },
        "rss_mb": {
            "before_load": rss_before,
            "after_load": rss_after_load,
            "after_inference": current_rss_mb(),
        },
    }


def grouped_errors(rows: Sequence[dict[str, str]], scores: Sequence[float], threshold: float) -> dict:
    groups: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        if row["label"] == "negative":
            groups[row["neg_type"] or "unspecified"].append((row["pair_id"], float(score)))
    return {
        group: {
            "count": len(items),
            "false_hits": [pair_id for pair_id, score in items if score >= threshold],
            "max_score": max(score for _, score in items),
        }
        for group, items in sorted(groups.items())
    }


def write_scored_csv(path: Path, rows: Sequence[dict[str, str]], scores: Sequence[float]) -> None:
    fieldnames = list(rows[0].keys())
    if "reranker_score" not in fieldnames:
        fieldnames.append("reranker_score")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, score in zip(rows, scores):
            output = dict(row)
            output["reranker_score"] = f"{score:.8f}"
            writer.writerow(output)


def evaluate_model(
    model_name: str,
    rows: Sequence[dict[str, str]],
    *,
    batch_size: int,
    device: str,
    cache_dir: Path,
    output_dir: Path,
    max_false_hits: int,
    min_recall: float,
    backend: str = "sentence-transformers",
) -> dict:
    scores, runtime = predict_scores(
        model_name,
        rows,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
        backend=backend,
    )
    labels = [row["label"] for row in rows]
    selected = select_threshold(
        labels,
        scores,
        max_false_hits=max_false_hits,
        min_recall=min_recall,
    )
    best_zero_fp = select_threshold(labels, scores, max_false_hits=0)
    best_at_recall_gate = select_min_false_hits(labels, scores, min_recall=min_recall)
    best_f1 = max(threshold_sweep(labels, scores), key=lambda result: result["f1"])
    similarities = [float(row["bi_encoder_similarity"]) for row in rows]
    selected_joint = select_joint_thresholds(
        labels,
        similarities,
        scores,
        max_false_hits=max_false_hits,
        min_recall=min_recall,
    )
    best_joint_at_recall_gate = select_joint_min_false_hits(
        labels,
        similarities,
        scores,
        min_recall=min_recall,
    )

    slug = model_slug(model_name)
    write_scored_csv(output_dir / f"{slug}.csv", rows, scores)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "pairs": len(rows),
            "positive": labels.count("positive"),
            "negative": labels.count("negative"),
        },
        "runtime": runtime,
        "score_distribution": {
            "positive": {
                "min": min(score for label, score in zip(labels, scores) if label == "positive"),
                "mean": statistics.fmean(score for label, score in zip(labels, scores) if label == "positive"),
                "max": max(score for label, score in zip(labels, scores) if label == "positive"),
            },
            "negative": {
                "min": min(score for label, score in zip(labels, scores) if label == "negative"),
                "mean": statistics.fmean(score for label, score in zip(labels, scores) if label == "negative"),
                "max": max(score for label, score in zip(labels, scores) if label == "negative"),
            },
        },
        "selected_threshold": selected,
        "best_zero_false_hit_threshold": best_zero_fp,
        "best_at_recall_gate": best_at_recall_gate,
        "best_f1_threshold": best_f1,
        "selected_joint_thresholds": selected_joint,
        "best_joint_at_recall_gate": best_joint_at_recall_gate,
    }
    if selected is not None:
        report["negative_type_errors"] = grouped_errors(rows, scores, selected["threshold"])
    with (output_dir / f"{slug}.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("rerank-eval-pairs.csv"),
    )
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf-cache"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--backend",
        choices=("sentence-transformers", "gptcache-onnx"),
        default="sentence-transformers",
    )
    parser.add_argument("--max-false-hits", type=int, default=0)
    parser.add_argument("--min-recall", type=float, default=0.90)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rows = read_pairs(args.input)

    summary = []
    for model_name in args.model:
        print(f"Evaluating {model_name} ...", flush=True)
        report = evaluate_model(
            model_name,
            rows,
            batch_size=args.batch_size,
            device=args.device,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            max_false_hits=args.max_false_hits,
            min_recall=args.min_recall,
            backend=args.backend,
        )
        summary.append(report)
        selected = report["selected_threshold"]
        print(f"  selected={selected}", flush=True)

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
