from __future__ import annotations

import hashlib
import json
import platform
import resource
import time
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .evaluation import LABELS, evaluate_classification_predictions


class TfidfBaselineError(ValueError):
    """Raised when TF-IDF baseline inputs violate the frozen experiment."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TfidfBaselineError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if platform.system() == "Darwin" else value / 1024.0


def _validate_splits(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    required = {
        "sample_id",
        "model_input_text",
        "difficulty_label",
        "duplicate_group_id",
        "source_dataset",
        "category",
        "split",
    }
    for name, frame, expected_split in (
        ("train", train, "train"),
        ("validation", validation, "validation"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise TfidfBaselineError(f"{name} data is missing columns: {missing}")
        if frame.empty or frame["sample_id"].duplicated().any():
            raise TfidfBaselineError(f"{name} data must be non-empty with unique sample_id")
        if set(frame["split"].astype(str)) != {expected_split}:
            raise TfidfBaselineError(f"{name} data contains a non-{expected_split} row")
        if not frame["difficulty_label"].isin(LABELS).all():
            raise TfidfBaselineError(f"{name} data contains an invalid label")
    overlap = set(train["duplicate_group_id"].astype(str)) & set(
        validation["duplicate_group_id"].astype(str)
    )
    if overlap:
        raise TfidfBaselineError("A duplicate group crosses train and validation")


def _latency_sample(pipeline: Pipeline, texts: pd.Series, seed: int) -> list[float]:
    sample = texts.sample(n=min(256, len(texts)), random_state=seed).astype(str).tolist()
    for text in sample[: min(20, len(sample))]:
        pipeline.predict_proba([text])
    latency: list[float] = []
    for text in sample:
        started = time.perf_counter_ns()
        pipeline.predict_proba([text])
        latency.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return latency


def run_tfidf_baseline(
    *,
    training_path: Path,
    validation_path: Path,
    dataset_manifest_path: Path,
    experiment_config_path: Path,
    output_dir: Path,
    threshold_role: str,
) -> dict[str, Any]:
    if threshold_role not in {"t1", "t2"}:
        raise TfidfBaselineError("threshold_role must be t1 or t2")
    dataset_manifest = _read_json(dataset_manifest_path)
    if dataset_manifest.get("coverage_resolution_status") != "complete":
        raise TfidfBaselineError("Dataset coverage resolutions are not complete")
    config = _read_json(experiment_config_path)
    train = pd.read_parquet(training_path)
    validation = pd.read_parquet(validation_path)
    _validate_splits(train, validation)

    tfidf = config["tfidf"]
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=tuple(tfidf["ngram_range"]),
                    min_df=int(tfidf["min_df"]),
                    sublinear_tf=bool(tfidf["sublinear_tf"]),
                    max_features=int(tfidf["max_features"]),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=float(tfidf["C"]),
                    class_weight=tfidf["class_weight"],
                    max_iter=int(tfidf["max_iter"]),
                    random_state=int(config["seed"]),
                    solver="lbfgs",
                ),
            ),
        ]
    )
    peak_before = _peak_rss_mb()
    started = time.perf_counter()
    pipeline.fit(train["model_input_text"].astype(str), train["difficulty_label"].astype(str))
    fit_seconds = time.perf_counter() - started
    peak_after = _peak_rss_mb()
    classifier = pipeline.named_steps["classifier"]
    class_indices = [list(classifier.classes_).index(label) for label in LABELS]
    probabilities = pipeline.predict_proba(validation["model_input_text"].astype(str))[
        :, class_indices
    ]
    latency = _latency_sample(
        pipeline, validation["model_input_text"], seed=int(config["seed"])
    )
    predictions, metrics = evaluate_classification_predictions(
        validation, probabilities, latency_ms=latency
    )
    metrics["fit_seconds"] = float(fit_seconds)
    metrics["peak_rss_mb"] = float(max(peak_before, peak_after))
    metrics["resource"] = {"device": "cpu", "peak_vram_mb": 0.0}

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    predictions_path = output_dir / "validation-predictions.parquet"
    metrics_path = output_dir / "metrics.json"
    joblib.dump(pipeline, model_path, compress=3)
    predictions.to_parquet(predictions_path, index=False)
    metrics["artifact_size_bytes"] = int(model_path.stat().st_size)
    _write_json(metrics_path, metrics)
    manifest = {
        "schema_version": "router-model-comparison-run-v1",
        "experiment_version": config["experiment_version"],
        "model_id": "tfidf-logreg",
        "model_family": "tfidf-logistic-regression",
        "threshold_role": threshold_role,
        "seed": int(config["seed"]),
        "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        "training_data_sha256": _sha256(training_path),
        "validation_data_sha256": _sha256(validation_path),
        "experiment_config_sha256": _sha256(experiment_config_path),
        "test_data_accessed": False,
        "training_contract": {
            "distribution": "natural",
            "sample_weight_used": False,
            "class_weight": tfidf["class_weight"],
            "tfidf": tfidf,
        },
        "artifacts": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (model_path, predictions_path, metrics_path)
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    (output_dir / "_SUCCESS").write_text(_sha256(manifest_path) + "\n", encoding="utf-8")
    return {"manifest": manifest, "metrics": metrics}
