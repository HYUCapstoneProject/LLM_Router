from __future__ import annotations

import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..shared.dataset_io import write_json
from .data import TIERS
from .evaluation import evaluate_penalty_curve, summarize_curve
from .objectives import METHODS, objective_loss, require_torch


LABEL_TO_ID = {"easy": 0, "medium": 1, "hard": 2}


class ObjectiveTrainingError(ValueError):
    """Raised when an objective-comparison training run is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ObjectiveTrainingError(f"Expected JSON object: {path}")
    return value


def _require_transformers() -> tuple[Any, Any, Any]:
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as error:
        raise RuntimeError(
            "Routing-objective training requires transformers; install the model-training extras"
        ) from error
    return AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def _set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _validate_frame(frame: pd.DataFrame, split: str, label_column: str) -> None:
    required = {
        "sample_id",
        "model_input_text",
        "source_dataset",
        "category",
        "sample_weight",
        label_column,
        *{f"success_{tier}" for tier in TIERS},
        *{f"quality_{tier}" for tier in TIERS},
        *{f"decision_cost_{tier}" for tier in TIERS},
        *{f"actual_cost_{tier}" for tier in TIERS},
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ObjectiveTrainingError(f"{split} data is missing columns: {missing}")
    if frame.empty or frame["sample_id"].duplicated().any():
        raise ObjectiveTrainingError(f"{split} rows must be non-empty and unique")
    if not frame[label_column].isin(LABEL_TO_ID).all():
        raise ObjectiveTrainingError(f"{split} contains invalid labels in {label_column}")


def _sample_smoke(frame: pd.DataFrame, maximum_rows: int, seed: int) -> pd.DataFrame:
    if maximum_rows <= 0 or len(frame) <= maximum_rows:
        return frame.reset_index(drop=True)
    sampled: list[pd.DataFrame] = []
    grouped = frame.groupby(["source_dataset", "difficulty_label"], sort=True)
    group_count = max(len(grouped), 1)
    per_group = max(maximum_rows // group_count, 1)
    for offset, (_, rows) in enumerate(grouped):
        sampled.append(
            rows.sample(n=min(per_group, len(rows)), random_state=seed + offset)
        )
    output = pd.concat(sampled, ignore_index=True)
    if len(output) > maximum_rows:
        output = output.sample(n=maximum_rows, random_state=seed)
    return output.reset_index(drop=True)


def _score_frame(frame: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    preferred = [
        "sample_id",
        "source_dataset",
        "category",
        "difficulty_label",
        "difficulty_label_t2",
        "evaluation_split",
        *[f"quality_{tier}" for tier in TIERS],
        *[f"success_{tier}" for tier in TIERS],
        *[f"actual_cost_{tier}" for tier in TIERS],
        *[f"decision_cost_{tier}" for tier in TIERS],
    ]
    output = frame.loc[:, [column for column in preferred if column in frame.columns]].copy()
    for index, tier in enumerate(TIERS):
        output[f"score_{tier}"] = scores[:, index]
    return output


def _predict_logits(
    model: Any,
    tokenizer: Any,
    frame: pd.DataFrame,
    *,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    torch = require_torch()
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            texts = frame.iloc[start : start + batch_size]["model_input_text"].astype(str).tolist()
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.detach().float().cpu().numpy()
            batches.append(logits)
    return np.concatenate(batches, axis=0)


def _batch_one_latency_ms(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
    device: str,
) -> list[float]:
    torch = require_torch()
    sample = list(texts)[:64]
    if not sample:
        return []
    model.eval()
    latency: list[float] = []
    with torch.no_grad():
        for index, text in enumerate(sample):
            encoded = tokenizer(
                [str(text)],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            started = time.perf_counter_ns()
            model(**encoded)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
            if index >= 5:
                latency.append(float(elapsed))
    return latency


def train_objective(
    *,
    method: str,
    training_path: Path,
    validation_path: Path,
    outcome_manifest_path: Path,
    experiment_config_path: Path,
    output_dir: Path,
    seed: int,
    label_column: str = "difficulty_label",
    device: str = "auto",
    maximum_train_rows: int = 0,
    maximum_validation_rows: int = 0,
    maximum_train_steps: int = 0,
    initial_model_dir: Path | None = None,
) -> dict[str, Any]:
    """Train one routing objective.

    `initial_model_dir` starts from a saved low/medium/high classifier (weights only; the optimizer and
    scheduler are new) instead of the pinned pretrained backbone (RF-06 warm starts).
    """
    if method not in METHODS:
        raise ObjectiveTrainingError(f"method must be one of {METHODS}")
    config = _load_json(experiment_config_path)
    outcome_manifest = _load_json(outcome_manifest_path)
    train = pd.read_parquet(training_path)
    validation = pd.read_parquet(validation_path)
    _validate_frame(train, "train", label_column)
    _validate_frame(validation, "validation", label_column)
    train = _sample_smoke(train, maximum_train_rows, seed)
    validation = _sample_smoke(validation, maximum_validation_rows, seed + 1)

    torch = require_torch()
    AutoModelForSequenceClassification, AutoTokenizer, scheduler_factory = (
        _require_transformers()
    )
    _set_seed(torch, seed)
    resolved_device = _device(torch, device)
    backbone = config["backbone"]
    training = config["training"]
    if initial_model_dir is None:
        tokenizer = AutoTokenizer.from_pretrained(
            backbone["model_id"], revision=backbone["revision"]
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            backbone["model_id"],
            revision=backbone["revision"],
            num_labels=3,
            id2label={0: "low", 1: "medium", 2: "high"},
            label2id={"low": 0, "medium": 1, "high": 2},
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(initial_model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(initial_model_dir)
        labels = [model.config.id2label[index] for index in range(model.config.num_labels)]
        if labels != list(TIERS):
            raise ObjectiveTrainingError(
                f"initial model outputs {labels}, expected {list(TIERS)}"
            )
    model.to(resolved_device)
    if resolved_device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    class RoutingDataset(torch.utils.data.Dataset):
        def __init__(self, rows: pd.DataFrame) -> None:
            self.rows = rows.reset_index(drop=True)

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            row = self.rows.iloc[index]
            return {
                "text": str(row["model_input_text"]),
                "difficulty_id": LABEL_TO_ID[str(row[label_column])],
                "success_targets": [float(row[f"success_{tier}"]) for tier in TIERS],
                "quality_targets": [float(row[f"quality_{tier}"]) for tier in TIERS],
                "decision_costs": [float(row[f"decision_cost_{tier}"]) for tier in TIERS],
                "sample_weight": float(row["sample_weight"]),
            }

    max_length = int(backbone["max_length"])

    def collate(records: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [record["text"] for record in records],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["difficulty_ids"] = torch.tensor(
            [record["difficulty_id"] for record in records], dtype=torch.long
        )
        for key in ("success_targets", "quality_targets", "decision_costs"):
            encoded[key] = torch.tensor(
                [record[key] for record in records], dtype=torch.float32
            )
        encoded["sample_weights"] = torch.tensor(
            [record["sample_weight"] for record in records], dtype=torch.float32
        )
        return encoded

    generator = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        RoutingDataset(train),
        batch_size=int(training["train_batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    accumulation = int(training["gradient_accumulation_steps"])
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    planned_steps = int(training["epochs"]) * steps_per_epoch
    total_steps = min(planned_steps, maximum_train_steps or planned_steps)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = scheduler_factory(
        optimizer,
        num_warmup_steps=int(total_steps * float(training["warmup_ratio"])),
        num_training_steps=total_steps,
    )
    use_amp = resolved_device.startswith("cuda") and training.get("precision") == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    optimizer_step = 0
    loss_history: list[dict[str, float | int]] = []
    stop = False
    for epoch in range(int(training["epochs"])):
        model.train()
        epoch_loss = 0.0
        micro_batches = 0
        for batch_index, batch in enumerate(train_loader):
            metadata = {
                key: batch.pop(key).to(resolved_device)
                for key in (
                    "difficulty_ids",
                    "success_targets",
                    "quality_targets",
                    "decision_costs",
                    "sample_weights",
                )
            }
            inputs = {key: value.to(resolved_device) for key, value in batch.items()}
            with torch.autocast(
                device_type="cuda" if resolved_device.startswith("cuda") else "cpu",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(**inputs).logits
                loss = objective_loss(
                    method,
                    logits,
                    difficulty_ids=metadata["difficulty_ids"],
                    success_targets=metadata["success_targets"],
                    quality_targets=metadata["quality_targets"],
                    decision_costs=metadata["decision_costs"],
                    sample_weights=metadata["sample_weights"],
                    regret_lambdas=training["regret_lambdas"],
                    pairwise_tie_weight=float(training["pairwise_tie_weight"]),
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            epoch_loss += float(loss.detach().cpu())
            micro_batches += 1
            should_step = (batch_index + 1) % accumulation == 0 or (
                batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step += 1
                if optimizer_step >= total_steps:
                    stop = True
                    break
        loss_history.append(
            {
                "epoch": epoch + 1,
                "optimizer_step": optimizer_step,
                "mean_training_loss": epoch_loss / max(micro_batches, 1),
            }
        )
        if stop:
            break

    fit_seconds = time.perf_counter() - started
    inference_started = time.perf_counter()
    scores = _predict_logits(
        model,
        tokenizer,
        validation,
        batch_size=int(training["eval_batch_size"]),
        max_length=max_length,
        device=resolved_device,
    )
    inference_seconds = time.perf_counter() - inference_started
    latency = _batch_one_latency_ms(
        model,
        tokenizer,
        validation["model_input_text"].astype(str).tolist(),
        max_length=max_length,
        device=resolved_device,
    )
    scored_validation = _score_frame(validation, scores)
    curve = evaluate_penalty_curve(
        validation,
        scores,
        method=method,
        penalties=config["evaluation"]["penalty_grid"],
    )
    summary, budget_penalties = summarize_curve(
        validation,
        curve,
        budgets=config["evaluation"]["relative_cost_budgets"],
        retention_targets=config["evaluation"]["quality_retention_targets"],
    )
    metrics = {
        "method": method,
        "seed": seed,
        "label_column": label_column,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "optimizer_steps": optimizer_step,
        "fit_seconds": float(fit_seconds),
        "validation_inference_seconds": float(inference_seconds),
        "validation_rows_per_second": float(len(validation) / max(inference_seconds, 1e-12)),
        "device": resolved_device,
        "peak_vram_mb": (
            float(torch.cuda.max_memory_allocated() / (1024**2))
            if resolved_device.startswith("cuda")
            else 0.0
        ),
        "batch_1_latency_ms": {
            "sample_count": len(latency),
            "mean": float(np.mean(latency)) if latency else None,
            "p50": float(np.quantile(latency, 0.50)) if latency else None,
            "p95": float(np.quantile(latency, 0.95)) if latency else None,
        },
        "loss_history": loss_history,
        "summary": summary,
        "validation_budget_penalties": budget_penalties,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    scores_path = output_dir / "validation-scores.parquet"
    curve_path = output_dir / "validation-curve.parquet"
    metrics_path = output_dir / "metrics.json"
    scored_validation.to_parquet(scores_path, index=False)
    curve.to_parquet(curve_path, index=False)
    write_json(metrics_path, metrics)
    model_files = sorted(path for path in model_dir.rglob("*") if path.is_file())
    manifest = {
        "schema_version": "routing-objective-run-v1",
        "experiment_version": config["experiment_version"],
        "outcome_dataset_version": outcome_manifest["dataset_version"],
        "method": method,
        "seed": seed,
        "label_column": label_column,
        "outcome_manifest_sha256": _sha256(outcome_manifest_path),
        "experiment_config_sha256": _sha256(experiment_config_path),
        "training_data_sha256": _sha256(training_path),
        "validation_data_sha256": _sha256(validation_path),
        "test_data_accessed": False,
        "routerarena_accessed": False,
        "maximum_train_rows": maximum_train_rows,
        "maximum_validation_rows": maximum_validation_rows,
        "maximum_train_steps": maximum_train_steps,
        "initial_model_sha256": (
            None
            if initial_model_dir is None
            else _sha256(Path(initial_model_dir) / "model.safetensors")
        ),
        "model_artifact_size_bytes": int(sum(path.stat().st_size for path in model_files)),
        "model_artifacts": {
            str(path.relative_to(model_dir)): {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in model_files
        },
        "artifacts": {
            path.name: {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (scores_path, curve_path, metrics_path)
        },
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    (output_dir / "_SUCCESS").write_text(_sha256(manifest_path) + "\n", encoding="utf-8")
    return {"manifest": manifest, "metrics": metrics}


def infer_scores(
    *,
    model_dir: Path,
    data_path: Path,
    output_path: Path,
    experiment_config_path: Path,
    device: str = "auto",
) -> dict[str, Any]:
    config = _load_json(experiment_config_path)
    frame = pd.read_parquet(data_path)
    torch = require_torch()
    AutoModelForSequenceClassification, AutoTokenizer, _ = _require_transformers()
    resolved_device = _device(torch, device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(resolved_device)
    scores = _predict_logits(
        model,
        tokenizer,
        frame,
        batch_size=int(config["training"]["eval_batch_size"]),
        max_length=int(config["backbone"]["max_length"]),
        device=resolved_device,
    )
    scored = _score_frame(frame, scores)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output_path, index=False)
    return {
        "row_count": int(len(scored)),
        "output_path": str(output_path),
        "sha256": _sha256(output_path),
        "device": resolved_device,
    }
