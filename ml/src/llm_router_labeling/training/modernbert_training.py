from __future__ import annotations

import json
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..shared.dataset_io import write_json
from .modernbert_runtime import (
    RuntimeContract,
    collect_runtime_versions,
    validate_runtime_contract,
)


LABEL_ORDER = ("easy", "medium", "hard")
LABEL2ID = {label: index for index, label in enumerate(LABEL_ORDER)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
REQUIRED_TRAINING_COLUMNS = {
    "sample_id",
    "model_input_text",
    "difficulty_label",
    "sample_weight",
    "split",
    "duplicate_group_id",
}


class ModernBertTrainingError(RuntimeError):
    """Raised when training inputs or outputs violate the shared contract."""


@dataclass(frozen=True)
class TrainingConfig:
    training_version: str
    model_id: str
    model_revision: str
    train_split: str
    validation_split: str
    max_length: int
    truncation: bool
    train_batch_size: int
    eval_batch_size: int
    epochs: int
    max_train_steps: int | None
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    seed: int
    device: str
    precision: str
    attention_implementation: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TrainingConfig":
        configured_label2id = {
            str(label): int(index)
            for label, index in values.get("label2id", {}).items()
        }
        configured_id2label = {
            int(index): str(label)
            for index, label in values.get("id2label", {}).items()
        }
        if configured_label2id != LABEL2ID or configured_id2label != ID2LABEL:
            raise ModernBertTrainingError(
                "Training labels must be easy=0, medium=1, hard=2"
            )
        max_steps = values.get("max_train_steps")
        config = cls(
            training_version=str(values["training_version"]),
            model_id=str(values["model_id"]),
            model_revision=str(values["model_revision"]),
            train_split=str(values.get("train_split", "train")),
            validation_split=str(values.get("validation_split", "validation")),
            max_length=int(values["max_length"]),
            truncation=bool(values.get("truncation", True)),
            train_batch_size=int(values["train_batch_size"]),
            eval_batch_size=int(values["eval_batch_size"]),
            epochs=int(values["epochs"]),
            max_train_steps=None if max_steps is None else int(max_steps),
            gradient_accumulation_steps=int(
                values.get("gradient_accumulation_steps", 1)
            ),
            gradient_checkpointing=bool(values.get("gradient_checkpointing", False)),
            learning_rate=float(values["learning_rate"]),
            weight_decay=float(values["weight_decay"]),
            warmup_ratio=float(values.get("warmup_ratio", 0.0)),
            seed=int(values["seed"]),
            device=str(values.get("device", "auto")),
            precision=str(values.get("precision", "fp32")),
            attention_implementation=str(
                values.get("attention_implementation", "eager")
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        positive_integers = {
            "max_length": self.max_length,
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "epochs": self.epochs,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
        for name, value in positive_integers.items():
            if value <= 0:
                raise ModernBertTrainingError(f"{name} must be positive")
        if self.max_train_steps is not None and self.max_train_steps <= 0:
            raise ModernBertTrainingError("max_train_steps must be positive or null")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ModernBertTrainingError(
                "learning_rate must be positive and weight_decay non-negative"
            )
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ModernBertTrainingError("warmup_ratio must be in [0, 1)")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ModernBertTrainingError("precision must be fp32, fp16, or bf16")
        if not self.model_revision or self.model_revision == "main":
            raise ModernBertTrainingError("model_revision must be immutable, not main")


@dataclass(frozen=True)
class TrainingResult:
    metrics: dict[str, Any]
    output_dir: Path


def load_training_config(path: Path) -> TrainingConfig:
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise ModernBertTrainingError("Training config must be a JSON object")
    return TrainingConfig.from_mapping(values)


def validate_training_dataframe(dataframe: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_TRAINING_COLUMNS - set(dataframe.columns))
    if missing:
        raise ModernBertTrainingError(f"Training Parquet is missing columns: {missing}")
    if dataframe.empty:
        raise ModernBertTrainingError("Training Parquet is empty")
    if dataframe["sample_id"].duplicated().any():
        raise ModernBertTrainingError("Training Parquet has duplicate sample_id values")
    if dataframe["model_input_text"].fillna("").astype(str).str.strip().eq("").any():
        raise ModernBertTrainingError("Training Parquet has an empty model_input_text")
    if not dataframe["difficulty_label"].isin(LABEL_ORDER).all():
        raise ModernBertTrainingError("Training Parquet has an invalid difficulty label")
    weights = pd.to_numeric(dataframe["sample_weight"], errors="coerce")
    if weights.isna().any() or not weights.gt(0).all():
        raise ModernBertTrainingError("Training sample_weight must be positive numeric")
    group_splits = dataframe.groupby("duplicate_group_id")["split"].nunique()
    if group_splits.gt(1).any():
        raise ModernBertTrainingError("A duplicate group crosses training splits")


def load_training_splits(
    path: Path, config: TrainingConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataframe = pd.read_parquet(path)
    validate_training_dataframe(dataframe)
    train = dataframe.loc[dataframe["split"].eq(config.train_split)].reset_index(
        drop=True
    )
    validation = dataframe.loc[
        dataframe["split"].eq(config.validation_split)
    ].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ModernBertTrainingError(
            f"Both {config.train_split!r} and {config.validation_split!r} must be non-empty"
        )
    return dataframe, train, validation


def classification_metrics(
    actual: Sequence[int], predicted: Sequence[int]
) -> dict[str, Any]:
    if len(actual) != len(predicted) or not actual:
        raise ModernBertTrainingError(
            "Metric inputs must have the same non-zero length"
        )
    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, label in ID2LABEL.items():
        true_positive = int(
            ((actual_array == index) & (predicted_array == index)).sum()
        )
        false_positive = int(
            ((actual_array != index) & (predicted_array == index)).sum()
        )
        false_negative = int(
            ((actual_array == index) & (predicted_array != index)).sum()
        )
        support = int((actual_array == index).sum())
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)
    return {
        "row_count": len(actual),
        "accuracy": float((actual_array == predicted_array).mean()),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
    }


def validate_checkpoint_contract(output_dir: Path) -> dict[str, Any]:
    config_path = output_dir / "config.json"
    tokenizer_config_path = output_dir / "tokenizer_config.json"
    if not config_path.is_file() or not tokenizer_config_path.is_file():
        raise ModernBertTrainingError(
            "Checkpoint must contain config.json and tokenizer_config.json"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)
    id2label = {
        int(index): str(label)
        for index, label in model_config.get("id2label", {}).items()
    }
    label2id = {
        str(label): int(index)
        for label, index in model_config.get("label2id", {}).items()
    }
    if id2label != ID2LABEL or label2id != LABEL2ID:
        raise ModernBertTrainingError(
            f"Checkpoint label contract mismatch: {id2label=}, {label2id=}"
        )
    model_files = [
        path.name
        for path in output_dir.iterdir()
        if path.name in {"model.safetensors", "pytorch_model.bin"}
    ]
    if not model_files:
        raise ModernBertTrainingError("Checkpoint has no saved model weights")
    return {
        "id2label": {str(index): label for index, label in ID2LABEL.items()},
        "label2id": LABEL2ID,
        "model_weight_files": sorted(model_files),
    }


def _require_training_backends() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        import transformers
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as error:
        raise RuntimeError(
            "ModernBERT training requires optional dependencies: "
            "install the platform PyTorch 2.5.1 package, then "
            "pip install -r ml/requirements-training.lock"
        ) from error
    return (
        torch,
        functional,
        transformers,
        (AutoModelForSequenceClassification, AutoTokenizer),
        get_linear_schedule_with_warmup,
    )


def _resolve_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(
    training_path: Path,
    output_dir: Path,
    config: TrainingConfig,
    *,
    runtime_contract: RuntimeContract,
    device_override: str | None = None,
    max_train_steps_override: int | None = None,
) -> TrainingResult:
    torch, functional, transformers, transformer_classes, scheduler_factory = (
        _require_training_backends()
    )
    AutoModelForSequenceClassification, AutoTokenizer = transformer_classes
    all_rows, train_rows, validation_rows = load_training_splits(
        training_path, config
    )
    requested_device = device_override or config.device
    device = _resolve_device(torch, requested_device)
    runtime_report = validate_runtime_contract(
        runtime_contract,
        collect_runtime_versions(torch, transformers),
        device=device,
    )
    if config.precision != "fp32" and not device.startswith("cuda"):
        raise ModernBertTrainingError(
            f"{config.precision} training is only enabled for CUDA, got {device}"
        )
    _set_seed(torch, config.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id, revision=config.model_revision
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        num_labels=len(LABEL_ORDER),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        attn_implementation=config.attention_implementation,
    )
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)

    class WeightedDataset(torch.utils.data.Dataset):
        def __init__(self, dataframe: pd.DataFrame) -> None:
            self.rows = dataframe

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, Any]:
            row = self.rows.iloc[index]
            return {
                "text": str(row["model_input_text"]),
                "label": LABEL2ID[str(row["difficulty_label"])],
                "weight": float(row["sample_weight"]),
            }

    def collate(records: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [record["text"] for record in records],
            padding=True,
            truncation=config.truncation,
            max_length=config.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(
            [record["label"] for record in records], dtype=torch.long
        )
        encoded["sample_weight"] = torch.tensor(
            [record["weight"] for record in records], dtype=torch.float32
        )
        return encoded

    generator = torch.Generator().manual_seed(config.seed)
    train_loader = torch.utils.data.DataLoader(
        WeightedDataset(train_rows),
        batch_size=config.train_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
    )
    validation_loader = torch.utils.data.DataLoader(
        WeightedDataset(validation_rows),
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    steps_per_epoch = math.ceil(
        len(train_loader) / config.gradient_accumulation_steps
    )
    planned_steps = steps_per_epoch * config.epochs
    maximum_steps = (
        max_train_steps_override
        if max_train_steps_override is not None
        else config.max_train_steps
    )
    total_steps = min(planned_steps, maximum_steps or planned_steps)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = scheduler_factory(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    use_scaler = config.precision == "fp16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    autocast_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(config.precision)

    def precision_context() -> Any:
        if autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    batch_losses: list[float] = []
    started = time.perf_counter()
    for _epoch in range(config.epochs):
        for batch_index, batch in enumerate(train_loader, start=1):
            weights = batch.pop("sample_weight").to(device)
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            with precision_context():
                logits = model(**inputs).logits
                row_losses = functional.cross_entropy(
                    logits, labels, reduction="none"
                )
                loss = (row_losses * weights).sum() / weights.sum()
                scaled_loss = loss / config.gradient_accumulation_steps
            batch_losses.append(float(loss.detach().cpu()))
            scaler.scale(scaled_loss).backward()
            should_step = (
                batch_index % config.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if optimizer_steps >= total_steps:
                break
        if optimizer_steps >= total_steps:
            break
    training_seconds = time.perf_counter() - started

    model.eval()
    actual: list[int] = []
    predicted: list[int] = []
    validation_losses: list[float] = []
    with torch.inference_mode():
        for batch in validation_loader:
            weights = batch.pop("sample_weight").to(device)
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            with precision_context():
                logits = model(**inputs).logits
                row_losses = functional.cross_entropy(
                    logits, labels, reduction="none"
                )
                loss = (row_losses * weights).sum() / weights.sum()
            validation_losses.append(float(loss.cpu()))
            actual.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(dim=-1).cpu().tolist())

    token_counts = [
        len(
            tokenizer(
                str(text), add_special_tokens=True, truncation=False
            )["input_ids"]
        )
        for text in all_rows["model_input_text"]
    ]
    validation_metrics = classification_metrics(actual, predicted)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    checkpoint_contract = validate_checkpoint_contract(output_dir)

    del optimizer, model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()

    reloaded_model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    reloaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    reloaded_model.to(device)
    reloaded_model.eval()
    reload_inputs = reloaded_tokenizer(
        [str(validation_rows.iloc[0]["model_input_text"])],
        truncation=config.truncation,
        max_length=config.max_length,
        return_tensors="pt",
    )
    reload_inputs = {key: value.to(device) for key, value in reload_inputs.items()}
    with torch.inference_mode():
        reload_index = int(reloaded_model(**reload_inputs).logits.argmax(dim=-1).item())
    reload_label = str(reloaded_model.config.id2label[reload_index])
    if reload_label not in LABEL_ORDER:
        raise ModernBertTrainingError(
            f"Reloaded checkpoint produced unknown label: {reload_label}"
        )

    metrics: dict[str, Any] = {
        "training_version": config.training_version,
        "runtime": runtime_report,
        "model": {
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "device": device,
            "precision": config.precision,
            "attention_implementation": config.attention_implementation,
        },
        "data": {
            "input_row_count": len(all_rows),
            "train_row_count": len(train_rows),
            "validation_row_count": len(validation_rows),
            "label_distribution": all_rows["difficulty_label"]
            .value_counts()
            .to_dict(),
        },
        "tokenization": {
            "max_length": config.max_length,
            "truncation": config.truncation,
            "truncated_row_count": sum(
                count > config.max_length for count in token_counts
            ),
            "token_count_min": min(token_counts),
            "token_count_median": float(np.median(token_counts)),
            "token_count_max": max(token_counts),
        },
        "optimization": {
            "optimizer_steps": optimizer_steps,
            "planned_steps": planned_steps,
            "max_train_steps": maximum_steps,
            "mean_train_loss": float(np.mean(batch_losses)),
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "warmup_steps": warmup_steps,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "training_seconds": training_seconds,
        },
        "validation": {
            **validation_metrics,
            "loss": float(np.mean(validation_losses)),
        },
        "checkpoint": {
            **checkpoint_contract,
            "reload_smoke_prediction": reload_label,
            "reload_smoke_passed": True,
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "training_config.json", asdict(config))
    return TrainingResult(metrics=metrics, output_dir=output_dir)
