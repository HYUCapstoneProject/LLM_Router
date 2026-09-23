from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..shared.dataset_io import sha256_file, write_json
from .modernbert_checkpointing import (
    CheckpointOptions,
    ModernBertCheckpointError,
    create_checkpoint,
    next_training_cursor,
    prune_checkpoints,
    training_fingerprint,
    verify_checkpoint,
)
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
    train_sampling_strategy: str = "shuffle"

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
            train_sampling_strategy=str(
                values.get("train_sampling_strategy", "shuffle")
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
        if self.train_sampling_strategy not in {
            "shuffle",
            "balanced_effective_batch",
        }:
            raise ModernBertTrainingError(
                "train_sampling_strategy must be shuffle or balanced_effective_batch"
            )
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


@dataclass(frozen=True)
class TrainingProgressState:
    optimizer_steps: int
    total_steps: int
    epoch_number: int
    total_epochs: int


TrainingProgressCallback = Callable[[TrainingProgressState], None]


@dataclass(frozen=True)
class EpochEndState:
    """Read-only observation point after one complete training epoch."""

    epoch_number: int
    optimizer_steps: int
    training_fingerprint: str
    checkpoint_path: Path
    model: Any
    tokenizer: Any
    device: str
    max_length: int


EpochEndCallback = Callable[[EpochEndState], None]


@dataclass(frozen=True)
class ValidationObservationState:
    """Inference-only RouterBench validation snapshot during an epoch."""

    epoch_number: int
    point_number: int
    points_per_epoch: int
    epoch_fraction: float
    optimizer_steps: int
    training_fingerprint: str
    metrics: dict[str, Any]


ValidationObservationCallback = Callable[[ValidationObservationState], None]


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
        raise ModernBertTrainingError(
            "Training Parquet has an invalid difficulty label"
        )
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
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
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


def balanced_effective_batch_indices(
    dataframe: pd.DataFrame,
    *,
    effective_batch_size: int,
    seed: int,
) -> list[int]:
    """Shuffle rows while keeping every optimizer unit class-balanced."""
    if effective_batch_size <= 0:
        raise ModernBertTrainingError("effective_batch_size must be positive")
    label_counts = dataframe["difficulty_label"].value_counts().to_dict()
    if set(label_counts) != set(LABEL_ORDER) or any(
        int(label_counts[label]) <= 0 for label in LABEL_ORDER
    ):
        raise ModernBertTrainingError(
            "balanced_effective_batch requires non-empty easy, medium, hard; "
            f"got {label_counts}"
        )

    rng = np.random.default_rng(seed)
    class_order = list(LABEL_ORDER)
    rng.shuffle(class_order)
    target_per_class = max(int(label_counts[label]) for label in LABEL_ORDER)
    pools: dict[str, list[int]] = {}
    for label in LABEL_ORDER:
        source_indices = dataframe.index[
            dataframe["difficulty_label"].astype(str).eq(label)
        ].to_numpy(dtype=np.int64)
        sampled_indices: list[int] = []
        while len(sampled_indices) < target_per_class:
            cycle = source_indices.copy()
            rng.shuffle(cycle)
            remaining = target_per_class - len(sampled_indices)
            sampled_indices.extend(cycle[:remaining].tolist())
        pools[label] = sampled_indices

    label_sequence = [
        class_order[position % len(class_order)]
        for position in range(target_per_class * len(LABEL_ORDER))
    ]
    ordered_indices: list[int] = []
    for start in range(0, len(label_sequence), effective_batch_size):
        unit_labels = label_sequence[start : start + effective_batch_size]
        rng.shuffle(unit_labels)
        for label in unit_labels:
            ordered_indices.append(pools[label].pop())
    if any(pools[label] for label in LABEL_ORDER):
        raise ModernBertTrainingError(
            "Balanced sampler did not consume every class pool"
        )
    return ordered_indices


def training_sample_count_per_epoch(
    dataframe: pd.DataFrame, *, sampling_strategy: str
) -> int:
    if sampling_strategy == "shuffle":
        return len(dataframe)
    if sampling_strategy == "balanced_effective_batch":
        counts = dataframe["difficulty_label"].value_counts()
        if set(counts.index.astype(str)) != set(LABEL_ORDER):
            raise ModernBertTrainingError(
                "balanced_effective_batch requires all three labels"
            )
        return int(counts.max()) * len(LABEL_ORDER)
    raise ModernBertTrainingError(
        f"Unsupported train sampling strategy: {sampling_strategy}"
    )


def run_training(
    training_path: Path,
    output_dir: Path,
    config: TrainingConfig,
    *,
    runtime_contract: RuntimeContract,
    device_override: str | None = None,
    max_train_steps_override: int | None = None,
    initial_model_dir: Path | None = None,
    checkpoint_options: CheckpointOptions | None = None,
    on_epoch_end: EpochEndCallback | None = None,
    validation_observations_per_epoch: int = 0,
    on_validation_observation: ValidationObservationCallback | None = None,
    on_progress: TrainingProgressCallback | None = None,
) -> TrainingResult:
    torch, functional, transformers, transformer_classes, scheduler_factory = (
        _require_training_backends()
    )
    AutoModelForSequenceClassification, AutoTokenizer = transformer_classes
    all_rows, train_rows, validation_rows = load_training_splits(training_path, config)
    requested_device = device_override or config.device
    device = _resolve_device(torch, requested_device)
    runtime_report = validate_runtime_contract(
        runtime_contract,
        collect_runtime_versions(torch, transformers),
        device=device,
    )
    if checkpoint_options is not None:
        checkpoint_options.validate()
    if (
        initial_model_dir is not None
        and checkpoint_options is not None
        and checkpoint_options.resume_from is not None
    ):
        raise ModernBertTrainingError(
            "initial_model_dir cannot be combined with checkpoint resume"
        )
    if on_epoch_end is not None and checkpoint_options is None:
        raise ModernBertTrainingError(
            "on_epoch_end requires checkpoint_options so the observed model is durable"
        )
    if validation_observations_per_epoch < 0:
        raise ModernBertTrainingError(
            "validation_observations_per_epoch must be non-negative"
        )
    if validation_observations_per_epoch == 0 and on_validation_observation is not None:
        raise ModernBertTrainingError(
            "on_validation_observation requires a positive observation count"
        )
    if config.precision != "fp32" and not device.startswith("cuda"):
        raise ModernBertTrainingError(
            f"{config.precision} training is only enabled for CUDA, got {device}"
        )
    _set_seed(torch, config.seed)

    sampled_train_rows_per_epoch = training_sample_count_per_epoch(
        train_rows, sampling_strategy=config.train_sampling_strategy
    )
    batches_in_epoch = math.ceil(sampled_train_rows_per_epoch / config.train_batch_size)
    steps_per_epoch = math.ceil(batches_in_epoch / config.gradient_accumulation_steps)
    planned_steps = steps_per_epoch * config.epochs
    maximum_steps = (
        max_train_steps_override
        if max_train_steps_override is not None
        else config.max_train_steps
    )
    total_steps = min(planned_steps, maximum_steps or planned_steps)
    training_data_sha256 = sha256_file(training_path)
    current_fingerprint = training_fingerprint(
        training_data_sha256=training_data_sha256,
        config=asdict(config),
        maximum_steps=maximum_steps,
        runtime_version=runtime_report["runtime_version"],
    )
    resume_dir = (
        checkpoint_options.resume_from if checkpoint_options is not None else None
    )
    if resume_dir is not None:
        verify_checkpoint(resume_dir, expected_fingerprint=current_fingerprint)

    if resume_dir is None and initial_model_dir is None:
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
    else:
        model_source = resume_dir or initial_model_dir
        assert model_source is not None
        tokenizer = AutoTokenizer.from_pretrained(model_source)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
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

    train_dataset = WeightedDataset(train_rows)

    def train_loader_for_epoch(epoch_index: int) -> Any:
        generator = torch.Generator().manual_seed(config.seed + epoch_index)
        if config.train_sampling_strategy == "balanced_effective_batch":
            sampler = balanced_effective_batch_indices(
                train_rows,
                effective_batch_size=(
                    config.train_batch_size * config.gradient_accumulation_steps
                ),
                seed=config.seed + epoch_index,
            )
            return torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config.train_batch_size,
                sampler=sampler,
                collate_fn=collate,
            )
        return torch.utils.data.DataLoader(
            train_dataset,
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
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = scheduler_factory(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    use_scaler = config.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    numeric_train_weights = pd.to_numeric(train_rows["sample_weight"], errors="raise")
    if config.train_sampling_strategy == "balanced_effective_batch":
        train_weight_mean = float(
            train_rows.assign(_sample_weight=numeric_train_weights)
            .groupby("difficulty_label")["_sample_weight"]
            .mean()
            .mean()
        )
    else:
        train_weight_mean = float(numeric_train_weights.mean())
    autocast_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(config.precision)

    def precision_context() -> Any:
        if autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=autocast_dtype)

    def evaluate_validation() -> dict[str, Any]:
        was_training = bool(model.training)
        model.eval()
        actual: list[int] = []
        predicted: list[int] = []
        validation_loss_sum = 0.0
        validation_weighted_loss_sum = 0.0
        validation_weight_sum = 0.0
        validation_row_count = 0
        try:
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
                    validation_loss_sum += float(row_losses.sum().cpu())
                    validation_weighted_loss_sum += float(
                        (row_losses * weights).sum().cpu()
                    )
                    validation_weight_sum += float(weights.sum().cpu())
                    validation_row_count += int(labels.numel())
                    actual.extend(labels.cpu().tolist())
                    predicted.extend(logits.argmax(dim=-1).cpu().tolist())
        finally:
            model.train(was_training)
        return {
            **classification_metrics(actual, predicted),
            "loss": validation_loss_sum / validation_row_count,
            "weighted_loss": (validation_weighted_loss_sum / validation_weight_sum),
        }

    def run_rng_isolated(callback: Callable[[], Any]) -> Any:
        python_random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        torch_random_state = torch.get_rng_state()
        cuda_random_state = (
            torch.cuda.get_rng_state_all() if device.startswith("cuda") else None
        )
        try:
            return callback()
        finally:
            random.setstate(python_random_state)
            np.random.set_state(numpy_random_state)
            torch.set_rng_state(torch_random_state)
            if cuda_random_state is not None:
                torch.cuda.set_rng_state_all(cuda_random_state)

    start_epoch_index = 0
    start_batch_index = 0
    optimizer_steps = 0
    train_loss_sum = 0.0
    train_loss_count = 0
    train_weighted_loss_sum = 0.0
    train_weight_sum = 0.0
    observation_train_loss_sum = 0.0
    observation_train_weighted_loss_sum = 0.0
    observation_train_weight_sum = 0.0
    observation_train_row_count = 0
    observation_start_optimizer_step = 0
    completed_training_seconds = 0.0
    resumed_from_step: int | None = None
    if resume_dir is not None:
        try:
            resume_state = torch.load(
                resume_dir / "training_state.pt",
                map_location=device,
                weights_only=False,
            )
        except Exception as error:
            raise ModernBertCheckpointError(
                f"Failed to load checkpoint state: {resume_dir}"
            ) from error
        if resume_state.get("training_fingerprint") != current_fingerprint:
            raise ModernBertCheckpointError(
                "Checkpoint state fingerprint does not match the current run"
            )
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        scaler.load_state_dict(resume_state["scaler_state_dict"])
        start_epoch_index = int(resume_state["epoch_index"])
        start_batch_index = int(resume_state["next_batch_index"])
        optimizer_steps = int(resume_state["optimizer_steps"])
        train_loss_sum = float(resume_state["train_loss_sum"])
        train_loss_count = int(resume_state["train_loss_count"])
        train_weighted_loss_sum = float(
            resume_state.get("train_weighted_loss_sum", train_loss_sum)
        )
        train_weight_sum = float(resume_state.get("train_weight_sum", train_loss_count))
        observation_train_loss_sum = float(
            resume_state.get("observation_train_loss_sum", 0.0)
        )
        observation_train_weighted_loss_sum = float(
            resume_state.get("observation_train_weighted_loss_sum", 0.0)
        )
        observation_train_weight_sum = float(
            resume_state.get("observation_train_weight_sum", 0.0)
        )
        observation_train_row_count = int(
            resume_state.get("observation_train_row_count", 0)
        )
        observation_start_optimizer_step = int(
            resume_state.get("observation_start_optimizer_step", optimizer_steps)
        )
        completed_training_seconds = float(resume_state.get("training_seconds", 0.0))
        resumed_from_step = optimizer_steps
        random.setstate(resume_state["python_random_state"])
        np.random.set_state(resume_state["numpy_random_state"])
        torch.set_rng_state(resume_state["torch_rng_state"])
        if device.startswith("cuda") and resume_state.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(resume_state["cuda_rng_state_all"])

    def save_resume_checkpoint(
        *, epoch_index: int, batch_index: int, session_started: float
    ) -> Path:
        if checkpoint_options is None:
            raise ModernBertCheckpointError("Checkpoint options are not configured")
        next_epoch_index, next_batch_index = next_training_cursor(
            epoch_index=epoch_index,
            batch_index=batch_index,
            batches_in_epoch=batches_in_epoch,
        )
        elapsed_seconds = completed_training_seconds + (
            time.perf_counter() - session_started
        )

        def write_payload(directory: Path) -> None:
            model.save_pretrained(directory, safe_serialization=True)
            tokenizer.save_pretrained(directory)
            torch.save(
                {
                    "training_fingerprint": current_fingerprint,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "epoch_index": next_epoch_index,
                    "next_batch_index": next_batch_index,
                    "optimizer_steps": optimizer_steps,
                    "train_loss_sum": train_loss_sum,
                    "train_loss_count": train_loss_count,
                    "train_weighted_loss_sum": train_weighted_loss_sum,
                    "train_weight_sum": train_weight_sum,
                    "observation_train_loss_sum": observation_train_loss_sum,
                    "observation_train_weighted_loss_sum": (
                        observation_train_weighted_loss_sum
                    ),
                    "observation_train_weight_sum": observation_train_weight_sum,
                    "observation_train_row_count": observation_train_row_count,
                    "observation_start_optimizer_step": (
                        observation_start_optimizer_step
                    ),
                    "training_seconds": elapsed_seconds,
                    "python_random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all()
                        if device.startswith("cuda")
                        else None
                    ),
                },
                directory / "training_state.pt",
            )

        checkpoint = create_checkpoint(
            checkpoint_options.checkpoint_dir,
            optimizer_steps=optimizer_steps,
            metadata={
                "training_fingerprint": current_fingerprint,
                "training_data_sha256": training_data_sha256,
                "optimizer_steps": optimizer_steps,
                "epoch_index": next_epoch_index,
                "next_batch_index": next_batch_index,
                "total_steps": total_steps,
            },
            write_payload=write_payload,
        )
        if checkpoint_options.on_checkpoint_saved is not None:
            checkpoint_options.on_checkpoint_saved(checkpoint)
        prune_checkpoints(
            checkpoint_options.checkpoint_dir,
            keep_last=checkpoint_options.keep_last_checkpoints,
        )
        return checkpoint

    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    latest_checkpoint: Path | None = None
    latest_validation_metrics: dict[str, Any] | None = None
    validation_targets: dict[int, list[tuple[int, int]]] = {}
    if validation_observations_per_epoch:
        for epoch_index in range(config.epochs):
            for point_number in range(1, validation_observations_per_epoch + 1):
                step_in_epoch = round(
                    point_number * steps_per_epoch / validation_observations_per_epoch
                )
                step_in_epoch = max(1, min(steps_per_epoch, step_in_epoch))
                global_step = epoch_index * steps_per_epoch + step_in_epoch
                if global_step <= total_steps:
                    validation_targets.setdefault(global_step, []).append(
                        (epoch_index + 1, point_number)
                    )
    for epoch_index in range(start_epoch_index, config.epochs):
        train_loader = train_loader_for_epoch(epoch_index)
        first_batch_index = start_batch_index if epoch_index == start_epoch_index else 0
        last_batch_index = -1
        for batch_index, batch in enumerate(train_loader):
            if batch_index < first_batch_index:
                continue
            last_batch_index = batch_index
            weights = batch.pop("sample_weight").to(device)
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            with precision_context():
                logits = model(**inputs).logits
                row_losses = functional.cross_entropy(logits, labels, reduction="none")
                window_start = (
                    batch_index // config.gradient_accumulation_steps
                ) * config.gradient_accumulation_steps
                window_end = min(
                    window_start + config.gradient_accumulation_steps,
                    len(train_loader),
                )
                first_row = window_start * config.train_batch_size
                last_row = min(
                    window_end * config.train_batch_size,
                    sampled_train_rows_per_epoch,
                )
                window_sample_count = last_row - first_row
                scaled_loss = (row_losses * weights).sum() / (
                    train_weight_mean * window_sample_count
                )
            detached_row_losses = row_losses.detach()
            batch_loss_sum = float(detached_row_losses.sum().cpu())
            batch_weighted_loss_sum = float((detached_row_losses * weights).sum().cpu())
            batch_weight_sum = float(weights.sum().cpu())
            batch_row_count = int(labels.numel())
            train_loss_sum += batch_loss_sum
            train_loss_count += batch_row_count
            train_weighted_loss_sum += batch_weighted_loss_sum
            train_weight_sum += batch_weight_sum
            observation_train_loss_sum += batch_loss_sum
            observation_train_weighted_loss_sum += batch_weighted_loss_sum
            observation_train_weight_sum += batch_weight_sum
            observation_train_row_count += batch_row_count
            scaler.scale(scaled_loss).backward()
            should_step = (
                batch_index + 1
            ) % config.gradient_accumulation_steps == 0 or batch_index + 1 == len(
                train_loader
            )
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if on_progress is not None:
                    on_progress(
                        TrainingProgressState(
                            optimizer_steps=optimizer_steps,
                            total_steps=total_steps,
                            epoch_number=epoch_index + 1,
                            total_epochs=config.epochs,
                        )
                    )
                for epoch_number, point_number in validation_targets.get(
                    optimizer_steps, []
                ):

                    def observe_validation() -> dict[str, Any]:
                        metrics = evaluate_validation()
                        if observation_train_row_count <= 0:
                            raise ModernBertTrainingError(
                                "Validation observation has no preceding train rows"
                            )
                        if on_validation_observation is not None:
                            on_validation_observation(
                                ValidationObservationState(
                                    epoch_number=epoch_number,
                                    point_number=point_number,
                                    points_per_epoch=validation_observations_per_epoch,
                                    epoch_fraction=(
                                        point_number / validation_observations_per_epoch
                                    ),
                                    optimizer_steps=optimizer_steps,
                                    training_fingerprint=current_fingerprint,
                                    metrics={
                                        **metrics,
                                        "train_loss": (
                                            observation_train_loss_sum
                                            / observation_train_row_count
                                        ),
                                        "train_weighted_loss": (
                                            observation_train_weighted_loss_sum
                                            / observation_train_weight_sum
                                        ),
                                        "train_loss_row_count": (
                                            observation_train_row_count
                                        ),
                                        "train_loss_optimizer_step_start_exclusive": (
                                            observation_start_optimizer_step
                                        ),
                                        "train_loss_optimizer_step_end_inclusive": (
                                            optimizer_steps
                                        ),
                                    },
                                )
                            )
                        return metrics

                    latest_validation_metrics = run_rng_isolated(observe_validation)
                    observation_train_loss_sum = 0.0
                    observation_train_weighted_loss_sum = 0.0
                    observation_train_weight_sum = 0.0
                    observation_train_row_count = 0
                    observation_start_optimizer_step = optimizer_steps
                if checkpoint_options is not None and (
                    optimizer_steps % checkpoint_options.save_every_optimizer_steps == 0
                    or optimizer_steps >= total_steps
                ):
                    latest_checkpoint = save_resume_checkpoint(
                        epoch_index=epoch_index,
                        batch_index=batch_index,
                        session_started=started,
                    )
            if optimizer_steps >= total_steps:
                break
        epoch_completed = last_batch_index + 1 == len(train_loader)
        if epoch_completed and on_epoch_end is not None:
            expected_checkpoint_name = f"step-{optimizer_steps:08d}"
            if (
                latest_checkpoint is None
                or latest_checkpoint.name != expected_checkpoint_name
            ):
                latest_checkpoint = save_resume_checkpoint(
                    epoch_index=epoch_index,
                    batch_index=last_batch_index,
                    session_started=started,
                )

            was_training = bool(model.training)
            model.eval()
            try:
                run_rng_isolated(
                    lambda: on_epoch_end(
                        EpochEndState(
                            epoch_number=epoch_index + 1,
                            optimizer_steps=optimizer_steps,
                            training_fingerprint=current_fingerprint,
                            checkpoint_path=latest_checkpoint,
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            max_length=config.max_length,
                        )
                    )
                )
            finally:
                model.train(was_training)
        if optimizer_steps >= total_steps:
            break
    training_seconds = completed_training_seconds + (time.perf_counter() - started)

    final_observation_is_current = bool(
        validation_targets.get(optimizer_steps)
        and validation_targets[optimizer_steps][-1]
        == (config.epochs, validation_observations_per_epoch)
    )
    validation_metrics = (
        latest_validation_metrics
        if final_observation_is_current and latest_validation_metrics is not None
        else run_rng_isolated(evaluate_validation)
    )

    token_counts = [
        len(
            tokenizer(
                str(text),
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )["input_ids"]
        )
        for text in all_rows["model_input_text"]
    ]
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
            "training_data_sha256": training_data_sha256,
            "input_row_count": len(all_rows),
            "train_row_count": len(train_rows),
            "sampled_train_row_count_per_epoch": sampled_train_rows_per_epoch,
            "validation_row_count": len(validation_rows),
            "label_distribution": all_rows["difficulty_label"].value_counts().to_dict(),
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
            "mean_train_loss": train_loss_sum / train_loss_count,
            "mean_weighted_train_loss": (train_weighted_loss_sum / train_weight_sum),
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "warmup_steps": warmup_steps,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": (
                config.train_batch_size * config.gradient_accumulation_steps
            ),
            "train_sampling_strategy": config.train_sampling_strategy,
            "train_sample_weight_mean": train_weight_mean,
            "training_seconds": training_seconds,
            "resumed_from_step": resumed_from_step,
            "training_fingerprint": current_fingerprint,
        },
        "validation": {
            **validation_metrics,
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
