from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from ..evaluation.routerarena_epoch_evaluation import RouterArenaEpochEvaluator
from .modernbert_checkpointing import (
    ARTIFACT_MANIFEST_NAME,
    CHECKPOINT_MANIFEST_NAME,
    CheckpointOptions,
    ModernBertCheckpointError,
    atomic_copy_artifact,
    atomic_copy_checkpoint,
    latest_completed_checkpoint,
    prune_checkpoints,
    seal_artifact_directory,
    verify_artifact_directory,
)
from .modernbert_runtime import RuntimeContract
from .modernbert_training import (
    EpochEndState,
    TrainingConfig,
    TrainingResult,
    ValidationObservationState,
    run_training,
    training_sample_count_per_epoch,
)
from .modernbert_validation_progress import RouterBenchValidationProgressTracker


ResumeMode = Literal["auto", "required", "never"]


@dataclass(frozen=True)
class ColabTrainingResult:
    status: str
    persistent_model_dir: Path
    metrics: dict[str, object]
    resumed_from: Path | None


def _load_metrics(model_dir: Path) -> dict[str, object]:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.is_file():
        raise ModernBertCheckpointError(
            f"Completed model has no metrics.json: {model_dir}"
        )
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if not isinstance(metrics, dict):
        raise ModernBertCheckpointError(
            f"Completed model metrics must be an object: {metrics_path}"
        )
    return metrics


def _completed_final_model(model_dir: Path) -> bool:
    try:
        verify_artifact_directory(
            model_dir, manifest_name=ARTIFACT_MANIFEST_NAME
        )
    except ModernBertCheckpointError:
        return False
    return True


def _completed_epoch_count(
    config: TrainingConfig, metrics: dict[str, object]
) -> int:
    optimization = metrics.get("optimization")
    if not isinstance(optimization, dict):
        raise ModernBertCheckpointError("Training metrics have no optimization object")
    planned_steps = int(optimization["planned_steps"])
    optimizer_steps = int(optimization["optimizer_steps"])
    steps_per_epoch = planned_steps // config.epochs
    if steps_per_epoch <= 0 or steps_per_epoch * config.epochs != planned_steps:
        raise ModernBertCheckpointError(
            "Training metrics cannot determine completed epoch boundaries"
        )
    return min(config.epochs, optimizer_steps // steps_per_epoch)


def should_sync_checkpoint_to_drive(
    metadata: dict[str, object], *, interval: int
) -> bool:
    if interval <= 0:
        raise ModernBertCheckpointError("Drive sync interval must be positive")
    optimizer_steps = int(metadata["optimizer_steps"])
    return (
        optimizer_steps % interval == 0
        or int(metadata.get("next_batch_index", -1)) == 0
        or optimizer_steps >= int(metadata["total_steps"])
    )


def run_colab_training(
    training_path: Path,
    local_run_dir: Path,
    persistent_run_dir: Path,
    config: TrainingConfig,
    *,
    runtime_contract: RuntimeContract,
    save_every_optimizer_steps: int = 50,
    drive_sync_every_optimizer_steps: int = 500,
    keep_last_checkpoints: int = 2,
    resume_mode: ResumeMode = "auto",
    device_override: str = "cuda",
    max_train_steps_override: int | None = None,
    epoch_evaluator: RouterArenaEpochEvaluator | None = None,
    routerbench_validations_per_epoch: int = 3,
) -> ColabTrainingResult:
    if resume_mode not in {"auto", "required", "never"}:
        raise ModernBertCheckpointError(f"Unsupported resume mode: {resume_mode}")
    if drive_sync_every_optimizer_steps <= 0:
        raise ModernBertCheckpointError(
            "drive_sync_every_optimizer_steps must be positive"
        )
    if routerbench_validations_per_epoch < 0:
        raise ModernBertCheckpointError(
            "routerbench_validations_per_epoch must be non-negative"
        )
    if not persistent_run_dir.parent.is_dir():
        raise ModernBertCheckpointError(
            "Persistent run parent does not exist. Mount Google Drive before training: "
            f"{persistent_run_dir.parent}"
        )
    local_run_dir.mkdir(parents=True, exist_ok=True)
    persistent_run_dir.mkdir(parents=True, exist_ok=True)
    validation_tracker = (
        RouterBenchValidationProgressTracker(
            local_output_dir=local_run_dir / "routerbench-validation-progress",
            persistent_output_dir=(
                persistent_run_dir / "routerbench-validation-progress"
            ),
            points_per_epoch=routerbench_validations_per_epoch,
        )
        if routerbench_validations_per_epoch
        else None
    )

    def require_completed_observations(
        metrics: dict[str, object], *, training_fingerprint: str
    ) -> None:
        if validation_tracker is None:
            return
        optimization = metrics.get("optimization")
        if not isinstance(optimization, dict):
            raise ModernBertCheckpointError(
                "Training metrics have no optimization object"
            )
        if int(optimization["optimizer_steps"]) == int(
            optimization["planned_steps"]
        ):
            validation_tracker.require_all(
                expected_epochs=config.epochs,
                training_fingerprint=training_fingerprint,
            )

    local_final = local_run_dir / "final-model"
    persistent_final = persistent_run_dir / "final-model"
    if _completed_final_model(persistent_final):
        final_metrics = _load_metrics(persistent_final)
        manifest = verify_artifact_directory(
            persistent_final, manifest_name=ARTIFACT_MANIFEST_NAME
        )
        if epoch_evaluator is not None:
            epoch_evaluator.require_all_epochs(
                expected_epochs=_completed_epoch_count(config, final_metrics),
                training_fingerprint=str(
                    manifest["metadata"]["training_fingerprint"]
                ),
            )
        require_completed_observations(
            final_metrics,
            training_fingerprint=str(manifest["metadata"]["training_fingerprint"]),
        )
        return ColabTrainingResult(
            status="already_completed",
            persistent_model_dir=persistent_final,
            metrics=final_metrics,
            resumed_from=None,
        )
    if _completed_final_model(local_final):
        final_metrics = _load_metrics(local_final)
        manifest = verify_artifact_directory(
            local_final, manifest_name=ARTIFACT_MANIFEST_NAME
        )
        fingerprint = str(manifest["metadata"]["training_fingerprint"])
        if epoch_evaluator is not None:
            epoch_evaluator.require_all_epochs(
                expected_epochs=_completed_epoch_count(config, final_metrics),
                training_fingerprint=fingerprint,
            )
        require_completed_observations(
            final_metrics, training_fingerprint=fingerprint
        )
        atomic_copy_artifact(
            local_final,
            persistent_final,
            manifest_name=ARTIFACT_MANIFEST_NAME,
            expected_fingerprint=fingerprint,
        )
        return ColabTrainingResult(
            status="completed_from_local_artifact",
            persistent_model_dir=persistent_final,
            metrics=final_metrics,
            resumed_from=None,
        )
    if local_final.exists():
        shutil.rmtree(local_final)

    persistent_checkpoints = persistent_run_dir / "checkpoints"
    local_checkpoints = local_run_dir / "checkpoints"
    persistent_latest = latest_completed_checkpoint(persistent_checkpoints)
    local_latest = latest_completed_checkpoint(local_checkpoints)
    available_latest = max(
        (path for path in (persistent_latest, local_latest) if path is not None),
        key=lambda path: path.name,
        default=None,
    )
    if resume_mode == "required" and available_latest is None:
        raise ModernBertCheckpointError(
            f"resume_mode=required but no completed checkpoint exists in "
            f"{persistent_checkpoints}"
        )
    if resume_mode == "never" and available_latest is not None:
        raise ModernBertCheckpointError(
            "resume_mode=never refuses to overwrite an existing persistent run"
        )

    local_resume: Path | None = None
    if available_latest is not None and resume_mode != "never":
        if available_latest.parent == persistent_checkpoints:
            local_resume = local_checkpoints / available_latest.name
            atomic_copy_checkpoint(available_latest, local_resume)
        else:
            local_resume = available_latest

    if validation_tracker is not None and local_resume is not None:
        resume_manifest = verify_artifact_directory(
            local_resume, manifest_name=CHECKPOINT_MANIFEST_NAME
        )
        resume_metadata = resume_manifest["metadata"]
        split_rows = pd.read_parquet(
            training_path, columns=["split", "difficulty_label"]
        )
        train_rows = split_rows.loc[split_rows["split"].eq(config.train_split)]
        sampled_train_row_count = training_sample_count_per_epoch(
            train_rows, sampling_strategy=config.train_sampling_strategy
        )
        batches_per_epoch = math.ceil(
            sampled_train_row_count / config.train_batch_size
        )
        steps_per_epoch = math.ceil(
            batches_per_epoch / config.gradient_accumulation_steps
        )
        validation_tracker.require_through_optimizer_step(
            optimizer_steps=int(resume_metadata["optimizer_steps"]),
            steps_per_epoch=steps_per_epoch,
            expected_epochs=config.epochs,
            training_fingerprint=str(resume_metadata["training_fingerprint"]),
        )

    def persist_checkpoint(local_checkpoint: Path) -> None:
        manifest = verify_artifact_directory(
            local_checkpoint,
            manifest_name=CHECKPOINT_MANIFEST_NAME,
        )
        metadata = manifest["metadata"]
        if not should_sync_checkpoint_to_drive(
            metadata, interval=drive_sync_every_optimizer_steps
        ):
            return
        fingerprint = str(metadata["training_fingerprint"])
        atomic_copy_checkpoint(
            local_checkpoint,
            persistent_checkpoints / local_checkpoint.name,
            expected_fingerprint=fingerprint,
        )
        prune_checkpoints(
            persistent_checkpoints, keep_last=keep_last_checkpoints
        )

    if (
        local_resume is not None
        and (
            persistent_latest is None
            or local_resume.name > persistent_latest.name
        )
    ):
        persist_checkpoint(local_resume)

    if epoch_evaluator is not None and local_resume is not None:
        recovered = epoch_evaluator.recover_epoch_from_checkpoint(local_resume)
        if recovered:
            print(
                json.dumps(
                    {
                        "event": "routerarena_epoch_evaluation_recovered",
                        "checkpoint": str(local_resume),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

    def observe_epoch(state: EpochEndState) -> None:
        if epoch_evaluator is None:
            return
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Epoch evaluation requires PyTorch") from error
        metrics = epoch_evaluator.evaluate_loaded_model(
            model=state.model,
            tokenizer=state.tokenizer,
            torch_module=torch,
            epoch_number=state.epoch_number,
            optimizer_steps=state.optimizer_steps,
            training_fingerprint=state.training_fingerprint,
            checkpoint_path=state.checkpoint_path,
            device=state.device,
        )
        print(
            json.dumps(
                {
                    "event": "routerarena_epoch_evaluation_completed",
                    "epoch_number": state.epoch_number,
                    "optimizer_steps": state.optimizer_steps,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "under_routing_rate": metrics["under_routing_rate"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    def observe_routerbench_validation(
        state: ValidationObservationState,
    ) -> None:
        if validation_tracker is None:
            return
        metrics = validation_tracker.record(state)
        print(
            json.dumps(
                {
                    "event": "routerbench_validation_observation_completed",
                    "epoch_number": state.epoch_number,
                    "point_number": state.point_number,
                    "points_per_epoch": state.points_per_epoch,
                    "optimizer_steps": state.optimizer_steps,
                    "train_loss": metrics["train_loss"],
                    "train_weighted_loss": metrics["train_weighted_loss"],
                    "validation_loss": metrics["loss"],
                    "validation_weighted_loss": metrics["weighted_loss"],
                    "train_loss_row_count": metrics["train_loss_row_count"],
                    "train_loss_optimizer_step_start_exclusive": metrics[
                        "train_loss_optimizer_step_start_exclusive"
                    ],
                    "train_loss_optimizer_step_end_inclusive": metrics[
                        "train_loss_optimizer_step_end_inclusive"
                    ],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    result: TrainingResult = run_training(
        training_path,
        local_final,
        config,
        runtime_contract=runtime_contract,
        device_override=device_override,
        max_train_steps_override=max_train_steps_override,
        checkpoint_options=CheckpointOptions(
            checkpoint_dir=local_checkpoints,
            save_every_optimizer_steps=save_every_optimizer_steps,
            keep_last_checkpoints=keep_last_checkpoints,
            resume_from=local_resume,
            on_checkpoint_saved=persist_checkpoint,
        ),
        on_epoch_end=observe_epoch if epoch_evaluator is not None else None,
        validation_observations_per_epoch=routerbench_validations_per_epoch,
        on_validation_observation=(
            observe_routerbench_validation if validation_tracker is not None else None
        ),
    )
    fingerprint = str(result.metrics["optimization"]["training_fingerprint"])
    seal_artifact_directory(
        local_final,
        manifest_name=ARTIFACT_MANIFEST_NAME,
        metadata={
            "artifact_type": "modernbert-final-model",
            "training_fingerprint": fingerprint,
            "optimizer_steps": result.metrics["optimization"]["optimizer_steps"],
        },
    )
    if epoch_evaluator is not None:
        epoch_evaluator.require_all_epochs(
            expected_epochs=_completed_epoch_count(config, result.metrics),
            training_fingerprint=fingerprint,
        )
    require_completed_observations(
        result.metrics, training_fingerprint=fingerprint
    )
    atomic_copy_artifact(
        local_final,
        persistent_final,
        manifest_name=ARTIFACT_MANIFEST_NAME,
        expected_fingerprint=fingerprint,
    )
    return ColabTrainingResult(
        status="completed",
        persistent_model_dir=persistent_final,
        metrics=result.metrics,
        resumed_from=local_resume,
    )
