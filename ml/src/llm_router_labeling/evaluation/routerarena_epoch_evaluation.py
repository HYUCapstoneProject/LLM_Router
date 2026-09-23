from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ..shared.dataset_io import load_config, sha256_file, write_json
from ..training.modernbert_checkpointing import (
    ARTIFACT_MANIFEST_NAME,
    CHECKPOINT_MANIFEST_NAME,
    ModernBertCheckpointError,
    atomic_copy_artifact,
    seal_artifact_directory,
    verify_artifact_directory,
)
from .routerarena_routing_evaluator import (
    HuggingFaceRouterPredictor,
    LoadedHuggingFaceRouterPredictor,
    RouterPredictor,
    run_routerarena_routing_evaluation,
)
from .routerarena_visualization import (
    write_routerarena_distribution_charts,
    write_routerarena_epoch_progress_chart,
)


EPOCH_EVALUATION_SCHEMA_VERSION = "routerarena-epoch-observation-v1"


class RouterArenaEpochEvaluationError(RuntimeError):
    """Raised when an inference-only epoch observation is unsafe or incomplete."""


class RouterArenaEpochEvaluator:
    """Persist RouterArena observations without feeding them back into training."""

    def __init__(
        self,
        *,
        training_path: Path,
        evaluation_path: Path,
        metadata_path: Path | None,
        config_path: Path,
        local_output_dir: Path,
        persistent_output_dir: Path,
        batch_size: int | None = None,
        max_length: int | None = None,
        device: str = "cuda",
    ) -> None:
        if training_path.resolve() == evaluation_path.resolve():
            raise RouterArenaEpochEvaluationError(
                "RouterArena evaluation input must not be the training Parquet"
            )
        if not evaluation_path.is_file():
            raise RouterArenaEpochEvaluationError(
                f"RouterArena evaluation Parquet does not exist: {evaluation_path}"
            )
        if metadata_path is not None and not metadata_path.is_file():
            raise RouterArenaEpochEvaluationError(
                f"RouterArena metadata Parquet does not exist: {metadata_path}"
            )
        if not config_path.is_file():
            raise RouterArenaEpochEvaluationError(
                f"RouterArena evaluation config does not exist: {config_path}"
            )
        self.evaluation = pd.read_parquet(evaluation_path)
        self.metadata = (
            pd.read_parquet(metadata_path) if metadata_path is not None else None
        )
        self.config = load_config(config_path)
        inference = self.config["inference"]
        self.batch_size = (
            int(inference["batch_size"]) if batch_size is None else batch_size
        )
        self.max_length = (
            int(inference["max_length"]) if max_length is None else max_length
        )
        if self.batch_size <= 0 or self.max_length <= 0:
            raise RouterArenaEpochEvaluationError(
                "RouterArena batch_size and max_length must be positive"
            )
        self.device = device
        self.local_output_dir = local_output_dir
        self.persistent_output_dir = persistent_output_dir
        self.local_output_dir.mkdir(parents=True, exist_ok=True)
        self.persistent_output_dir.mkdir(parents=True, exist_ok=True)
        fingerprint_payload = {
            "schema_version": EPOCH_EVALUATION_SCHEMA_VERSION,
            "evaluation_sha256": sha256_file(evaluation_path),
            "metadata_sha256": (
                sha256_file(metadata_path) if metadata_path is not None else None
            ),
            "config_sha256": sha256_file(config_path),
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }
        self.evaluation_fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _epoch_name(epoch_number: int) -> str:
        if epoch_number <= 0:
            raise RouterArenaEpochEvaluationError("epoch_number must be positive")
        return f"epoch-{epoch_number:02d}"

    def _verify_epoch(
        self, directory: Path, *, training_fingerprint: str
    ) -> dict[str, Any]:
        manifest = verify_artifact_directory(
            directory,
            manifest_name=ARTIFACT_MANIFEST_NAME,
            expected_fingerprint=training_fingerprint,
        )
        metadata = manifest["metadata"]
        if metadata.get("schema_version") != EPOCH_EVALUATION_SCHEMA_VERSION:
            raise RouterArenaEpochEvaluationError(
                f"Unsupported epoch evaluation schema: {directory}"
            )
        if metadata.get("evaluation_fingerprint") != self.evaluation_fingerprint:
            raise RouterArenaEpochEvaluationError(
                "Existing RouterArena epoch artifact uses different evaluation data/config"
            )
        return manifest

    def _completed_metrics(
        self, *, epoch_number: int, training_fingerprint: str
    ) -> dict[str, Any] | None:
        destination = self.persistent_output_dir / self._epoch_name(epoch_number)
        if not destination.exists():
            return None
        self._verify_epoch(destination, training_fingerprint=training_fingerprint)
        with (destination / "metrics.json").open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        if not isinstance(metrics, dict):
            raise RouterArenaEpochEvaluationError(
                f"Epoch metrics must be a JSON object: {destination}"
            )
        return metrics

    def _evaluate_with_predictor(
        self,
        predictor: RouterPredictor,
        *,
        epoch_number: int,
        optimizer_steps: int,
        training_fingerprint: str,
        checkpoint_path: Path,
    ) -> dict[str, Any]:
        existing = self._completed_metrics(
            epoch_number=epoch_number,
            training_fingerprint=training_fingerprint,
        )
        if existing is not None:
            self.refresh_progress(training_fingerprint=training_fingerprint)
            return existing

        result = run_routerarena_routing_evaluation(
            self.evaluation,
            predictor,
            label_order=self.config["label_order"],
            route_mapping=self.config["route_mapping"],
            evaluation_version=self.config["evaluation_version"],
            label_semantics=self.config["label_semantics"],
            metadata=self.metadata,
        )
        result.metrics["epoch_observation"] = {
            "schema_version": EPOCH_EVALUATION_SCHEMA_VERSION,
            "epoch_number": epoch_number,
            "optimizer_steps": optimizer_steps,
            "training_fingerprint": training_fingerprint,
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "checkpoint_path": str(checkpoint_path),
            "purpose": "inference-only-observation",
            "affects_loss": False,
            "affects_gradients": False,
            "affects_optimizer": False,
            "affects_early_stopping": False,
            "affects_checkpoint_selection": False,
        }

        epoch_name = self._epoch_name(epoch_number)
        destination = self.local_output_dir / epoch_name
        if destination.exists():
            self._verify_epoch(
                destination, training_fingerprint=training_fingerprint
            )
        else:
            temporary = self.local_output_dir / (
                f".{epoch_name}.partial-{uuid.uuid4().hex}"
            )
            try:
                temporary.mkdir(parents=True)
                result.predictions.to_parquet(
                    temporary / "predictions.parquet", index=False
                )
                write_json(temporary / "metrics.json", result.metrics)
                write_json(temporary / "model_info.json", dict(predictor.info()))
                write_json(
                    temporary / "usage_contract.json",
                    result.metrics["epoch_observation"],
                )
                write_routerarena_distribution_charts(result.metrics, temporary)
                seal_artifact_directory(
                    temporary,
                    manifest_name=ARTIFACT_MANIFEST_NAME,
                    metadata={
                        "schema_version": EPOCH_EVALUATION_SCHEMA_VERSION,
                        "training_fingerprint": training_fingerprint,
                        "evaluation_fingerprint": self.evaluation_fingerprint,
                        "epoch_number": epoch_number,
                        "optimizer_steps": optimizer_steps,
                    },
                )
                self._verify_epoch(
                    temporary, training_fingerprint=training_fingerprint
                )
                os.replace(temporary, destination)
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise

        atomic_copy_artifact(
            destination,
            self.persistent_output_dir / epoch_name,
            manifest_name=ARTIFACT_MANIFEST_NAME,
            expected_fingerprint=training_fingerprint,
        )
        self.refresh_progress(training_fingerprint=training_fingerprint)
        return result.metrics

    def evaluate_loaded_model(
        self,
        *,
        model: Any,
        tokenizer: Any,
        torch_module: Any,
        epoch_number: int,
        optimizer_steps: int,
        training_fingerprint: str,
        checkpoint_path: Path,
        device: str,
    ) -> dict[str, Any]:
        predictor = LoadedHuggingFaceRouterPredictor(
            model,
            tokenizer,
            torch_module,
            checkpoint_path=str(checkpoint_path),
            label_order=self.config["label_order"],
            batch_size=self.batch_size,
            max_length=self.max_length,
            device=device,
        )
        return self._evaluate_with_predictor(
            predictor,
            epoch_number=epoch_number,
            optimizer_steps=optimizer_steps,
            training_fingerprint=training_fingerprint,
            checkpoint_path=checkpoint_path,
        )

    def evaluate_predictor(
        self,
        predictor: RouterPredictor,
        *,
        epoch_number: int,
        optimizer_steps: int,
        training_fingerprint: str,
        checkpoint_path: Path,
    ) -> dict[str, Any]:
        """Evaluate a predictor under the same inference-only artifact contract."""
        return self._evaluate_with_predictor(
            predictor,
            epoch_number=epoch_number,
            optimizer_steps=optimizer_steps,
            training_fingerprint=training_fingerprint,
            checkpoint_path=checkpoint_path,
        )

    def recover_epoch_from_checkpoint(self, checkpoint_path: Path) -> bool:
        manifest = verify_artifact_directory(
            checkpoint_path, manifest_name=CHECKPOINT_MANIFEST_NAME
        )
        metadata = manifest["metadata"]
        next_batch_index = int(metadata.get("next_batch_index", -1))
        epoch_number = int(metadata.get("epoch_index", 0))
        if next_batch_index != 0 or epoch_number <= 0:
            return False
        training_fingerprint = str(metadata["training_fingerprint"])
        if self._completed_metrics(
            epoch_number=epoch_number,
            training_fingerprint=training_fingerprint,
        ) is not None:
            self.refresh_progress(training_fingerprint=training_fingerprint)
            return False
        predictor = HuggingFaceRouterPredictor(
            str(checkpoint_path),
            label_order=self.config["label_order"],
            batch_size=self.batch_size,
            max_length=self.max_length,
            device=self.device,
        )
        try:
            self._evaluate_with_predictor(
                predictor,
                epoch_number=epoch_number,
                optimizer_steps=int(metadata["optimizer_steps"]),
                training_fingerprint=training_fingerprint,
                checkpoint_path=checkpoint_path,
            )
        finally:
            del predictor
            gc.collect()
            try:
                import torch

                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        return True

    def refresh_progress(self, *, training_fingerprint: str) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        for directory in sorted(self.persistent_output_dir.glob("epoch-*")):
            try:
                self._verify_epoch(
                    directory, training_fingerprint=training_fingerprint
                )
                with (directory / "metrics.json").open(
                    "r", encoding="utf-8"
                ) as handle:
                    metrics = json.load(handle)
                observation = metrics["epoch_observation"]
                records.append(
                    {
                        "epoch_number": int(observation["epoch_number"]),
                        "optimizer_steps": int(observation["optimizer_steps"]),
                        "accuracy": float(metrics["accuracy"]),
                        "macro_f1": float(metrics["macro_f1"]),
                        "under_routing_rate": float(metrics["under_routing_rate"]),
                        "over_routing_rate": float(metrics["over_routing_rate"]),
                        "mean_absolute_tier_error": float(
                            metrics["mean_absolute_tier_error"]
                        ),
                    }
                )
            except (KeyError, OSError, json.JSONDecodeError, ModernBertCheckpointError):
                continue
        records.sort(key=lambda row: row["epoch_number"])
        if not records:
            raise RouterArenaEpochEvaluationError(
                "No completed RouterArena epoch evaluations are available"
            )
        baseline = records[0]
        for record in records:
            record["delta_from_first_epoch"] = {
                "accuracy": record["accuracy"] - baseline["accuracy"],
                "macro_f1": record["macro_f1"] - baseline["macro_f1"],
                "under_routing_rate": (
                    record["under_routing_rate"] - baseline["under_routing_rate"]
                ),
                "mean_absolute_tier_error": (
                    record["mean_absolute_tier_error"]
                    - baseline["mean_absolute_tier_error"]
                ),
            }
        payload = {
            "schema_version": EPOCH_EVALUATION_SCHEMA_VERSION,
            "purpose": "inference-only-observation",
            "training_fingerprint": training_fingerprint,
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "epochs": records,
        }
        progress_json = self.persistent_output_dir / "epoch_progress.json"
        temporary_json = progress_json.with_name(
            f".{progress_json.name}.partial-{uuid.uuid4().hex}"
        )
        write_json(temporary_json, payload)
        os.replace(temporary_json, progress_json)
        progress_svg = self.persistent_output_dir / "epoch_progress.svg"
        temporary_svg = progress_svg.with_name(
            f".{progress_svg.name}.partial-{uuid.uuid4().hex}"
        )
        write_routerarena_epoch_progress_chart(records, temporary_svg)
        os.replace(temporary_svg, progress_svg)
        return payload

    def require_all_epochs(
        self, *, expected_epochs: int, training_fingerprint: str
    ) -> dict[str, Any]:
        if expected_epochs < 0:
            raise RouterArenaEpochEvaluationError(
                "expected_epochs must be non-negative"
            )
        if expected_epochs == 0:
            return {
                "schema_version": EPOCH_EVALUATION_SCHEMA_VERSION,
                "purpose": "inference-only-observation",
                "training_fingerprint": training_fingerprint,
                "evaluation_fingerprint": self.evaluation_fingerprint,
                "epochs": [],
            }
        missing = [
            epoch_number
            for epoch_number in range(1, expected_epochs + 1)
            if self._completed_metrics(
                epoch_number=epoch_number,
                training_fingerprint=training_fingerprint,
            )
            is None
        ]
        if missing:
            raise RouterArenaEpochEvaluationError(
                "Completed model is missing RouterArena observations for epochs "
                f"{missing}; use a new run directory to obtain all intermediate results"
            )
        return self.refresh_progress(training_fingerprint=training_fingerprint)
