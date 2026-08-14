from __future__ import annotations

import json
import os
import shutil
import uuid
from html import escape
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..shared.dataset_io import write_json
from .modernbert_checkpointing import (
    ARTIFACT_MANIFEST_NAME,
    ModernBertCheckpointError,
    atomic_copy_artifact,
    seal_artifact_directory,
    verify_artifact_directory,
)
from .modernbert_training import ValidationObservationState


VALIDATION_PROGRESS_SCHEMA_VERSION = "routerbench-validation-progress-v1"


class ModernBertValidationProgressError(RuntimeError):
    """Raised when a RouterBench validation observation is incomplete or unsafe."""


def _progress_svg(records: Sequence[Mapping[str, Any]]) -> str:
    if not records:
        raise ModernBertValidationProgressError(
            "At least one validation observation is required"
        )
    width, height = 920, 520
    left, top, chart_width, chart_height = 80, 70, 760, 320
    rows: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="460" y="30" text-anchor="middle" font-family="sans-serif" '
        'font-size="18" font-weight="bold">RouterBench validation progress</text>',
    ]
    for tick in range(6):
        ratio = tick / 5
        y = top + chart_height - ratio * chart_height
        rows.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" '
            'y2="{y:.1f}" stroke="#e0e0e0"/>'
        )
        rows.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{ratio:.1f}</text>'
        )
    denominator = max(1, len(records) - 1)
    x_positions = [
        left + chart_width * index / denominator for index in range(len(records))
    ]
    styles = {
        "accuracy": ("Accuracy", "#1565C0"),
        "macro_f1": ("Macro F1", "#2E7D32"),
    }
    for key, (label, color) in styles.items():
        points = []
        for x, record in zip(x_positions, records, strict=True):
            value = max(0.0, min(1.0, float(record[key])))
            y = top + chart_height - value * chart_height
            points.append(f"{x:.1f},{y:.1f}")
            rows.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
            )
        rows.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        legend_x = 210 if key == "accuracy" else 420
        rows.append(
            f'<line x1="{legend_x}" y1="455" x2="{legend_x + 30}" y2="455" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        rows.append(
            f'<text x="{legend_x + 38}" y="460" font-family="sans-serif" '
            f'font-size="13">{escape(label)}</text>'
        )
    for x, record in zip(x_positions, records, strict=True):
        rows.append(
            f'<text x="{x:.1f}" y="{top + chart_height + 22}" '
            'text-anchor="middle" font-family="sans-serif" font-size="10" '
            f'transform="rotate(35 {x:.1f} {top + chart_height + 22})">'
            f'E{int(record["epoch_number"])} '
            f'{int(record["point_number"])}/{int(record["points_per_epoch"])}'
            '</text>'
        )
    rows.append(
        '<text x="460" y="500" text-anchor="middle" font-family="sans-serif" '
        'font-size="12">Train/validation loss comparison is saved in '
        'loss_progress.svg.</text>'
    )
    rows.append('</svg>')
    return "\n".join(rows) + "\n"


def _loss_progress_svg(records: Sequence[Mapping[str, Any]]) -> str:
    comparable = [record for record in records if record.get("train_loss") is not None]
    if not comparable:
        raise ModernBertValidationProgressError(
            "At least one train/validation loss pair is required"
        )
    width, height = 920, 520
    left, top, chart_width, chart_height = 80, 70, 760, 320
    maximum_loss = max(
        max(float(record["train_loss"]), float(record["validation_loss"]))
        for record in comparable
    )
    y_max = max(0.1, maximum_loss * 1.1)
    rows: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="460" y="30" text-anchor="middle" font-family="sans-serif" '
        'font-size="18" font-weight="bold">Train vs validation loss</text>',
    ]
    for tick in range(6):
        ratio = tick / 5
        y = top + chart_height - ratio * chart_height
        rows.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" '
            'y2="{:.1f}" stroke="#e0e0e0"/>'.format(y)
        )
        rows.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{ratio * y_max:.3f}</text>'
        )
    denominator = max(1, len(comparable) - 1)
    x_positions = [
        left + chart_width * index / denominator
        for index in range(len(comparable))
    ]
    styles = {
        "train_loss": ("Train loss", "#EF6C00"),
        "validation_loss": ("Validation loss", "#6A1B9A"),
    }
    for style_index, (key, (label, color)) in enumerate(styles.items()):
        points = []
        for x, record in zip(x_positions, comparable, strict=True):
            value = float(record[key])
            y = top + chart_height - value / y_max * chart_height
            points.append(f"{x:.1f},{y:.1f}")
            rows.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
            )
        rows.append(
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        legend_x = 250 + style_index * 260
        rows.append(
            f'<line x1="{legend_x}" y1="455" x2="{legend_x + 30}" y2="455" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        rows.append(
            f'<text x="{legend_x + 38}" y="460" font-family="sans-serif" '
            f'font-size="13">{escape(label)}</text>'
        )
    for x, record in zip(x_positions, comparable, strict=True):
        rows.append(
            f'<text x="{x:.1f}" y="{top + chart_height + 22}" '
            'text-anchor="middle" font-family="sans-serif" font-size="10" '
            f'transform="rotate(35 {x:.1f} {top + chart_height + 22})">'
            f'E{int(record["epoch_number"])} '
            f'{int(record["point_number"])}/{int(record["points_per_epoch"])}'
            '</text>'
        )
    rows.append(
        '<text x="460" y="500" text-anchor="middle" font-family="sans-serif" '
        'font-size="12">Both lines use unweighted mean cross-entropy for a '
        'direct generalization comparison.</text>'
    )
    rows.append('</svg>')
    return "\n".join(rows) + "\n"


class RouterBenchValidationProgressTracker:
    """Persist the 1/3, 2/3 and 3/3 validation observations to Drive."""

    def __init__(
        self,
        *,
        local_output_dir: Path,
        persistent_output_dir: Path,
        points_per_epoch: int,
    ) -> None:
        if points_per_epoch <= 0:
            raise ModernBertValidationProgressError(
                "points_per_epoch must be positive"
            )
        self.local_output_dir = local_output_dir
        self.persistent_output_dir = persistent_output_dir
        self.points_per_epoch = points_per_epoch
        self.local_output_dir.mkdir(parents=True, exist_ok=True)
        self.persistent_output_dir.mkdir(parents=True, exist_ok=True)

    def _observation_name(self, epoch_number: int, point_number: int) -> str:
        index = (epoch_number - 1) * self.points_per_epoch + point_number
        return f"observation-{index:02d}"

    def _verify(
        self, directory: Path, *, training_fingerprint: str
    ) -> dict[str, Any]:
        manifest = verify_artifact_directory(
            directory,
            manifest_name=ARTIFACT_MANIFEST_NAME,
            expected_fingerprint=training_fingerprint,
        )
        metadata = manifest["metadata"]
        if metadata.get("schema_version") != VALIDATION_PROGRESS_SCHEMA_VERSION:
            raise ModernBertValidationProgressError(
                f"Unsupported validation progress schema: {directory}"
            )
        if int(metadata.get("points_per_epoch", -1)) != self.points_per_epoch:
            raise ModernBertValidationProgressError(
                "Existing validation artifacts use a different observation cadence"
            )
        return manifest

    def record(self, state: ValidationObservationState) -> dict[str, Any]:
        if state.points_per_epoch != self.points_per_epoch:
            raise ModernBertValidationProgressError(
                "Observation state and tracker cadence do not match"
            )
        name = self._observation_name(state.epoch_number, state.point_number)
        persistent_destination = self.persistent_output_dir / name
        if persistent_destination.exists():
            self._verify(
                persistent_destination,
                training_fingerprint=state.training_fingerprint,
            )
            with (persistent_destination / "metrics.json").open(
                "r", encoding="utf-8"
            ) as handle:
                metrics = json.load(handle)
            self.refresh_progress(training_fingerprint=state.training_fingerprint)
            return metrics

        contract = {
            "schema_version": VALIDATION_PROGRESS_SCHEMA_VERSION,
            "purpose": "training-validation-observation",
            "dataset": "routerbench-validation",
            "epoch_number": state.epoch_number,
            "point_number": state.point_number,
            "points_per_epoch": state.points_per_epoch,
            "epoch_fraction": state.epoch_fraction,
            "optimizer_steps": state.optimizer_steps,
            "training_fingerprint": state.training_fingerprint,
            "affects_gradients": False,
            "affects_optimizer": False,
            "affects_early_stopping": False,
            "affects_checkpoint_selection": False,
        }
        payload = {**state.metrics, "validation_observation": contract}
        destination = self.local_output_dir / name
        if destination.exists():
            self._verify(destination, training_fingerprint=state.training_fingerprint)
        else:
            temporary = self.local_output_dir / f".{name}.partial-{uuid.uuid4().hex}"
            try:
                temporary.mkdir(parents=True)
                write_json(temporary / "metrics.json", payload)
                write_json(temporary / "usage_contract.json", contract)
                seal_artifact_directory(
                    temporary,
                    manifest_name=ARTIFACT_MANIFEST_NAME,
                    metadata={
                        "schema_version": VALIDATION_PROGRESS_SCHEMA_VERSION,
                        "training_fingerprint": state.training_fingerprint,
                        "epoch_number": state.epoch_number,
                        "point_number": state.point_number,
                        "points_per_epoch": state.points_per_epoch,
                        "optimizer_steps": state.optimizer_steps,
                    },
                )
                self._verify(
                    temporary, training_fingerprint=state.training_fingerprint
                )
                os.replace(temporary, destination)
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        atomic_copy_artifact(
            destination,
            persistent_destination,
            manifest_name=ARTIFACT_MANIFEST_NAME,
            expected_fingerprint=state.training_fingerprint,
        )
        self.refresh_progress(training_fingerprint=state.training_fingerprint)
        return payload

    def refresh_progress(self, *, training_fingerprint: str) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        for directory in sorted(self.persistent_output_dir.glob("observation-*")):
            try:
                self._verify(directory, training_fingerprint=training_fingerprint)
                with (directory / "metrics.json").open(
                    "r", encoding="utf-8"
                ) as handle:
                    metrics = json.load(handle)
                observation = metrics["validation_observation"]
                records.append(
                    {
                        "epoch_number": int(observation["epoch_number"]),
                        "point_number": int(observation["point_number"]),
                        "points_per_epoch": int(observation["points_per_epoch"]),
                        "epoch_fraction": float(observation["epoch_fraction"]),
                        "optimizer_steps": int(observation["optimizer_steps"]),
                        "loss": float(metrics["loss"]),
                        "validation_loss": float(metrics["loss"]),
                        "validation_weighted_loss": float(
                            metrics.get("weighted_loss", metrics["loss"])
                        ),
                        "train_loss": (
                            None
                            if metrics.get("train_loss") is None
                            else float(metrics["train_loss"])
                        ),
                        "train_weighted_loss": (
                            None
                            if metrics.get("train_weighted_loss") is None
                            else float(metrics["train_weighted_loss"])
                        ),
                        "train_loss_row_count": int(
                            metrics.get("train_loss_row_count", 0)
                        ),
                        "train_loss_optimizer_step_start_exclusive": int(
                            metrics.get(
                                "train_loss_optimizer_step_start_exclusive", 0
                            )
                        ),
                        "train_loss_optimizer_step_end_inclusive": int(
                            metrics.get(
                                "train_loss_optimizer_step_end_inclusive",
                                observation["optimizer_steps"],
                            )
                        ),
                        "accuracy": float(metrics["accuracy"]),
                        "macro_f1": float(metrics["macro_f1"]),
                    }
                )
            except (
                KeyError,
                OSError,
                ValueError,
                json.JSONDecodeError,
                ModernBertCheckpointError,
            ):
                continue
        records.sort(key=lambda row: row["optimizer_steps"])
        if not records:
            raise ModernBertValidationProgressError(
                "No completed RouterBench validation observations are available"
            )
        payload = {
            "schema_version": VALIDATION_PROGRESS_SCHEMA_VERSION,
            "purpose": "training-validation-observation",
            "training_fingerprint": training_fingerprint,
            "points_per_epoch": self.points_per_epoch,
            "observations": records,
        }
        json_path = self.persistent_output_dir / "validation_progress.json"
        temporary_json = json_path.with_name(
            f".{json_path.name}.partial-{uuid.uuid4().hex}"
        )
        write_json(temporary_json, payload)
        os.replace(temporary_json, json_path)
        svg_path = self.persistent_output_dir / "validation_progress.svg"
        temporary_svg = svg_path.with_name(
            f".{svg_path.name}.partial-{uuid.uuid4().hex}"
        )
        temporary_svg.write_text(_progress_svg(records), encoding="utf-8")
        os.replace(temporary_svg, svg_path)
        if any(record.get("train_loss") is not None for record in records):
            loss_svg_path = self.persistent_output_dir / "loss_progress.svg"
            temporary_loss_svg = loss_svg_path.with_name(
                f".{loss_svg_path.name}.partial-{uuid.uuid4().hex}"
            )
            temporary_loss_svg.write_text(
                _loss_progress_svg(records), encoding="utf-8"
            )
            os.replace(temporary_loss_svg, loss_svg_path)
        return payload

    def require_all(
        self, *, expected_epochs: int, training_fingerprint: str
    ) -> dict[str, Any]:
        missing = self._missing(
            (
                (epoch_number, point_number)
                for epoch_number in range(1, expected_epochs + 1)
                for point_number in range(1, self.points_per_epoch + 1)
            ),
            training_fingerprint=training_fingerprint,
        )
        if missing:
            raise ModernBertValidationProgressError(
                f"Completed model is missing validation observations: {missing}"
            )
        return self.refresh_progress(training_fingerprint=training_fingerprint)

    def require_through_optimizer_step(
        self,
        *,
        optimizer_steps: int,
        steps_per_epoch: int,
        expected_epochs: int,
        training_fingerprint: str,
    ) -> None:
        expected = []
        for epoch_number in range(1, expected_epochs + 1):
            for point_number in range(1, self.points_per_epoch + 1):
                step_in_epoch = round(
                    point_number * steps_per_epoch / self.points_per_epoch
                )
                target_step = (epoch_number - 1) * steps_per_epoch + max(
                    1, min(steps_per_epoch, step_in_epoch)
                )
                if target_step <= optimizer_steps:
                    expected.append((epoch_number, point_number))
        missing = self._missing(
            expected, training_fingerprint=training_fingerprint
        )
        if missing:
            raise ModernBertValidationProgressError(
                "Resume checkpoint is newer than missing RouterBench validation "
                f"observations {missing}. Use a new run directory to obtain all points."
            )

    def _missing(
        self,
        expected: Iterable[tuple[int, int]],
        *,
        training_fingerprint: str,
    ) -> list[tuple[int, int]]:
        missing: list[tuple[int, int]] = []
        for epoch_number, point_number in expected:
            directory = self.persistent_output_dir / self._observation_name(
                epoch_number, point_number
            )
            try:
                self._verify(directory, training_fingerprint=training_fingerprint)
            except (ModernBertCheckpointError, ModernBertValidationProgressError):
                missing.append((epoch_number, point_number))
        return missing
