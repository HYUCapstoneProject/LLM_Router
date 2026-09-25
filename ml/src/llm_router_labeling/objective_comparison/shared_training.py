"""R0/R1/R2 comparison through the same checkpointable ModernBERT loop."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..shared.dataset_io import sha256_file, write_json
from ..training.modernbert_checkpointing import (
    ARTIFACT_MANIFEST_NAME,
    CheckpointOptions,
    latest_completed_checkpoint,
    seal_artifact_directory,
    training_fingerprint,
    verify_artifact_directory,
)
from ..training.modernbert_runtime import (
    load_runtime_contract,
    validate_runtime_contract,
)
from ..training.modernbert_training import (
    load_training_config,
    run_training,
    validate_checkpoint_contract,
)
from .evaluation import route_actions, summarize_source_curves
from .objectives import PRIMARY_METHODS, require_torch
from .training import (
    _batch_one_latency_ms,
    _predict_logits,
    _require_transformers,
    _score_frame,
)


def verified_training_projection(
    base_training_path: Path, outcome_dir: Path, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Join targets without changing the frozen R0 text, labels, order or weights."""
    manifest = json.loads((outcome_dir / "manifest.json").read_text())
    if (outcome_dir / "_SUCCESS").read_text().strip() != sha256_file(
        outcome_dir / "manifest.json"
    ):
        raise ValueError("Outcome completion marker does not seal its manifest")
    if (
        manifest["base_dataset_manifest_sha256"]
        != config["base_dataset_manifest_sha256"]
        or manifest["dataset_version"] != config["dataset_version"]
        or manifest["experiment_version"] != config["experiment_version"]
    ):
        raise ValueError("Outcome data belongs to another frozen dataset")
    if sha256_file(base_training_path) != config["base_training_sha256"]:
        raise ValueError("R0 base training checksum mismatch")
    pieces = []
    for name in (
        "train.parquet",
        "validation.parquet",
        "arms.json",
        "candidate-statistics.parquet",
    ):
        if sha256_file(outcome_dir / name) != manifest["artifacts"][name]["sha256"]:
            raise ValueError(f"Outcome checksum mismatch: {name}")
        if name.endswith(".parquet") and name != "candidate-statistics.parquet":
            pieces.append(pd.read_parquet(outcome_dir / name))
    outcomes = pd.concat(pieces, ignore_index=True)
    base = pd.read_parquet(base_training_path)
    if (
        set(base["sample_id"]) != set(outcomes["sample_id"])
        or outcomes["sample_id"].duplicated().any()
    ):
        raise ValueError("R0/R1/R2 must use exactly the same sample IDs")
    aligned = outcomes.set_index("sample_id").loc[base["sample_id"]].reset_index()
    for name in (
        "model_input_text",
        "difficulty_label",
        "sample_weight",
        "split",
        "duplicate_group_id",
        "source_dataset",
        "category",
        "dataset_version",
        "label_rule_version",
        "preprocessing_version",
        "split_version",
    ):
        if not base[name].reset_index(drop=True).equals(aligned[name]):
            raise ValueError(f"R0/R1/R2 input contract differs: {name}")
    if (
        set(base["split"]) != {"train", "validation"}
        or not base["sample_weight"].eq(1.0).all()
    ):
        raise ValueError(
            "Comparison requires train/D_dev only with effective weight one"
        )
    new_columns = [name for name in aligned.columns if name not in base.columns]
    combined = pd.concat([base.reset_index(drop=True), aligned[new_columns]], axis=1)
    validation = combined.loc[combined["split"].eq("validation")].reset_index(drop=True)
    return combined, validation, manifest


def _verify_imported_r0(
    model_dir: Path,
    *,
    training_config: Any,
    runtime_contract: Any,
    base_training_sha256: str,
    expected_train_rows: int,
    expected_validation_rows: int,
) -> dict[str, Any]:
    """Require the exact fixed-budget R0, including its sealed runtime fingerprint."""
    sealed = verify_artifact_directory(model_dir, manifest_name=ARTIFACT_MANIFEST_NAME)
    original_config = json.loads((model_dir / "training_config.json").read_text())
    metrics = json.loads((model_dir / "metrics.json").read_text())
    if original_config != asdict(training_config):
        raise ValueError("Imported R0 was trained with different settings")
    if training_config.max_train_steps is not None:
        raise ValueError("A truncated R0 budget cannot be imported")
    expected_steps = training_config.epochs * math.ceil(
        math.ceil(expected_train_rows / training_config.train_batch_size)
        / training_config.gradient_accumulation_steps
    )
    expected_fingerprint = training_fingerprint(
        training_data_sha256=base_training_sha256,
        config=asdict(training_config),
        maximum_steps=None,
        runtime_version=runtime_contract.runtime_version,
    )
    if (
        metrics["data"]["training_data_sha256"] != base_training_sha256
        or metrics["data"]["train_row_count"] != expected_train_rows
        or metrics["data"]["validation_row_count"] != expected_validation_rows
    ):
        raise ValueError("Imported R0 used different data")
    if (
        metrics["optimization"]["optimizer_steps"] != expected_steps
        or metrics["optimization"]["planned_steps"] != expected_steps
        or metrics["optimization"]["max_train_steps"] is not None
    ):
        raise ValueError("Imported R0 did not complete its full common budget")
    if (
        metrics["optimization"]["training_fingerprint"] != expected_fingerprint
        or sealed.get("metadata", {}).get("training_fingerprint")
        != expected_fingerprint
    ):
        raise ValueError("Imported R0 training fingerprint mismatch")
    validate_runtime_contract(
        runtime_contract, metrics["runtime"], device=training_config.device
    )
    validate_checkpoint_contract(model_dir)
    model_config = json.loads((model_dir / "config.json").read_text())
    if model_config.get("router_objective", "ordinal") != "ordinal":
        raise ValueError("Imported checkpoint is not R0")
    return metrics


def train_shared_objective(
    *,
    method: str,
    base_training_path: Path,
    outcome_dir: Path,
    experiment_config_path: Path,
    training_config_path: Path,
    runtime_config_path: Path,
    output_dir: Path,
    import_r0_dir: Path | None = None,
    maximum_train_steps: int = 0,
    on_checkpoint_saved: Any = None,
) -> dict[str, Any]:
    if method not in PRIMARY_METHODS:
        raise ValueError("The primary comparison includes R0/R1/R2 only")
    if import_r0_dir is not None and (method != "ordinal" or maximum_train_steps):
        raise ValueError("Only a completed full R0 can be imported")
    config = json.loads(experiment_config_path.read_text())
    if tuple(config.get("methods", ())) != PRIMARY_METHODS:
        raise ValueError("Shared comparison must declare the complete R0/R1/R2 set")
    training_config = load_training_config(training_config_path)
    runtime_contract = load_runtime_contract(runtime_config_path)
    if training_config.max_train_steps is not None:
        raise ValueError("Common full training must use the complete epoch budget")
    if training_config.seed != config["seed"]:
        raise ValueError("Training and comparison seeds differ")
    expected = config["training"]
    for name in (
        "epochs",
        "train_batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "gradient_clip_norm",
        "precision",
        "gradient_checkpointing",
        "attention_implementation",
        "train_sampling_strategy",
    ):
        if getattr(training_config, name) != expected[name]:
            raise ValueError(f"Common training setting differs: {name}")
    for name, value in config["backbone"].items():
        attribute = {
            "revision": "model_revision",
            "model_id": "model_id",
            "max_length": "max_length",
        }.get(name)
        if attribute and getattr(training_config, attribute) != value:
            raise ValueError(f"Common backbone setting differs: {name}")
    combined, validation, outcome_manifest = verified_training_projection(
        base_training_path, outcome_dir, config
    )
    maximum_train_rows = maximum_validation_rows = 0
    if maximum_train_steps:
        maximum_train_rows = int(config["smoke"]["maximum_train_rows"])
        maximum_validation_rows = int(config["smoke"]["maximum_validation_rows"])
        if maximum_train_steps > int(config["smoke"]["maximum_train_steps"]):
            raise ValueError("Pilot exceeds the frozen step budget")
        parts = []
        for split, limit in (
            ("train", maximum_train_rows),
            ("validation", maximum_validation_rows),
        ):
            rows = combined.loc[combined["split"].eq(split)]
            parts.append(
                rows.sample(n=min(len(rows), limit), random_state=training_config.seed)
            )
        combined = pd.concat(parts, ignore_index=True)
        validation = combined.loc[combined["split"].eq("validation")].reset_index(
            drop=True
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "_SUCCESS").exists():
        raise ValueError("A completed comparison run is immutable")
    training_path = output_dir / "training.parquet"
    combined.to_parquet(training_path, index=False)
    if import_r0_dir is not None:
        original_metrics = _verify_imported_r0(
            import_r0_dir,
            training_config=training_config,
            runtime_contract=runtime_contract,
            base_training_sha256=config["base_training_sha256"],
            expected_train_rows=int(combined["split"].eq("train").sum()),
            expected_validation_rows=len(validation),
        )
        model_dir = import_r0_dir
        train_metrics = original_metrics
    else:
        if method != "ordinal":
            training_config = replace(
                training_config,
                training_version=f"modernbert-objectives-continual-v1-{method}",
            )
        checkpoints = output_dir / "checkpoints"
        result = run_training(
            training_path,
            output_dir / "model",
            training_config,
            runtime_contract=runtime_contract,
            max_train_steps_override=maximum_train_steps or None,
            checkpoint_options=CheckpointOptions(
                checkpoint_dir=checkpoints,
                save_every_optimizer_steps=500,
                keep_last_checkpoints=2,
                resume_from=latest_completed_checkpoint(checkpoints),
                on_checkpoint_saved=on_checkpoint_saved,
            ),
            objective_method=method,
            pairwise_tie_weight=float(expected["pairwise_tie_weight"]),
        )
        model_dir, train_metrics = result.output_dir, result.metrics
        seal_artifact_directory(
            model_dir,
            manifest_name=ARTIFACT_MANIFEST_NAME,
            metadata={
                "training_fingerprint": train_metrics["optimization"][
                    "training_fingerprint"
                ],
                "method": method,
            },
        )
    if (
        not maximum_train_steps
        and train_metrics["optimization"]["optimizer_steps"]
        != train_metrics["optimization"]["planned_steps"]
    ):
        raise ValueError("Full comparison did not finish the common optimizer budget")

    torch = require_torch()
    AutoModel, AutoTokenizer, _ = _require_transformers()
    device = train_metrics["model"]["device"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(
        model_dir, attn_implementation=training_config.attention_implementation
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    inference_started = time.perf_counter()
    scores = _predict_logits(
        model,
        tokenizer,
        validation,
        batch_size=training_config.eval_batch_size,
        max_length=training_config.max_length,
        device=device,
    )
    inference_seconds = time.perf_counter() - inference_started
    latency_texts = validation["model_input_text"].tolist()
    if len(latency_texts) <= 5:
        # The latency helper discards five warmup calls. Keep tiny smoke inputs
        # measurable without changing the scored validation rows.
        latency_texts = [
            latency_texts[index % len(latency_texts)] for index in range(5)
        ] + latency_texts
    latency = _batch_one_latency_ms(
        model,
        tokenizer,
        latency_texts,
        max_length=training_config.max_length,
        device=device,
    )
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    evaluation = config["evaluation"]
    summary, penalties, curve = summarize_source_curves(
        validation,
        scores,
        method=method,
        penalties=evaluation["penalty_grid"],
        budgets=evaluation["relative_cost_budgets"],
        retention_targets=evaluation["quality_retention_targets"],
    )
    default_budget = f"{float(evaluation['default_relative_cost_budget']):.4f}"
    selected_costs = []
    for source, source_penalties in penalties.items():
        if default_budget not in source_penalties:
            continue
        mask = validation["source_dataset"].eq(source).to_numpy()
        rows = validation.loc[mask]
        actions = route_actions(
            method,
            scores[mask],
            rows[
                [f"decision_cost_{tier}" for tier in ("low", "medium", "high")]
            ].to_numpy(float),
            source_penalties[default_budget],
        )
        costs = rows[
            [f"actual_cost_{tier}" for tier in ("low", "medium", "high")]
        ].to_numpy(float)
        selected_costs.extend(costs[np.arange(len(rows)), actions].tolist())
    metrics = {
        "method": method,
        "seed": training_config.seed,
        "train_rows": int(combined["split"].eq("train").sum()),
        "validation_rows": len(validation),
        "optimizer_steps": train_metrics["optimization"]["optimizer_steps"],
        "fit_seconds": train_metrics["optimization"]["training_seconds"],
        "total_parameter_count": parameter_count,
        "validation_inference_seconds": inference_seconds,
        "batch_1_latency_ms": {
            "sample_count": len(latency),
            "p50": float(np.quantile(latency, 0.5)),
            "p95": float(np.quantile(latency, 0.95)),
        },
        "validation_mean_actual_cost": (
            float(np.mean(selected_costs)) if selected_costs else None
        ),
        "summary": summary,
        "validation_budget_penalties": penalties,
        "training_metrics": train_metrics,
    }
    _score_frame(validation, scores).to_parquet(
        output_dir / "validation-scores.parquet", index=False
    )
    curve.to_parquet(output_dir / "validation-curve.parquet", index=False)
    write_json(output_dir / "metrics.json", metrics)
    model_files = sorted(path for path in model_dir.rglob("*") if path.is_file())
    manifest = {
        "schema_version": "routing-objective-run-v1",
        "experiment_version": config["experiment_version"],
        "method": method,
        "seed": training_config.seed,
        "outcome_dataset_version": outcome_manifest["dataset_version"],
        "outcome_manifest_sha256": sha256_file(outcome_dir / "manifest.json"),
        "experiment_config_sha256": sha256_file(experiment_config_path),
        "base_training_sha256": config["base_training_sha256"],
        "training_data_sha256": sha256_file(training_path),
        "validation_data_sha256": outcome_manifest["artifacts"]["validation.parquet"][
            "sha256"
        ],
        "common_training_config": asdict(load_training_config(training_config_path)),
        "maximum_train_rows": maximum_train_rows,
        "maximum_validation_rows": maximum_validation_rows,
        "maximum_train_steps": maximum_train_steps,
        "imported_r0_checkpoint": str(import_r0_dir) if import_r0_dir else None,
        "model_dir": str(model_dir),
        "test_data_accessed": False,
        "routerarena_accessed": False,
        "runtime_version": runtime_contract.runtime_version,
        "runtime_contract_sha256": sha256_file(runtime_config_path),
        "model_artifact_size_bytes": sum(path.stat().st_size for path in model_files),
        "model_artifacts": {
            str(path.relative_to(model_dir)): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in model_files
        },
        "artifacts": {
            name: {
                "size_bytes": (output_dir / name).stat().st_size,
                "sha256": sha256_file(output_dir / name),
            }
            for name in (
                "metrics.json",
                "validation-scores.parquet",
                "validation-curve.parquet",
            )
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "_SUCCESS").write_text(
        sha256_file(output_dir / "manifest.json") + "\n"
    )
    return {"manifest": manifest, "metrics": metrics}
