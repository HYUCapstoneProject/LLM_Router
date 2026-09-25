"""Four fixed-budget research arms; no held-out inference or service activation."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..objective_comparison.objectives import validate_regret_lambdas
from ..objective_comparison.training import _predict_logits, _require_transformers
from ..shared.dataset_io import sha256_file, write_json
from ..training.modernbert_checkpointing import (
    ARTIFACT_MANIFEST_NAME,
    CheckpointOptions,
    latest_completed_checkpoint,
    seal_artifact_directory,
    verify_artifact_directory,
)
from ..training.modernbert_runtime import load_runtime_contract
from ..training.modernbert_training import TrainingConfig, run_training

ARMS = {
    "r1-natural": ("pointwise", "natural"),
    "r1-balanced": ("pointwise", "balanced"),
    "r3-natural": ("regret", "natural"),
    "r3-balanced": ("regret", "balanced"),
}
ROLES = ("train", "calibration", "selection")
TIERS = ("low", "medium", "high")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _exact_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"Resume contract changed: {path.name}")
    else:
        write_json(path, value)


def _exact_parquet(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        if not pd.read_parquet(path).equals(frame):
            raise ValueError("Resume training rows or weights changed")
    else:
        frame.to_parquet(path, index=False)


def _record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _verify_completed(root: Path, expected_plan_sha: str) -> dict[str, Any]:
    manifest = _read(root / "manifest.json")
    if (root / "_SUCCESS").read_text().strip() != sha256_file(root / "manifest.json"):
        raise ValueError("Research completion marker mismatch")
    if manifest["run_plan_sha256"] != expected_plan_sha:
        raise ValueError("Completed research arm belongs to another plan")
    if root.name in ARMS:
        method, view = ARMS[root.name]
        if (manifest.get("arm"), manifest.get("objective_method"),
                manifest.get("training_view")) != (root.name, method, view):
            raise ValueError("Completed research arm identity mismatch")
    for name, expected in manifest["artifacts"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or _record(path) != expected:
            raise ValueError(f"Research artifact checksum mismatch: {name}")
    return manifest


def _training_config(
    config: dict[str, Any], train_rows: int, arm: str
) -> TrainingConfig:
    training = config["training"]
    if training.get("initialization") != "fresh-pinned-pretrained-only":
        raise ValueError(
            "All research arms require fresh pinned pretrained initialization"
        )
    if training.get("train_sampling_strategy") != "shuffle":
        raise ValueError(
            "Research R1/R3 uses the same shuffle, never difficulty balancing"
        )
    if training.get("checkpoint_selection") != "fixed_budget_final":
        raise ValueError("Every arm must use its fixed-budget final checkpoint")
    steps = int(training["max_optimizer_steps"])
    if steps < 1 or train_rows < 1:
        raise ValueError(
            "A positive common optimizer budget and train rows are required"
        )
    batch, accumulation = (
        int(training["train_batch_size"]),
        int(training["gradient_accumulation_steps"]),
    )
    if batch < 1 or accumulation < 1:
        raise ValueError("Batch and accumulation must be positive")
    steps_per_epoch = math.ceil(math.ceil(train_rows / batch) / accumulation)
    backbone = config["backbone"]
    if re.fullmatch(r"[0-9a-f]{40}", str(backbone["revision"])) is None:
        raise ValueError("Backbone revision must be an immutable commit SHA")
    result = TrainingConfig(
        training_version=f"{config['experiment_version']}-{arm}",
        model_id=str(backbone["model_id"]),
        model_revision=str(backbone["revision"]),
        train_split="train",
        validation_split="calibration",
        max_length=int(backbone["max_length"]),
        truncation=True,
        train_batch_size=batch,
        eval_batch_size=int(training["eval_batch_size"]),
        epochs=math.ceil(steps / steps_per_epoch),
        max_train_steps=steps,
        gradient_accumulation_steps=accumulation,
        train_sampling_strategy="shuffle",
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        warmup_ratio=float(training["warmup_ratio"]),
        seed=int(config["seed"]),
        device=str(training.get("device", "cuda")),
        precision=str(training["precision"]),
        attention_implementation=str(training["attention_implementation"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
    )
    result.validate()
    return result


def _inputs(
    dataset_dir: Path, config_path: Path
) -> tuple[dict, dict, dict[str, pd.DataFrame]]:
    from .data import (
        canonical_config_sha256,
        load_role,
        load_training_view,
        verify_generalization_dataset,
    )

    config = _read(config_path)
    declared_arms = [
        {"name": name, "method": method, "view": view}
        for name, (method, view) in ARMS.items()
    ]
    if config.get("arms") != declared_arms:
        raise ValueError(
            "Experiment must declare the complete ordered four-arm comparison"
        )
    manifest = verify_generalization_dataset(dataset_dir, roles=ROLES)
    if manifest["config_sha256"] != canonical_config_sha256(config):
        raise ValueError("Dataset and training experiment config differ")
    penalties = validate_regret_lambdas(config["training"]["regret_lambdas"])
    config["training"]["regret_lambdas"] = list(penalties)
    frames = {}
    for view in ("natural", "balanced"):
        frame, _ = load_training_view(dataset_dir, view=view)
        frames[view] = frame.reset_index(drop=True)
    natural, balanced = frames["natural"], frames["balanced"]
    if natural.empty or not natural.drop(columns="sample_weight").equals(
        balanced.drop(columns="sample_weight")
    ):
        raise ValueError("The two ablations must differ only in train loss weights")
    if not natural.sample_weight.eq(1.0).all():
        raise ValueError("Natural training weights must be one")
    weights = balanced.sample_weight.to_numpy(float)
    weight_cap = float(config["balancing"]["max_sample_weight"])
    if (
        not np.isfinite(weights).all()
        or (weights <= 0).any()
        or not np.isclose(weights.mean(), 1.0, atol=1e-10, rtol=0)
        or not math.isfinite(weight_cap)
        or weight_cap < 1
        or (weights > weight_cap + 1e-10).any()
    ):
        raise ValueError(
            "Balanced train weights must be bounded, finite, positive and mean one"
        )
    # The dataset builder guarantees a bounded mean-one distribution; retain its
    # exact weights rather than clipping or normalizing them a second time.
    for role in ("calibration", "selection"):
        frame = load_role(dataset_dir, role).reset_index(drop=True)
        if frame.empty or not frame.sample_weight.eq(1.0).all():
            raise ValueError("Calibration and selection must remain naturally weighted")
        frames[role] = frame
    for role in ("natural", "calibration", "selection"):
        frame = frames[role]
        if frame.sample_id.duplicated().any() or frame.duplicate_group_id.isna().any():
            raise ValueError("Role IDs must be unique with known duplicate groups")
    for left, right in (
        ("natural", "calibration"),
        ("natural", "selection"),
        ("calibration", "selection"),
    ):
        if set(frames[left].sample_id) & set(frames[right].sample_id) or set(
            frames[left].duplicate_group_id
        ) & set(frames[right].duplicate_group_id):
            raise ValueError("Training/calibration/selection overlap")
    return config, manifest, frames


def plan_four_arms(
    *, dataset_dir: Path, config_path: Path, runtime_config_path: Path
) -> dict[str, Any]:
    """Read only pre-holdout roles and return a reviewable plan; no model loading."""
    config, manifest, frames = _inputs(dataset_dir, config_path)
    runtime = load_runtime_contract(runtime_config_path)
    source_root = Path(__file__).resolve().parent.parent
    source_hashes = {
        name: sha256_file(source_root / name)
        for name in (
            "generalization/training.py",
            "generalization/data.py",
            "training/modernbert_training.py",
            "training/modernbert_checkpointing.py",
            "objective_comparison/objectives.py",
            "objective_comparison/training.py",
        )
    }
    plans = {}
    for arm, (method, view) in ARMS.items():
        training = _training_config(config, len(frames[view]), arm)
        weights = frames[view].sample_weight
        plans[arm] = {
            "objective_method": method,
            "training_view": view,
            "training_config": asdict(training),
            "train_rows": len(frames[view]),
            "train_weight_min": float(weights.min()),
            "train_weight_max": float(weights.max()),
            "train_weight_mean": float(weights.mean()),
            "loss": "independent-success-bce-v1"
            if method == "pointwise"
            else "utility-gap-weighted-ce-v1",
        }
    return {
        "schema_version": "generalization-four-arm-plan-v1",
        "experiment_version": config["experiment_version"],
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "dataset_config_sha256": manifest["config_sha256"],
        "experiment_config_sha256": sha256_file(config_path),
        "runtime_config_sha256": sha256_file(runtime_config_path),
        "runtime_version": runtime.runtime_version,
        "training_code_sha256": source_hashes,
        "arms": plans,
        "arm_order": list(ARMS),
        "seed": int(config["seed"]),
        "regret_lambdas": config["training"]["regret_lambdas"],
        "regret_loss_is_expected_regret_minimization": False,
        "weighting_contract": "bounded-train-source-task-loss-weights-same-shuffle-v1",
        "same_train_row_order": True,
        "warm_start_or_imported_previous_run": False,
        "checkpoint_selection": "fixed_budget_final",
        "epoch_policy": "derive_cycles_to_reach_common_optimizer_steps; final_cycle_may_be_partial",
        "scored_roles": ["calibration", "selection"],
        "calibration_rows": len(frames["calibration"]),
        "selection_rows": len(frames["selection"]),
        "task_holdout_accessed": False,
        "legacy_test_or_routerarena_accessed": False,
        "production_approved": False,
    }


def run_four_arms(
    *,
    dataset_dir: Path,
    config_path: Path,
    runtime_config_path: Path,
    output_dir: Path,
    on_checkpoint_saved: Callable[[str, Path], None] | None = None,
    arms_to_run: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run selected arms against one immutable four-arm plan; seal only all four."""
    selected = list(ARMS) if arms_to_run is None else list(arms_to_run)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(ARMS):
        raise ValueError("arms_to_run must contain unique known research arms")
    plan = plan_four_arms(
        dataset_dir=dataset_dir,
        config_path=config_path,
        runtime_config_path=runtime_config_path,
    )
    config, _, frames = _inputs(dataset_dir, config_path)
    runtime = load_runtime_contract(runtime_config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _exact_json(output_dir / "run-plan.json", plan)
    plan_sha = sha256_file(output_dir / "run-plan.json")
    if (output_dir / "_SUCCESS").exists():
        completed = _verify_completed(output_dir, plan_sha)
        for arm in ARMS:
            _verify_completed(output_dir / arm, plan_sha)
        return completed
    manifests = {}
    for arm, (method, view) in ARMS.items():
        if arm not in selected:
            continue
        root = output_dir / arm
        root.mkdir(exist_ok=True)
        if (root / "_SUCCESS").exists():
            manifests[arm] = _verify_completed(root, plan_sha)
            continue
        training = _training_config(config, len(frames[view]), arm)
        combined = pd.concat(
            [
                frames[view].assign(split="train"),
                frames["calibration"].assign(split="calibration"),
            ],
            ignore_index=True,
        )
        training_path = root / "training.parquet"
        _exact_parquet(training_path, combined)
        checkpoints = root / "checkpoints"
        result = run_training(
            training_path,
            root / "model",
            training,
            runtime_contract=runtime,
            objective_method=method,
            regret_lambdas=plan["regret_lambdas"] if method == "regret" else None,
            checkpoint_options=CheckpointOptions(
                checkpoint_dir=checkpoints,
                save_every_optimizer_steps=int(
                    config["training"].get("checkpoint_every_optimizer_steps", 500)
                ),
                keep_last_checkpoints=2,
                resume_from=latest_completed_checkpoint(checkpoints),
                on_checkpoint_saved=(
                    lambda path, arm=arm: on_checkpoint_saved(arm, path)
                )
                if on_checkpoint_saved
                else None,
            ),
        )
        if (
            result.metrics["optimization"]["optimizer_steps"]
            != training.max_train_steps
        ):
            raise ValueError("Research arm did not finish the common step budget")
        seal_artifact_directory(
            root / "model",
            manifest_name=ARTIFACT_MANIFEST_NAME,
            metadata={
                "research_only": True,
                "arm": arm,
                "run_plan_sha256": plan_sha,
                "training_fingerprint": result.metrics["optimization"][
                    "training_fingerprint"
                ],
            },
        )
        AutoModel, AutoTokenizer, _ = _require_transformers()
        model = AutoModel.from_pretrained(
            root / "model", attn_implementation=training.attention_implementation
        ).to(result.metrics["model"]["device"])
        tokenizer = AutoTokenizer.from_pretrained(root / "model")
        for role in ("calibration", "selection"):
            frame = frames[role]
            raw = _predict_logits(
                model,
                tokenizer,
                frame,
                batch_size=training.eval_batch_size,
                max_length=training.max_length,
                device=result.metrics["model"]["device"],
            )
            if raw.shape != (len(frame), 3) or not np.isfinite(raw).all():
                raise ValueError("Invalid research inference logits")
            scored = frame.drop(columns=["model_input_text"]).copy()
            for index, tier in enumerate(TIERS):
                scored[f"score_{tier}"] = raw[:, index]
            scored["research_arm"] = arm
            scored["objective_method"] = method
            scored.to_parquet(root / f"{role}-scores.parquet", index=False)
        del model
        verify_artifact_directory(root / "model", manifest_name=ARTIFACT_MANIFEST_NAME)
        artifact_paths = [
            training_path,
            *sorted((root / "model").rglob("*")),
            root / "calibration-scores.parquet",
            root / "selection-scores.parquet",
        ]
        manifest = {
            "schema_version": "generalization-four-arm-run-v1",
            "run_plan_sha256": plan_sha,
            "arm": arm,
            "objective_method": method,
            "training_view": view,
            "optimizer_steps": training.max_train_steps,
            "task_holdout_accessed": False,
            "policy_or_arm_selected": False,
            "production_approved": False,
            "artifacts": {
                str(path.relative_to(root)): _record(path)
                for path in artifact_paths
                if path.is_file()
            },
        }
        write_json(root / "manifest.json", manifest)
        (root / "_SUCCESS").write_text(sha256_file(root / "manifest.json") + "\n")
        manifests[arm] = manifest
    # Verify the same immutable input contract again after all computation.
    if (
        plan_four_arms(
            dataset_dir=dataset_dir,
            config_path=config_path,
            runtime_config_path=runtime_config_path,
        )
        != plan
    ):
        raise ValueError("Research inputs changed during training")
    # Independently produced arm directories can be merged under the identical
    # run-plan. Verify every completed arm, including those skipped this time.
    manifests = {
        arm: _verify_completed(output_dir / arm, plan_sha)
        for arm in ARMS
        if (output_dir / arm / "_SUCCESS").exists()
    }
    if len(manifests) != len(ARMS):
        partial = {
            "schema_version": "generalization-partial-run-v1",
            "run_plan_sha256": plan_sha,
            "comparison_complete": False,
            "requested_arms": selected,
            "completed_arms": list(manifests),
            "missing_arms": [arm for arm in ARMS if arm not in manifests],
            "task_holdout_accessed": False,
            "policy_or_arm_selected": False,
            "production_approved": False,
            "artifacts": {
                f"{arm}/manifest.json": _record(output_dir / arm / "manifest.json")
                for arm in manifests
            },
        }
        write_json(output_dir / "partial-manifest.json", partial)
        return partial
    artifacts = {"run-plan.json": _record(output_dir / "run-plan.json")}
    for arm in ARMS:
        _verify_completed(output_dir / arm, plan_sha)
        for name in ("manifest.json", "_SUCCESS"):
            artifacts[f"{arm}/{name}"] = _record(output_dir / arm / name)
    manifest = {
        "schema_version": "generalization-four-arm-comparison-v1",
        "run_plan_sha256": plan_sha,
        "comparison_complete": True,
        "arms": list(manifests),
        "task_holdout_accessed": False,
        "policy_or_arm_selected": False,
        "production_approved": False,
        "artifacts": artifacts,
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "_SUCCESS").write_text(
        sha256_file(output_dir / "manifest.json") + "\n"
    )
    return manifest
