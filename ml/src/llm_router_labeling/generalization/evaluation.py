"""Calibrate, lock an arm, then evaluate untouched G1 tasks without retuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..objective_comparison.evaluation import route_actions, summarize_source_curves
from ..objective_comparison.training import _predict_logits, _require_transformers
from ..shared.dataset_io import sha256_file, write_json
from .data import canonical_config_sha256, load_role, verify_generalization_dataset
from .training import ARMS, TIERS, _read, _record, _verify_completed

METRIC = "source_macro_locked_policy_quality_auc"


def train_baselines(train: pd.DataFrame) -> dict:
    """Fit candidate-index baselines within each actual source pool, on train only."""

    def best(rows):
        q = rows[[f"quality_{t}" for t in TIERS]].mean().to_numpy()
        c = rows[[f"actual_cost_{t}" for t in TIERS]].mean().to_numpy()
        return int(min(range(3), key=lambda i: (-q[i], c[i], i)))

    return {
        str(source): {
            "source_best": best(rows),
            "task_best": {
                str(task): best(group) for task, group in rows.groupby("task_group_id")
            },
            "unseen_task_fallback": "source_best_fitted_on_train",
        }
        for source, rows in train.groupby("source_dataset")
    }


def _settings(config: dict) -> tuple[list[float], float]:
    spec = config["evaluation"]
    budgets = [float(x) for x in spec["relative_cost_budgets"]]
    penalties = np.asarray(spec["penalty_grid"], dtype=float)
    default = float(spec["default_relative_cost_budget"])
    if (
        len(budgets) < 2
        or budgets != sorted(set(budgets))
        or not np.isfinite(budgets).all()
        or budgets[0] != 0
        or budgets[-1] != 1
        or default not in budgets
        or len(penalties) == 0
        or not np.isfinite(penalties).all()
        or (penalties < 0).any()
        or spec["selection_metric"] != METRIC
        or spec.get("require_selection_budget_feasible") is not True
        or spec.get("lambda_selection_role") != "calibration"
        or spec.get("arm_selection_role") != "selection"
        or spec.get("task_holdout_requires_selection_lock") is not True
    ):
        raise ValueError("Invalid frozen G1 evaluation contract")
    return budgets, default


def _validate_scores(frame: pd.DataFrame, scores: pd.DataFrame, arm: str) -> np.ndarray:
    columns = [c for c in frame.columns if c != "model_input_text"]
    if scores.sample_id.duplicated().any() or not frame[columns].reset_index(
        drop=True
    ).equals(scores[columns].reset_index(drop=True)):
        raise ValueError(
            "Score IDs, order, role or outcomes differ from sealed dataset"
        )
    if (
        not scores.research_arm.eq(arm).all()
        or not scores.objective_method.eq(ARMS[arm][0]).all()
    ):
        raise ValueError("Score objective/arm mismatch")
    raw = scores[[f"score_{tier}" for tier in TIERS]].to_numpy(float)
    if raw.shape != (len(frame), 3) or not np.isfinite(raw).all():
        raise ValueError("Scores must be finite three-candidate logits")
    return raw


def _verify_runs(dataset_dir: Path, runs_dir: Path, config_path: Path) -> dict:
    config = _read(config_path)
    _settings(config)
    manifest = verify_generalization_dataset(
        dataset_dir, roles=("train", "calibration", "selection")
    )
    plan = _read(runs_dir / "run-plan.json")
    if (
        manifest["config_sha256"] != canonical_config_sha256(config)
        or plan["dataset_manifest_sha256"] != sha256_file(dataset_dir / "manifest.json")
        or plan["dataset_config_sha256"] != manifest["config_sha256"]
        or plan["experiment_config_sha256"] != sha256_file(config_path)
        or plan["arm_order"] != list(ARMS)
    ):
        raise ValueError("Dataset, config and completed training plan differ")
    plan_sha = sha256_file(runs_dir / "run-plan.json")
    completed = _verify_completed(runs_dir, plan_sha)
    if completed.get("comparison_complete") is not True or completed["arms"] != list(
        ARMS
    ):
        raise ValueError("All four arms must complete before selection")
    for arm, (method, view) in ARMS.items():
        item = _verify_completed(runs_dir / arm, plan_sha)
        if (
            item["arm"] != arm
            or item["objective_method"] != method
            or item["training_view"] != view
        ):
            raise ValueError("Arm manifest mismatch")
        if item["optimizer_steps"] != int(config["training"]["max_optimizer_steps"]):
            raise ValueError("Unequal optimizer budgets")
    return config


def fixed_policy_report(
    frame: pd.DataFrame,
    raw: np.ndarray,
    *,
    method: str,
    policies: dict[str, dict[str, float]],
    budgets: list[float],
    default: float,
    baselines: dict | None = None,
) -> dict[str, Any]:
    """Only applies calibration policies. Outcomes here never choose a penalty."""
    sources = sorted(frame.source_dataset.astype(str).unique())
    if set(sources) != set(policies):
        raise ValueError("Evaluation source pools must match calibration")
    result = {}
    for source in sources:
        mask = frame.source_dataset.eq(source).to_numpy()
        rows, logits = frame.loc[mask], raw[mask]
        q = rows[[f"quality_{t}" for t in TIERS]].to_numpy(float)
        c = rows[[f"actual_cost_{t}" for t in TIERS]].to_numpy(float)
        d = rows[[f"decision_cost_{t}" for t in TIERS]].to_numpy(float)
        qspan, cspan = q[:, 2].mean() - q[:, 0].mean(), c[:, 2].mean() - c[:, 0].mean()
        valid = bool(qspan > 0 and cspan > 0)
        points, normalized = {}, []
        for budget in budgets:
            key = f"{budget:.4f}"
            penalty = policies[source][key]
            actions = route_actions(method, logits, d, penalty)
            selected_q = q[np.arange(len(rows)), actions]
            selected_c = c[np.arange(len(rows)), actions]
            relative = (
                float((selected_c.mean() - c[:, 0].mean()) / cspan)
                if cspan > 0
                else None
            )
            quality = float(selected_q.mean())
            points[key] = {
                "penalty": penalty,
                "mean_quality": quality,
                "mean_actual_cost": float(selected_c.mean()),
                "relative_cost": relative,
                "budget_exceeded": None
                if relative is None
                else bool(relative > budget + 1e-12),
                "per_task": {
                    str(task): {
                        "rows": int((rows.task_group_id == task).sum()),
                        "mean_quality": float(
                            selected_q[rows.task_group_id.eq(task)].mean()
                        ),
                        "mean_actual_cost": float(
                            selected_c[rows.task_group_id.eq(task)].mean()
                        ),
                    }
                    for task in sorted(rows.task_group_id.unique())
                },
            }
            normalized.append(
                float(np.clip((quality - q[:, 0].mean()) / qspan, 0, 1))
                if valid
                else np.nan
            )
        auc = (
            float(np.trapezoid(normalized, budgets) / np.ptp(budgets))
            if valid
            else None
        )
        result[source] = {
            "rows": len(rows),
            "normalization_valid": valid,
            "locked_policy_quality_auc": auc,
            "budgets": points,
            "always": {
                tier: {
                    "quality": float(q[:, i].mean()),
                    "actual_cost": float(c[:, i].mean()),
                }
                for i, tier in enumerate(TIERS)
            },
            "oracle_quality_upper_bound": float(q.max(axis=1).mean()),
        }
        if baselines is not None:
            fitted = baselines[source]
            baseline_actions = {
                "train_best_single_within_source_pool": np.full(
                    len(rows), fitted["source_best"], dtype=int
                ),
                "train_source_task_with_source_fallback": np.array(
                    [
                        fitted["task_best"].get(str(task), fitted["source_best"])
                        for task in rows.task_group_id
                    ],
                    dtype=int,
                ),
            }
            result[source]["train_fitted_baselines"] = {
                name: {
                    "mean_quality": float(q[np.arange(len(rows)), actions].mean()),
                    "mean_actual_cost": float(c[np.arange(len(rows)), actions].mean()),
                }
                for name, actions in baseline_actions.items()
            }
    key = f"{default:.4f}"
    valid = bool(result) and all(x["normalization_valid"] for x in result.values())
    cost_valid = bool(result) and all(
        x["budgets"][key]["relative_cost"] is not None for x in result.values()
    )
    return {
        METRIC: float(
            np.mean([x["locked_policy_quality_auc"] for x in result.values()])
        )
        if valid
        else None,
        "default_budget_feasible": cost_valid
        and all(x["budgets"][key]["budget_exceeded"] is False for x in result.values()),
        "default_source_macro_quality": float(
            np.mean([x["budgets"][key]["mean_quality"] for x in result.values()])
        ),
        "default_source_macro_relative_cost": float(
            np.mean([x["budgets"][key]["relative_cost"] for x in result.values()])
        )
        if cost_valid
        else None,
        "per_source": result,
        "metric_note": "AUC over nominal budget grid under locked calibration policies; actual budget violations reported separately; not a reoptimized frontier nAUC",
    }


def _seal_new(root: Path, manifest: dict) -> dict:
    # Refuse reuse so a completed selection or holdout report cannot be overwritten.
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "manifest.json", manifest)
    (root / "_SUCCESS").write_text(sha256_file(root / "manifest.json") + "\n")
    return manifest


def _selection_payload(*, dataset_dir: Path, runs_dir: Path, config_path: Path) -> dict:
    config = _verify_runs(dataset_dir, runs_dir, config_path)
    budgets, default = _settings(config)
    calibration, selection = (
        load_role(dataset_dir, role) for role in ("calibration", "selection")
    )
    baselines = train_baselines(load_role(dataset_dir, "train"))
    reports, policies, calibration_reports = {}, {}, {}
    for arm, (method, _) in ARMS.items():
        raw_cal, raw_sel = (
            _validate_scores(
                frame, pd.read_parquet(runs_dir / arm / f"{role}-scores.parquet"), arm
            )
            for role, frame in (("calibration", calibration), ("selection", selection))
        )
        cal_report, locked, _ = summarize_source_curves(
            calibration,
            raw_cal,
            method=method,
            penalties=config["evaluation"]["penalty_grid"],
            budgets=budgets,
            retention_targets=[],
        )
        if cal_report["source_macro_quality_cost_nauc"] is None:
            raise ValueError("Calibration source normalization is invalid")
        policies[arm], calibration_reports[arm] = locked, cal_report
        reports[arm] = fixed_policy_report(
            selection,
            raw_sel,
            method=method,
            policies=locked,
            budgets=budgets,
            default=default,
            baselines=baselines,
        )
    eligible = [
        a
        for a in ARMS
        if reports[a]["default_budget_feasible"] and reports[a][METRIC] is not None
    ]
    if not eligible:
        raise ValueError(
            "No arm meets the frozen selection budget in every source; do not inspect holdout. A revised protocol needs a new experiment version."
        )
    winner = min(
        eligible,
        key=lambda a: (
            -reports[a][METRIC],
            reports[a]["default_source_macro_relative_cost"],
            a,
        ),
    )
    _verify_runs(dataset_dir, runs_dir, config_path)
    return {
        "schema_version": "generalization-selection-lock-v1",
        "selected_arm": winner,
        "selection_metric": METRIC,
        "eligible_arms": eligible,
        "tie_break": "descending metric, ascending default source-macro relative cost, lexical arm",
        "policies": policies,
        "baselines": baselines,
        "calibration_reports": calibration_reports,
        "selection_reports": reports,
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "run_manifest_sha256": sha256_file(runs_dir / "manifest.json"),
        "config_sha256": sha256_file(config_path),
        "evaluation_source_sha256": {
            "generalization/evaluation.py": sha256_file(Path(__file__)),
            "objective_comparison/evaluation.py": sha256_file(
                Path(__file__).parent.parent / "objective_comparison/evaluation.py"
            ),
        },
        "task_holdout_accessed": False,
        "production_approved": False,
    }


def lock_selection(
    *, dataset_dir: Path, runs_dir: Path, config_path: Path, output_dir: Path
) -> dict:
    return _seal_new(
        output_dir,
        _selection_payload(
            dataset_dir=dataset_dir, runs_dir=runs_dir, config_path=config_path
        ),
    )


def _verify_lock(
    *, dataset_dir: Path, runs_dir: Path, config_path: Path, lock_dir: Path
) -> tuple[dict, dict]:
    # This happens before opening holdout data or loading a model.
    lock = _read(lock_dir / "manifest.json")
    if (lock_dir / "_SUCCESS").read_text().strip() != sha256_file(
        lock_dir / "manifest.json"
    ):
        raise ValueError("Selection lock checksum mismatch")
    config = _verify_runs(dataset_dir, runs_dir, config_path)
    if (
        lock["schema_version"] != "generalization-selection-lock-v1"
        or lock["dataset_manifest_sha256"] != sha256_file(dataset_dir / "manifest.json")
        or lock["run_manifest_sha256"] != sha256_file(runs_dir / "manifest.json")
        or lock["config_sha256"] != sha256_file(config_path)
        or lock["selected_arm"] not in ARMS
        or set(lock["policies"]) != set(ARMS)
    ):
        raise ValueError("Selection lock belongs to different experiment inputs")
    # A self-consistent checksum alone does not establish correct calibration or
    # selection. Reproduce the complete decision from sealed pre-holdout evidence.
    expected = _selection_payload(
        dataset_dir=dataset_dir, runs_dir=runs_dir, config_path=config_path
    )
    if lock != expected:
        raise ValueError(
            "Selection lock policies or winner do not match pre-holdout evidence"
        )
    return config, lock


def infer_holdout(
    *,
    dataset_dir: Path,
    runs_dir: Path,
    config_path: Path,
    lock_dir: Path,
    output_dir: Path,
    device: str = "cuda",
) -> dict:
    config, lock = _verify_lock(
        dataset_dir=dataset_dir,
        runs_dir=runs_dir,
        config_path=config_path,
        lock_dir=lock_dir,
    )
    # Refuse a completed or partially populated output; retries use another path.
    output_dir.mkdir(parents=True, exist_ok=False)
    frame = load_role(dataset_dir, "task_holdout")
    AutoModel, AutoTokenizer, _ = _require_transformers()
    artifacts = {}
    for arm, (method, _) in ARMS.items():
        model_path = runs_dir / arm / "model"
        model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            attn_implementation=config["training"]["attention_implementation"],
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        raw = _predict_logits(
            model,
            tokenizer,
            frame,
            batch_size=int(config["training"]["eval_batch_size"]),
            max_length=int(config["backbone"]["max_length"]),
            device=device,
        )
        scores = frame.drop(columns="model_input_text").copy()
        for i, tier in enumerate(TIERS):
            scores[f"score_{tier}"] = raw[:, i]
        scores["research_arm"], scores["objective_method"] = arm, method
        _validate_scores(frame, scores, arm)
        path = output_dir / f"{arm}-scores.parquet"
        scores.to_parquet(path, index=False)
        artifacts[path.name] = _record(path)
        del model
    _verify_lock(
        dataset_dir=dataset_dir,
        runs_dir=runs_dir,
        config_path=config_path,
        lock_dir=lock_dir,
    )
    load_role(dataset_dir, "task_holdout")
    manifest = {
        "schema_version": "generalization-holdout-inference-v1",
        "selected_arm": lock["selected_arm"],
        "selection_lock_sha256": sha256_file(lock_dir / "manifest.json"),
        "artifacts": artifacts,
        "task_holdout_accessed": True,
        "production_approved": False,
        "device": device,
        "max_length": int(config["backbone"]["max_length"]),
        "eval_batch_size": int(config["training"]["eval_batch_size"]),
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "_SUCCESS").write_text(
        sha256_file(output_dir / "manifest.json") + "\n"
    )
    return manifest


def evaluate_holdout(
    *,
    dataset_dir: Path,
    runs_dir: Path,
    config_path: Path,
    lock_dir: Path,
    scores_dir: Path,
    output_dir: Path,
) -> dict:
    config, lock = _verify_lock(
        dataset_dir=dataset_dir,
        runs_dir=runs_dir,
        config_path=config_path,
        lock_dir=lock_dir,
    )
    inference = _read(scores_dir / "manifest.json")
    if (
        (scores_dir / "_SUCCESS").read_text().strip()
        != sha256_file(scores_dir / "manifest.json")
        or inference["schema_version"] != "generalization-holdout-inference-v1"
        or inference["selection_lock_sha256"] != sha256_file(lock_dir / "manifest.json")
        or inference["selected_arm"] != lock["selected_arm"]
    ):
        raise ValueError("Holdout inference seal or selection lock mismatch")
    frame = load_role(dataset_dir, "task_holdout")
    budgets, default = _settings(config)
    reports = {}
    for arm, (method, _) in ARMS.items():
        path = scores_dir / f"{arm}-scores.parquet"
        if _record(path) != inference["artifacts"][path.name]:
            raise ValueError("Holdout scores checksum mismatch")
        raw = _validate_scores(frame, pd.read_parquet(path), arm)
        reports[arm] = fixed_policy_report(
            frame,
            raw,
            method=method,
            policies=lock["policies"][arm],
            budgets=budgets,
            default=default,
            baselines=lock["baselines"],
        )
    return _seal_new(
        output_dir,
        {
            "schema_version": "generalization-holdout-report-v1",
            "selected_arm": lock["selected_arm"],
            "arm_reselected_on_holdout": False,
            "selection_lock_sha256": sha256_file(lock_dir / "manifest.json"),
            "inference_manifest_sha256": sha256_file(scores_dir / "manifest.json"),
            "reports": reports,
            "production_approved": False,
            "interpretation": "Task-disjoint diagnostic for freshly initialized G1 arms; parent data and taxonomy were historically exposed. No deployment or model-pool transfer approval.",
        },
    )
