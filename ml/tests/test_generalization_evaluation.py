"""Leakage and artifact-boundary tests using local synthetic logits, no models."""

import numpy as np
import pandas as pd
import pytest
from llm_router_labeling.generalization import evaluation as ev
from llm_router_labeling.generalization.data import canonical_config_sha256
from llm_router_labeling.generalization.training import ARMS, _record
from llm_router_labeling.shared.dataset_io import sha256_file, write_json


def seal(root, manifest):
    write_json(root / "manifest.json", manifest)
    (root / "_SUCCESS").write_text(sha256_file(root / "manifest.json"))


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    config = {
        "training": {"max_optimizer_steps": 3},
        "evaluation": {
            "relative_cost_budgets": [0.0, 0.5, 1.0],
            "penalty_grid": [0.0, 1.0, 100.0],
            "default_relative_cost_budget": 0.5,
            "selection_metric": ev.METRIC,
            "require_selection_budget_feasible": True,
            "lambda_selection_role": "calibration",
            "arm_selection_role": "selection",
            "task_holdout_requires_selection_lock": True,
        },
    }
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    dataset, runs = tmp_path / "dataset", tmp_path / "runs"
    dataset.mkdir()
    runs.mkdir()
    manifest = {"config_sha256": canonical_config_sha256(config)}
    write_json(dataset / "manifest.json", manifest)
    frames = {}
    accesses = []
    for role in ("train", "calibration", "selection", "task_holdout"):
        records = []
        for source in ("a", "b"):
            for i in range(4):
                row = {
                    "sample_id": f"{role}-{source}-{i}",
                    "source_dataset": source,
                    "task_group_id": "unseen" if role == "task_holdout" else "seen",
                    "generalization_role": role,
                    "model_input_text": f"{role} {source} {i}",
                    "duplicate_group_id": f"{role}-{source}-{i}",
                    "sample_weight": 1.0,
                }
                for j, tier in enumerate(ev.TIERS):
                    row.update(
                        {
                            f"quality_{tier}": [0.0, 0.5, 1.0][j],
                            f"actual_cost_{tier}": [1.0, 2.0, 3.0][j],
                            f"decision_cost_{tier}": [0.0, 0.5, 1.0][j],
                            f"success_{tier}": float(j > 0),
                            f"model_{tier}": f"{source}-{tier}",
                        }
                    )
                records.append(row)
        frames[role] = pd.DataFrame(records)

    def load(_root, role):
        accesses.append(role)
        return frames[role].copy()

    monkeypatch.setattr(ev, "verify_generalization_dataset", lambda *a, **k: manifest)
    monkeypatch.setattr(ev, "load_role", load)
    plan = {
        "dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
        "dataset_config_sha256": manifest["config_sha256"],
        "experiment_config_sha256": sha256_file(config_path),
        "arm_order": list(ARMS),
    }
    write_json(runs / "run-plan.json", plan)
    plan_sha = sha256_file(runs / "run-plan.json")
    for arm, (method, view) in ARMS.items():
        root = runs / arm
        root.mkdir()
        artifacts = {}
        for role in ("calibration", "selection"):
            scored = frames[role].drop(columns="model_input_text").copy()
            for j, tier in enumerate(ev.TIERS):
                scored[f"score_{tier}"] = [-1.0, 0.0, 1.0][j]
            scored["research_arm"], scored["objective_method"] = arm, method
            path = root / f"{role}-scores.parquet"
            scored.to_parquet(path, index=False)
            artifacts[path.name] = _record(path)
        seal(
            root,
            {
                "run_plan_sha256": plan_sha,
                "arm": arm,
                "objective_method": method,
                "training_view": view,
                "optimizer_steps": 3,
                "artifacts": artifacts,
            },
        )

    def reseal():
        for arm in ARMS:
            root = runs / arm
            m = ev._read(root / "manifest.json")
            m["artifacts"] = {name: _record(root / name) for name in m["artifacts"]}
            seal(root, m)
        artifacts = {"run-plan.json": _record(runs / "run-plan.json")}
        for arm in ARMS:
            for name in ("manifest.json", "_SUCCESS"):
                artifacts[f"{arm}/{name}"] = _record(runs / arm / name)
        seal(
            runs,
            {
                "run_plan_sha256": plan_sha,
                "comparison_complete": True,
                "arms": list(ARMS),
                "artifacts": artifacts,
            },
        )

    reseal()
    return (
        {"dataset_dir": dataset, "runs_dir": runs, "config_path": config_path},
        frames,
        accesses,
        reseal,
    )


def test_lock_does_not_open_holdout_and_is_immutable(experiment, tmp_path):
    args, _, accesses, _ = experiment
    output = tmp_path / "lock"
    result = ev.lock_selection(**args, output_dir=output)
    assert "task_holdout" not in accesses
    assert result["task_holdout_accessed"] is False
    assert result["selection_reports"][result["selected_arm"]][
        "default_budget_feasible"
    ]
    with pytest.raises(FileExistsError):
        ev.lock_selection(**args, output_dir=output)


def test_score_tamper_is_rejected(experiment, tmp_path):
    args, _, _, _ = experiment
    with (args["runs_dir"] / "r1-natural/selection-scores.parquet").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        ev.lock_selection(**args, output_dir=tmp_path / "lock")


def test_resealed_wrong_outcomes_are_rejected(experiment, tmp_path):
    args, _, _, reseal = experiment
    path = args["runs_dir"] / "r1-natural/selection-scores.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "quality_low"] = 0.99
    frame.to_parquet(path, index=False)
    reseal()
    with pytest.raises(ValueError, match="outcomes"):
        ev.lock_selection(**args, output_dir=tmp_path / "lock")


def test_selection_never_recalibrates_lambda(experiment):
    _, frames, _, _ = experiment
    frame = frames["selection"]
    policies = {
        source: {"0.0000": 0.0, "0.5000": 0.0, "1.0000": 0.0} for source in ("a", "b")
    }
    # A new penalty would make the policy feasible, but held-out quality cannot choose it.
    report = ev.fixed_policy_report(
        frame,
        np.tile([-1.0, 0.0, 1.0], (len(frame), 1)),
        method="pointwise",
        policies=policies,
        budgets=[0.0, 0.5, 1.0],
        default=0.5,
    )
    assert report["default_budget_feasible"] is False
    assert report["per_source"]["a"]["budgets"]["0.5000"]["penalty"] == 0.0
    assert report["per_source"]["a"]["budgets"]["0.5000"]["budget_exceeded"]


def test_no_feasible_arm_means_no_lock(experiment, monkeypatch, tmp_path):
    args, _, accesses, _ = experiment
    original = ev.fixed_policy_report

    def infeasible(*a, **kw):
        result = original(*a, **kw)
        result["default_budget_feasible"] = False
        return result

    monkeypatch.setattr(ev, "fixed_policy_report", infeasible)
    out = tmp_path / "lock"
    with pytest.raises(ValueError, match="No arm"):
        ev.lock_selection(**args, output_dir=out)
    assert not out.exists() and "task_holdout" not in accesses


def test_holdout_requires_lock_before_reading_data(experiment, tmp_path):
    args, _, accesses, _ = experiment
    with pytest.raises(FileNotFoundError):
        ev.infer_holdout(
            **args, lock_dir=tmp_path / "missing", output_dir=tmp_path / "scores"
        )
    assert "task_holdout" not in accesses


def test_changed_lock_is_rejected_before_holdout(experiment, tmp_path):
    args, _, accesses, _ = experiment
    lock = tmp_path / "lock"
    ev.lock_selection(**args, output_dir=lock)
    with (lock / "manifest.json").open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="lock checksum"):
        ev.infer_holdout(**args, lock_dir=lock, output_dir=tmp_path / "scores")
    assert "task_holdout" not in accesses


@pytest.mark.parametrize("mutation", ["winner", "policies", "metric"])
def test_resealed_invalid_lock_is_rejected_before_holdout(
    experiment, tmp_path, mutation
):
    args, _, accesses, _ = experiment
    lock = tmp_path / "lock"
    result = ev.lock_selection(**args, output_dir=lock)
    if mutation == "winner":
        result["selected_arm"] = next(a for a in ARMS if a != result["selected_arm"])
    elif mutation == "policies":
        result["policies"] = {arm: {} for arm in ARMS}
    else:
        result["selection_metric"] = "bogus"
    seal(lock, result)
    with pytest.raises(ValueError, match="pre-holdout evidence"):
        ev.infer_holdout(**args, lock_dir=lock, output_dir=tmp_path / "scores")
    assert "task_holdout" not in accesses


def test_holdout_keeps_winner_and_unseen_task_fallback(experiment, tmp_path):
    args, frames, _, _ = experiment
    lock = tmp_path / "lock"
    selected = ev.lock_selection(**args, output_dir=lock)
    scores = tmp_path / "scores"
    scores.mkdir()
    artifacts = {}
    for arm, (method, _) in ARMS.items():
        frame = frames["task_holdout"].drop(columns="model_input_text").copy()
        for j, tier in enumerate(ev.TIERS):
            frame[f"score_{tier}"] = (
                [100.0, 0.0, -100.0][j]
                if arm == selected["selected_arm"]
                else [-100.0, 0.0, 100.0][j]
            )
        frame["research_arm"], frame["objective_method"] = arm, method
        path = scores / f"{arm}-scores.parquet"
        frame.to_parquet(path, index=False)
        artifacts[path.name] = _record(path)
    seal(
        scores,
        {
            "schema_version": "generalization-holdout-inference-v1",
            "selected_arm": selected["selected_arm"],
            "selection_lock_sha256": sha256_file(lock / "manifest.json"),
            "artifacts": artifacts,
        },
    )
    report = ev.evaluate_holdout(
        **args, lock_dir=lock, scores_dir=scores, output_dir=tmp_path / "report"
    )
    assert report["selected_arm"] == selected["selected_arm"]
    assert report["arm_reselected_on_holdout"] is False
    baseline = report["reports"][selected["selected_arm"]]["per_source"]["a"][
        "train_fitted_baselines"
    ]
    assert (
        baseline["train_source_task_with_source_fallback"]
        == baseline["train_best_single_within_source_pool"]
    )


def test_degenerate_holdout_normalization_is_na(experiment):
    _, frames, _, _ = experiment
    frame = frames["task_holdout"].copy()
    frame["quality_high"] = frame.quality_low
    policies = {s: {"0.0000": 0.0, "0.5000": 1.0, "1.0000": 0.0} for s in ("a", "b")}
    report = ev.fixed_policy_report(
        frame,
        np.zeros((len(frame), 3)),
        method="regret",
        policies=policies,
        budgets=[0.0, 0.5, 1.0],
        default=0.5,
    )
    assert report[ev.METRIC] is None
    assert report["default_budget_feasible"] is True
    assert report["default_source_macro_relative_cost"] == 0.0
