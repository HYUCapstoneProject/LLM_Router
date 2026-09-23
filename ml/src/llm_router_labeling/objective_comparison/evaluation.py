from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .data import TIERS


class ObjectiveEvaluationError(ValueError):
    """Raised when routing scores or outcomes violate the evaluation contract."""


def _score_transform(method: str, scores: np.ndarray) -> np.ndarray:
    if method == "pointwise":
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))
    if method == "ordinal":
        shifted = scores - scores.max(axis=1, keepdims=True)
        values = np.exp(shifted)
        return values / values.sum(axis=1, keepdims=True)
    return scores


def route_actions(
    method: str,
    scores: np.ndarray,
    decision_costs: np.ndarray,
    penalty: float,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    decision_costs = np.asarray(decision_costs, dtype=float)
    if scores.shape != decision_costs.shape or scores.ndim != 2 or scores.shape[1] != 3:
        raise ObjectiveEvaluationError("scores and decision_costs must both have shape (N, 3)")
    transformed = _score_transform(method, scores)
    return np.argmax(transformed - float(penalty) * decision_costs, axis=1)


def _selected(matrix: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return matrix[np.arange(len(actions)), actions]


def evaluate_penalty_curve(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    method: str,
    penalties: Iterable[float],
) -> pd.DataFrame:
    if len(frame) != len(scores):
        raise ObjectiveEvaluationError("Prediction and outcome row counts differ")
    quality = frame.loc[:, [f"quality_{tier}" for tier in TIERS]].to_numpy(float)
    actual_cost = frame.loc[:, [f"actual_cost_{tier}" for tier in TIERS]].to_numpy(float)
    decision_cost = frame.loc[:, [f"decision_cost_{tier}" for tier in TIERS]].to_numpy(float)
    source = frame["source_dataset"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    always_low_cost = float(actual_cost[:, 0].mean())
    always_high_cost = float(actual_cost[:, 2].mean())
    cost_span = always_high_cost - always_low_cost
    if cost_span <= 0:
        raise ObjectiveEvaluationError("Always-high mean cost must exceed always-low mean cost")
    for penalty in penalties:
        actions = route_actions(method, scores, decision_cost, float(penalty))
        selected_quality = _selected(quality, actions)
        selected_cost = _selected(actual_cost, actions)
        per_source_quality = {
            value: float(selected_quality[source == value].mean())
            for value in sorted(set(source))
        }
        rows.append(
            {
                "penalty": float(penalty),
                "mean_quality": float(selected_quality.mean()),
                "source_macro_quality": float(np.mean(list(per_source_quality.values()))),
                "mean_actual_cost": float(selected_cost.mean()),
                "relative_cost": float((selected_cost.mean() - always_low_cost) / cost_span),
                "low_rate": float((actions == 0).mean()),
                "medium_rate": float((actions == 1).mean()),
                "high_rate": float((actions == 2).mean()),
                "routerbench_quality": float(
                    selected_quality[source == "routerbench"].mean()
                ),
                "xroutebench_quality": float(
                    selected_quality[source == "xroutebench"].mean()
                ),
            }
        )
    curve = pd.DataFrame(rows).sort_values(
        ["relative_cost", "mean_quality", "penalty"],
        ascending=[True, False, True],
        kind="stable",
    )
    return curve.drop_duplicates("relative_cost", keep="first").reset_index(drop=True)


def _quality_at_budget(curve: pd.DataFrame, budget: float) -> tuple[float, float]:
    eligible = curve.loc[curve["relative_cost"].le(float(budget) + 1e-12)]
    if eligible.empty:
        selected = curve.iloc[curve["relative_cost"].argmin()]
    else:
        selected = eligible.sort_values(
            ["mean_quality", "relative_cost", "penalty"],
            ascending=[False, True, True],
            kind="stable",
        ).iloc[0]
    return float(selected["mean_quality"]), float(selected["penalty"])


def summarize_curve(
    frame: pd.DataFrame,
    curve: pd.DataFrame,
    *,
    budgets: Sequence[float],
    retention_targets: Sequence[float],
) -> tuple[dict[str, Any], dict[str, float]]:
    quality = frame.loc[:, [f"quality_{tier}" for tier in TIERS]].to_numpy(float)
    low_quality = float(quality[:, 0].mean())
    high_quality = float(quality[:, 2].mean())
    oracle_quality = float(quality.max(axis=1).mean())
    budget_quality: dict[str, float] = {}
    budget_penalties: dict[str, float] = {}
    for budget in budgets:
        value, penalty = _quality_at_budget(curve, float(budget))
        key = f"{float(budget):.4f}"
        budget_quality[key] = value
        budget_penalties[key] = penalty
    ordered_budgets = np.array([float(value) for value in budgets], dtype=float)
    ordered_quality = np.array(
        [budget_quality[f"{value:.4f}"] for value in ordered_budgets], dtype=float
    )
    quality_span = max(high_quality - low_quality, 1e-12)
    recovered = np.clip((ordered_quality - low_quality) / quality_span, 0.0, 1.0)
    nauc = float(np.trapezoid(recovered, ordered_budgets) / max(np.ptp(ordered_budgets), 1e-12))
    minimum_cost_for_retention: dict[str, float | None] = {}
    for target in retention_targets:
        required = high_quality * float(target)
        eligible = curve.loc[curve["mean_quality"].ge(required)]
        minimum_cost_for_retention[f"{float(target):.4f}"] = (
            None if eligible.empty else float(eligible["relative_cost"].min())
        )
    summary = {
        "quality_cost_nauc": nauc,
        "always_low_quality": low_quality,
        "always_high_quality": high_quality,
        "oracle_quality": oracle_quality,
        "oracle_gap": float(oracle_quality - max(ordered_quality)),
        "quality_at_relative_cost_budget": budget_quality,
        "minimum_relative_cost_for_quality_retention": minimum_cost_for_retention,
    }
    return summary, budget_penalties


def predictions_frame(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    method: str,
    penalty: float,
) -> pd.DataFrame:
    decision_cost = frame.loc[:, [f"decision_cost_{tier}" for tier in TIERS]].to_numpy(float)
    quality = frame.loc[:, [f"quality_{tier}" for tier in TIERS]].to_numpy(float)
    actual_cost = frame.loc[:, [f"actual_cost_{tier}" for tier in TIERS]].to_numpy(float)
    actions = route_actions(method, scores, decision_cost, penalty)
    output = frame.loc[:, ["sample_id", "source_dataset", "category", "difficulty_label"]].copy()
    output["selected_tier"] = [TIERS[index] for index in actions]
    output["selected_quality"] = _selected(quality, actions)
    output["selected_actual_cost"] = _selected(actual_cost, actions)
    output["penalty"] = float(penalty)
    for index, tier in enumerate(TIERS):
        output[f"score_{tier}"] = scores[:, index]
    return output


def aggregate_seed_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ObjectiveEvaluationError("At least one seed summary is required")
    values = np.array([float(item["quality_cost_nauc"]) for item in summaries])
    return {
        "seed_count": int(len(values)),
        "quality_cost_nauc_mean": float(values.mean()),
        "quality_cost_nauc_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "quality_cost_nauc_min": float(values.min()),
    }
