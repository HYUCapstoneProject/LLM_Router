"""F6: training-free score update on top of the frozen operating router 1.0.

The ModernBERT weights never change. After each window only four numbers are
re-fitted: two weights on the frozen logit gaps and two ordered thresholds of a
cumulative-logit (ordinal) model of the tier a request needs. This is the
"score table" family of the report's section 2.3 (Eagle, UniRoute) adapted to
a three-tier difficulty router.

Only what the service can observe is used. A request routed to tier t that
ends without escalation says "needed <= t" (no information when t is high);
an escalation that stops at tier r says "needed == r". Requests that fail
everywhere look, to the service, like an escalation that stopped at high.
Because the logged routing is a deterministic function of the same logits the
model conditions on, the censoring is ignorable and plain maximum likelihood
over all requests (not only escalations) is valid.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit

from experiment import (FROZEN, LABELS, file_pin, label_metrics, paired_bootstrap, read, route_metrics,
                        sha, simulate)

ARM = "f6_training_free"
ESCALATION_ONLY = "f6_escalation_only"
LOGIT_COLUMNS = [f"logit_{name}" for name in LABELS]
EPS = 1e-12


def features(frame):
    logits = frame[LOGIT_COLUMNS].to_numpy(float)
    return np.column_stack([logits[:, 1] - logits[:, 0], logits[:, 2] - logits[:, 1]])


def observations(sim):
    """Service-observable evidence about the needed tier, per routed request.

    Returns (lower, upper): needed tier lies in [lower, upper]. Rows with
    [0, 2] carry no information and are dropped by the caller.
    """
    initial = sim.initial_tier.to_numpy(int)
    resolved = sim.resolved_tier.to_numpy(int)
    stopped = np.where(resolved < 0, 2, resolved)
    escalated = stopped > initial
    lower = np.where(escalated, stopped, 0)
    upper = np.where(escalated, stopped, initial)
    return lower, upper


def unpack(params):
    w = params[:2]
    theta0 = params[2]
    theta1 = theta0 + math.exp(params[3])
    return w, theta0, theta1


def cumulative(params, x):
    w, theta0, theta1 = unpack(params)
    eta = x @ w
    return expit(theta0 - eta), expit(theta1 - eta)


def negative_log_likelihood(params, x, lower, upper):
    w, theta0, theta1 = unpack(params)
    eta = x @ w
    log_f0, log_f1 = log_expit(theta0 - eta), log_expit(theta1 - eta)
    f0, f1 = np.exp(log_f0), np.exp(log_f1)
    total = 0.0
    # needed <= 0 / needed <= 1
    total += log_f0[(lower == 0) & (upper == 0)].sum()
    total += log_f1[(lower == 0) & (upper == 1)].sum()
    # needed == 1 / needed == 2
    mid = (lower == 1) & (upper == 1)
    total += np.log(np.clip(f1[mid] - f0[mid], EPS, None)).sum()
    total += log_expit(-(theta1 - eta[(lower == 2) & (upper == 2)])).sum()
    return -total


def fit(x, lower, upper):
    keep = ~((lower == 0) & (upper == 2))
    x, lower, upper = x[keep], lower[keep], upper[keep]
    start = np.array([1.0, 1.0, 0.0, 0.0])
    result = minimize(negative_log_likelihood, start, args=(x, lower, upper), method="L-BFGS-B")
    if not np.isfinite(result.x).all():
        raise RuntimeError(f"Ordinal fit diverged: {result.message}")
    return result.x, {"rows": int(keep.sum()), "nll": float(result.fun), "iterations": int(result.nit),
                      "converged": bool(result.success),
                      "exact_counts": {name: int(((lower == i) & (upper == i)).sum()) for i, name in enumerate(LABELS)},
                      "at_most_easy": int(((lower == 0) & (upper == 0)).sum()),
                      "at_most_medium": int(((lower == 0) & (upper == 1)).sum())}


def need_probabilities(params, x):
    f0, f1 = cumulative(params, x)
    return np.column_stack([f0, f1 - f0, 1.0 - f1])


def decide(params, x):
    """Same semantics as production R0: route to the most probable needed tier."""
    return need_probabilities(params, x).argmax(1)


def params_record(params):
    w, theta0, theta1 = unpack(params)
    return {"w_medium_minus_easy": float(w[0]), "w_hard_minus_medium": float(w[1]),
            "theta_easy": float(theta0), "theta_medium": float(theta1)}


def run_stream(stream, windows, mode):
    """Prequential replay. Window j+1 is routed with parameters fitted on windows 1..j.

    mode='fixed_logger': evidence always comes from operating 1.0's routing (as F1-F3).
    mode='on_policy': evidence comes from F6's own routing of each window.
    mode='escalation_only': fixed logger, but only escalation events are used.
    """
    frozen = stream[LOGIT_COLUMNS].to_numpy(float).argmax(1)
    x_all = features(stream)
    decisions = np.full(len(stream), -1)
    xs, lows, ups, fits = [], [], [], []
    params = None
    for window in range(1, windows + 1):
        mask = stream.window.eq(window).to_numpy()
        part = stream[mask]
        routed = frozen[mask] if params is None else decide(params, x_all[mask])
        decisions[mask] = routed
        logged = frozen[mask] if mode in ("fixed_logger", "escalation_only") else routed
        lower, upper = observations(simulate(part, logged))
        if mode == "escalation_only":
            event = (lower == upper) & (lower > logged)
            lower, upper = lower[event], upper[event]
            x_window = x_all[mask][event]
        else:
            x_window = x_all[mask]
        xs.append(x_window)
        lows.append(lower)
        ups.append(upper)
        if window < windows:
            params, info = fit(np.concatenate(xs), np.concatenate(lows), np.concatenate(ups))
            fits.append({"after_window": window, **params_record(params), **info})
    return decisions, params, fits


def needed_tier(frame):
    success = frame[[f"success_{t}" for t in ("low", "medium", "high")]].to_numpy(float) == 1
    first = np.where(success.any(1), success.argmax(1), 2)
    return first


def routing_error(frame, predicted):
    need = needed_tier(frame)
    p = np.asarray(predicted)
    return {"under_needed_rate": float((p < need).mean()), "over_needed_rate": float((p > need).mean()),
            "exact_needed_rate": float((p == need).mean()), "two_step_under_rate": float((need - p >= 2).mean())}


def metrics(frame, predicted):
    return {**label_metrics(frame, predicted), **route_metrics(frame, predicted), **routing_error(frame, predicted)}


def load_run(run_dir):
    run = Path(run_dir)
    plan = read(run / "plan.json")
    file_pin(run / "plan.json", read(run / "plan.sha256.json")["sha256"])
    stream_path = run / "shared-data" / "stream-windows.parquet"
    file_pin(stream_path, plan["data_sha256"]["stream-windows"])
    stream = pd.read_parquet(stream_path)
    finals = {}
    for role in ("gate", "forget"):
        directory = run / "scores" / FROZEN / "seed-0" / "final" / role
        receipt = read(directory / "receipt.json")
        if receipt["plan_sha256"] != sha(run / "plan.json"):
            raise ValueError("F0 final scores belong to another plan")
        file_pin(directory / "predictions.parquet", receipt["predictions_sha256"])
        finals[role] = pd.read_parquet(directory / "predictions.parquet")
    return plan, stream, finals


def evaluate(plan, stream, finals):
    cfg = plan["config"]
    windows = cfg["windows"]
    frozen = stream[LOGIT_COLUMNS].to_numpy(float).argmax(1)
    if not np.array_equal(frozen, stream.initial_label.map(LABELS.index).to_numpy()):
        raise ValueError("Stored logits do not reproduce the sealed logger routing")
    arms = {FROZEN: (frozen, None, [])}
    for name, mode in ((ARM, "fixed_logger"), (f"{ARM}_on_policy", "on_policy"), (ESCALATION_ONLY, "escalation_only")):
        arms[name] = run_stream(stream, windows, mode)
    later = stream.window.ge(2).to_numpy()
    base = stream[later].reset_index(drop=True)
    prequential, per_window, pooled = {}, [], base[["sample_id", "source_dataset", "duplicate_group_id"]].copy()
    for name, (decisions, _, _) in arms.items():
        prequential[name] = metrics(base, decisions[later])
        sim = simulate(base, decisions[later])
        pooled[f"{name}_quality"] = sim.initial_quality.to_numpy()
        pooled[f"{name}_cost"] = sim.simulated_cost.to_numpy()
        pooled[f"{name}_depth"] = sim.call_depth.to_numpy()
        for window in range(1, windows + 1):
            mask = stream.window.eq(window).to_numpy()
            per_window.append({"arm": name, "window": window, **metrics(stream[mask], decisions[mask])})
    contrasts = []
    for name in arms:
        if name == FROZEN:
            continue
        for metric in ("quality", "cost", "depth"):
            value, low, high = paired_bootstrap(pooled, f"{FROZEN}_{metric}", f"{name}_{metric}",
                                                cfg["bootstrap_draws"], cfg["bootstrap_seed"])
            contrasts.append({"contrast": f"{name}_minus_{FROZEN}", "metric": metric, "difference": value,
                              "ci95_low": low, "ci95_high": high})
    final = {}
    for name, (decisions, params, _) in arms.items():
        for role, frame in finals.items():
            if not np.array_equal(frame[LOGIT_COLUMNS].to_numpy(float).argmax(1), frame.prediction.map(LABELS.index).to_numpy()):
                raise ValueError(f"{role} logits do not reproduce F0 predictions")
            predicted = frame.prediction.map(LABELS.index).to_numpy() if params is None else decide(params, features(frame))
            final[f"{name}/{role}"] = metrics(frame, predicted)
    judgments = {}
    for name in arms:
        if name == FROZEN:
            continue
        q = next(c for c in contrasts if c["contrast"] == f"{name}_minus_{FROZEN}" and c["metric"] == "quality")
        c = next(c for c in contrasts if c["contrast"] == f"{name}_minus_{FROZEN}" and c["metric"] == "cost")
        drop = final[f"{FROZEN}/forget"]["source_macro_f1"] - final[f"{name}/forget"]["source_macro_f1"]
        judgments[name] = {"quality_minus_f0": q["difference"], "effect": q["difference"] > cfg["minimum_improvement"],
                           "cost_minus_f0": c["difference"], "cost_increase": c["difference"] > 0,
                           "retention_drop": drop, "retention_within_tolerance": drop <= cfg["maximum_forgetting_drop"]}
    return {"prequential_windows_2_6": prequential, "per_window": per_window, "paired": contrasts, "final": final,
            "fits": {name: fits for name, (_, _, fits) in arms.items() if fits}, "judgments": judgments,
            "criteria": {"minimum_improvement": cfg["minimum_improvement"],
                         "maximum_forgetting_drop": cfg["maximum_forgetting_drop"]},
            "production_approved": False}


def write_report(result, out_dir, source_run):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps({"source_run": str(source_run), **result}, indent=2, ensure_ascii=False) + "\n")
    pd.DataFrame(result["per_window"]).to_csv(out / "per-window.csv", index=False)
    keys = ["predicted_easy_share", "predicted_hard_share", "initial_quality", "first_try_rate", "resolved_rate",
            "mean_call_depth", "mean_cost", "source_macro_f1", "under_needed_rate", "over_needed_rate", "exact_needed_rate"]
    summary = pd.DataFrame(result["prequential_windows_2_6"]).loc[keys]
    final = pd.DataFrame(result["final"]).loc[["source_macro_f1", "initial_quality", "mean_cost", "resolved_rate"]]
    text = ("# F6 학습 없이 갱신 (training-free score update) 결과\n\n"
            "운영 1.0의 가중치는 고정하고, 고정 logit 차이 2개에 대한 가중치와 순서형 문턱 2개만 피드백으로 다시 맞췄다. "
            "서비스에서 관측 가능한 신호(상향 없이 끝난 요청 = 그 단계 이하로 충분, 상향이 멈춘 단계 = 필요 단계)만 썼다.\n\n"
            "## Prequential (window 2~6)\n\n" + "```\n" + summary.round(4).to_string() + "\n```" + "\n\n"
            "## F0 대비 paired 비교 (질문 그룹 bootstrap 95%)\n\n" + "```\n" + pd.DataFrame(result["paired"]).round(5).to_string(index=False) + "\n```" + "\n\n"
            "## 최종 파라미터로 gate·retention 평가\n\n" + "```\n" + final.round(4).to_string() + "\n```" + "\n\n"
            "## 판정 (계획서 8절 기준)\n\n```json\n" + json.dumps(result["judgments"], indent=2) + "\n```\n\n"
            "## window별 적합 파라미터\n\n```json\n" + json.dumps(result["fits"], indent=2) + "\n```\n\n"
            "운영 설정·CURRENT는 이 결과로 변경하지 않는다.\n")
    (out / "REPORT.md").write_text(text, encoding="utf-8")
    return out
