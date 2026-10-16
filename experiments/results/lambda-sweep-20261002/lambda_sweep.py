"""Dense λ sweep over stored router scores (no retraining, CPU only).

λ is chosen per source on a tuning split and then applied unchanged to the
evaluation split(s). Selection rule and relative-cost definition follow
ml/src/llm_router_labeling/objective_comparison/evaluation.py.

Usage: python lambda_sweep.py --a-dir <A comparison-results> --a-test-dir <sealed inference>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TIERS = ("low", "medium", "high")
SOURCES = ("routerbench", "xroutebench")
LAMBDAS = np.concatenate([[0.0], np.logspace(-4, 2, 601)])
BUDGET = 0.5
RETENTION = (0.95, 0.98)
G1 = REPO / "experiments/results/routing-generalization-v1-20260925"
FROZEN = REPO / "ml/artifacts/frozen-evaluation-outcomes/continual-v1-20260924-28cbc342c510"
METHOD_LABEL = {"ordinal": "R0", "pointwise": "R1", "pairwise": "R2", "regret": "R3"}


def transform(method: str, raw: np.ndarray) -> np.ndarray:
    if method == "pointwise":
        return 1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))
    if method == "ordinal":
        shifted = np.exp(raw - raw.max(axis=1, keepdims=True))
        return shifted / shifted.sum(axis=1, keepdims=True)
    return raw


def matrices(frame: pd.DataFrame, method: str) -> dict[str, np.ndarray]:
    get = lambda p: frame[[f"{p}_{t}" for t in TIERS]].to_numpy(float)  # noqa: E731
    return {
        "score": transform(method, get("score")),
        "q": get("quality"),
        "c": get("actual_cost"),
        "d": get("decision_cost"),
    }


def curve(m: dict[str, np.ndarray], lambdas=LAMBDAS) -> pd.DataFrame:
    """Evaluate every λ on one source slice."""
    q, c = m["q"], m["c"]
    low_c, high_c = c[:, 0].mean(), c[:, 2].mean()
    idx = np.arange(len(q))
    rows = []
    for lam in lambdas:
        a = np.argmax(m["score"] - lam * m["d"], axis=1)
        sel_c = c[idx, a].mean()
        rows.append(
            {
                "lambda": float(lam),
                "quality": float(q[idx, a].mean()),
                "relative_cost": float((sel_c - low_c) / (high_c - low_c)),
                "mean_actual_cost": float(sel_c),
                **{f"{t}_rate": float((a == i).mean()) for i, t in enumerate(TIERS)},
            }
        )
    return pd.DataFrame(rows)


def reference(m: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "always_low_quality": float(m["q"][:, 0].mean()),
        "always_medium_quality": float(m["q"][:, 1].mean()),
        "always_high_quality": float(m["q"][:, 2].mean()),
        "always_medium_relative_cost": float(
            (m["c"][:, 1].mean() - m["c"][:, 0].mean())
            / (m["c"][:, 2].mean() - m["c"][:, 0].mean())
        ),
        "oracle_quality": float(m["q"].max(axis=1).mean()),
        "rows": int(len(m["q"])),
    }


def choose(tune: pd.DataFrame, ref: dict[str, float]) -> dict[str, dict]:
    """Criteria are evaluated on the tuning split only. Ties → smaller cost, smaller λ."""
    out: dict[str, dict] = {"lambda0": {"lambda": 0.0, "feasible_on_tuning": True}}
    ok = tune[tune.relative_cost <= BUDGET + 1e-12]
    pick = ok.sort_values(["quality", "relative_cost", "lambda"], ascending=[False, True, True])
    out[f"budget{BUDGET:g}"] = (
        {"lambda": float(pick.iloc[0]["lambda"]), "feasible_on_tuning": True}
        if len(pick)
        else {"lambda": None, "feasible_on_tuning": False}
    )
    for target in RETENTION:
        need = ref["always_high_quality"] * target
        ok = tune[tune.quality >= need - 1e-12]
        pick = ok.sort_values(["relative_cost", "quality", "lambda"], ascending=[True, False, False])
        out[f"retain{int(target * 100)}"] = (
            {"lambda": float(pick.iloc[0]["lambda"]), "feasible_on_tuning": True,
             "required_quality": need}
            if len(pick)
            else {"lambda": None, "feasible_on_tuning": False, "required_quality": need}
        )
    return out


def point(m: dict[str, np.ndarray], lam: float) -> dict[str, float]:
    return curve(m, [lam]).iloc[0].to_dict()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runs(a_dir: Path, a_test: Path) -> list[dict]:
    plan = []
    for method in ("ordinal", "pointwise", "pairwise"):
        plan.append(
            {
                "experiment": "A",
                "arm": METHOD_LABEL[method],
                "method": method,
                "tune": ("D_dev", [a_dir / method / "validation-scores.parquet"]),
                "evals": [
                    ("test", [a_test / method / f"test_{s}-scores.parquet" for s in SOURCES]),
                ],
            }
        )
    for arm in ("r1-natural", "r1-balanced", "r3-natural", "r3-balanced"):
        method = "pointwise" if arm.startswith("r1") else "regret"
        runs_dir = G1 / "merged-runs" / arm
        plan.append(
            {
                "experiment": "G1",
                "arm": arm.upper(),
                "method": method,
                "tune": ("calibration", [runs_dir / "calibration-scores.parquet"]),
                "evals": [
                    ("selection", [runs_dir / "selection-scores.parquet"]),
                    ("task_holdout", [G1 / "holdout-scores" / f"{arm}-scores.parquet"]),
                ],
            }
        )
    return plan


def load(paths: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def macro_summary(points: pd.DataFrame) -> pd.DataFrame:
    """Source-macro view on eval splits; NaN unless every source has a feasible λ."""
    ev = points[points.role.eq("eval")]
    keys = ["experiment", "arm", "split", "criterion"]
    rows = []
    for key, g in ev.groupby(keys, sort=False):
        complete = g["lambda"].notna().all()
        row = dict(zip(keys, key))
        row["feasible_sources"] = int(g["lambda"].notna().sum())
        for col in ("quality", "relative_cost", "low_rate", "medium_rate", "high_rate"):
            row[col] = float(g[f"at_{col}"].mean()) if complete else None
        row.update({f"lambda_{s}": v for s, v in zip(g.source, g["lambda"])})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", type=Path, required=True)
    ap.add_argument("--a-test-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()

    curves, points, inputs = [], [], {}
    for run in runs(args.a_dir, args.a_test_dir):
        split_name, paths = run["tune"]
        frames = {split_name: load(paths)}
        for name, eval_paths in run["evals"]:
            frames[name] = load(eval_paths)
            paths = paths + eval_paths
        for p in paths:
            inputs[str(p.relative_to(REPO) if p.is_relative_to(REPO) else p)] = sha(p)
        for source in SOURCES:
            per = {
                name: matrices(f[f.source_dataset.eq(source)], run["method"])
                for name, f in frames.items()
            }
            refs = {name: reference(m) for name, m in per.items()}
            tune_curve = curve(per[split_name])
            chosen = choose(tune_curve, refs[split_name])
            for name, m in per.items():
                c = tune_curve if name == split_name else curve(m)
                c = c.assign(experiment=run["experiment"], arm=run["arm"], method=run["method"],
                             source=source, split=name, role="tune" if name == split_name else "eval")
                curves.append(c)
                for criterion, sel in chosen.items():
                    row = {"experiment": run["experiment"], "arm": run["arm"], "method": run["method"],
                           "source": source, "split": name,
                           "role": "tune" if name == split_name else "eval",
                           "criterion": criterion, **sel, **refs[name]}
                    if sel["lambda"] is not None:
                        row.update({f"at_{k}": v for k, v in point(m, sel["lambda"]).items() if k != "lambda"})
                    points.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    all_curves = pd.concat(curves, ignore_index=True)
    all_points = pd.DataFrame(points)
    all_curves.to_csv(args.out / "curves.csv", index=False)
    all_points.to_csv(args.out / "chosen-points.csv", index=False)
    macro_summary(all_points).to_csv(args.out / "macro-summary.csv", index=False)
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "lambda_grid": {"zero": True, "logspace_min": 1e-4, "logspace_max": 100, "points": len(LAMBDAS)},
                "budget": BUDGET,
                "retention_targets": list(RETENTION),
                "lambda_scope": "per source, chosen on tuning split only",
                "tuning_split": {"A": "D_dev (validation)", "G1": "calibration"},
                "eval_splits": {"A": ["test"], "G1": ["selection", "task_holdout"]},
                "a_inputs_origin": {
                    "validation": "gs://llm-router-mlserver/training-runs/objectives-continual-v1-A-20260924/comparison-results.tar",
                    "test": "gs://llm-router-mlserver/training-runs/objectives-continual-v1-A-20260924/sealed-tests-v1/sealed-inference.tar",
                },
                "input_sha256": inputs,
                "existing_locks_modified": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(all_points.shape, all_curves.shape)


if __name__ == "__main__":
    main()
