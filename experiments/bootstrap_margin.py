#!/usr/bin/env python3
"""우리 라우터의 프런티어 대비 마진에 신뢰구간을 붙인다.

파레토에서 나온 +0.0119 가 실제 우위인지 표본 잡음인지 답하려면 구간이 필요하다.
점추정만 보고하면 "그게 유의미한 차이냐"는 질문에 답할 수 없다.

질의를 복원추출로 다시 뽑아 프런티어와 마진을 통째로 다시 계산한다.
프런티어 자체도 같은 데이터에서 추정되므로 함께 흔들어야 한다 —
프런티어를 고정한 채 우리 점만 흔들면 구간이 실제보다 좁게 나온다.

    python experiments/bootstrap_margin.py --iterations 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pareto_baselines import (  # noqa: E402
    MODELS, load, pareto_frontier, upper_convex_hull, frontier_performance_at,
)


def gap_for(perf: np.ndarray, cost: np.ndarray,
            fixed_perf: np.ndarray, fixed_cost: np.ndarray) -> float:
    """이번 표본에서의 (우리 성능 − 같은 비용에서 혼합이 내는 성능)."""
    points = [
        {"policy": MODELS[i], "performance": float(fixed_perf[i]),
         "cost_per_query": float(fixed_cost[i])}
        for i in range(len(MODELS))
    ]
    hull = upper_convex_hull(pareto_frontier(points))
    mixed = frontier_performance_at(hull, float(cost.mean()))
    if mixed is None:
        return float("nan")
    return float(perf.mean()) - mixed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path(".venv-experiments/data/routerbench_0shot.pkl"))
    parser.add_argument("--predictions", type=Path,
                        default=Path("experiments/results/pareto/router_predictions_test.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("experiments/results/pareto/bootstrap.json"))
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260830)
    # pareto_baselines 가 validation 에서 고른 사다리와 같아야 한다.
    parser.add_argument("--tier-easy", default="zero-one-ai/Yi-34B-Chat")
    parser.add_argument("--tier-medium", default="zero-one-ai/Yi-34B-Chat")
    parser.add_argument("--tier-hard", default="gpt-4-1106-preview")
    args = parser.parse_args()

    df = load(args.data).reset_index(drop=True)
    preds = pd.read_csv(args.predictions)
    df = df[df.sample_id.isin(set(preds.sample_id))].reset_index(drop=True)
    preds = preds.set_index("sample_id").loc[df.sample_id].reset_index()
    n = len(df)

    ladder = {"easy": args.tier_easy, "medium": args.tier_medium, "hard": args.tier_hard}

    # 질의당 성능·비용을 정책별로 미리 배열로 만든다. 부트스트랩은 인덱스만 다시 뽑는다.
    fixed_perf = np.array([df[m].to_numpy(dtype=float) for m in MODELS])
    fixed_cost = np.array([df[f"{m}|total_cost"].to_numpy(dtype=float) for m in MODELS])

    chosen = preds.predicted_tier.map(ladder).to_numpy()
    model_index = {m: i for i, m in enumerate(MODELS)}
    rows = np.arange(n)
    cols = np.array([model_index[m] for m in chosen])
    ours_perf = fixed_perf[cols, rows]
    ours_cost = fixed_cost[cols, rows]

    observed = gap_for(ours_perf, ours_cost, fixed_perf.mean(axis=1), fixed_cost.mean(axis=1))
    print(f"질의 {n:,}개 · 부트스트랩 {args.iterations:,}회")
    print(f"관측된 마진: {observed:+.4f}\n")

    rng = np.random.default_rng(args.seed)
    gaps = np.empty(args.iterations)
    for i in range(args.iterations):
        idx = rng.integers(0, n, size=n)
        gaps[i] = gap_for(ours_perf[idx], ours_cost[idx],
                          fixed_perf[:, idx].mean(axis=1), fixed_cost[:, idx].mean(axis=1))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{args.iterations}")

    gaps = gaps[~np.isnan(gaps)]
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    positive = float((gaps > 0).mean())

    print(f"\n95% 신뢰구간 : [{lo:+.4f}, {hi:+.4f}]")
    print(f"평균         : {gaps.mean():+.4f}")
    print(f"마진 > 0 비율 : {positive:.1%}")
    print()
    if lo > 0:
        print("구간이 0 을 넘지 않는다 — 프런티어 대비 우위가 표본 잡음으로 설명되지 않는다.")
    else:
        print("구간이 0 을 포함한다 — 우위를 통계적으로 주장할 수 없다.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "queries": n,
        "iterations": int(len(gaps)),
        "tier_ladder": ladder,
        "observed_gap": observed,
        "bootstrap_mean": float(gaps.mean()),
        "ci95": [float(lo), float(hi)],
        "share_positive": positive,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
