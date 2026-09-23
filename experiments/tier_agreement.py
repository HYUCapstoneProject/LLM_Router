#!/usr/bin/env python3
"""11개 모델을 3티어로 묶어 질문 난이도를 만들고, 우리 라우터 예측과의 일치율을 잰다.

파레토(`pareto_baselines.py`)는 "기존 기법 대비 값어치가 있나"에 답한다.
이 스크립트는 다른 질문에 답한다 — "우리 분류기가 티어를 제대로 고르나".

동기: 파레토는 티어를 RouterBench 모델에 배정하는 사다리가 필요한데,
그 선택이 자유도가 된다. 배정 없이 분류 품질만 직접 재면 자유도가 사라질 것
같지만 **사라지지 않는다.** 대신 두 개가 새로 생긴다.

    (1) 11개 모델을 어떻게 3그룹으로 묶는가
    (2) "이 티어가 이 문제를 푼다"를 어떻게 판정하는가

그래서 이 스크립트는 하나만 고르지 않고 **그룹 4가지 × 규칙 3가지를 전부**
돌려 민감도를 함께 보고한다. 하나만 고르면 그 선택이 곧 결론이 된다.

    python experiments/tier_agreement.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pareto_baselines import MODELS, load  # noqa: E402

TIER = {"easy": 0, "medium": 1, "hard": 2}


def split_groups(order: np.ndarray, sizes: tuple[int, ...]) -> list[list[int]]:
    """모델 인덱스를 정렬 순서대로 잘라 3그룹으로 만든다."""
    out, i = [], 0
    for s in sizes:
        out.append(list(order[i:i + s]))
        i += s
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path(".venv-experiments/data/routerbench_0shot.pkl"))
    parser.add_argument("--predictions", type=Path,
                        default=Path("experiments/results/pareto/router_predictions_test.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("experiments/results/tier-agreement"))
    args = parser.parse_args()

    raw = load(args.data)
    preds = pd.read_csv(args.predictions)
    df = raw[raw.sample_id.isin(set(preds.sample_id))].reset_index(drop=True)
    preds = preds.set_index("sample_id").loc[df.sample_id].reset_index()

    perf = np.array([df[m].to_numpy(dtype=float) for m in MODELS]).T
    accuracy = perf.mean(axis=0)
    cost = np.array([df[f"{m}|total_cost"].mean() for m in MODELS])
    predicted = np.array([TIER[t] for t in preds.predicted_tier])

    groupings = {
        "성능순 4/4/3": split_groups(np.argsort(accuracy), (4, 4, 3)),
        "성능순 3/4/4": split_groups(np.argsort(accuracy), (3, 4, 4)),
        "비용순 4/4/3": split_groups(np.argsort(cost), (4, 4, 3)),
        "비용순 3/5/3": split_groups(np.argsort(cost), (3, 5, 3)),
    }
    # 판정 규칙. 이름이 아니라 정의가 결론을 바꾼다 — 아래 결과를 볼 것.
    rules = {
        "any": lambda tier: (perf[:, tier] >= 1.0).any(axis=1),
        "majority": lambda tier: (perf[:, tier] >= 1.0).mean(axis=1) > 0.5,
        "mean>=0.5": lambda tier: perf[:, tier].mean(axis=1) >= 0.5,
    }

    rows = []
    for gname, groups in groupings.items():
        for rname, rule in rules.items():
            passes = np.stack([rule(t) for t in groups], axis=1)
            solvable = passes.any(axis=1)
            # 정답 티어 = 통과하는 가장 낮은 티어. 아무도 못 풀면 채점에서 제외한다.
            truth = np.where(solvable, passes.argmax(axis=1), -1)
            m = solvable
            rows.append({
                "grouping": gname,
                "rule": rname,
                "scored": int(m.sum()),
                "unsolvable": int((~solvable).sum()),
                "gt_easy": int((truth[m] == 0).sum()),
                "gt_medium": int((truth[m] == 1).sum()),
                "gt_hard": int((truth[m] == 2).sum()),
                "agreement": float((predicted[m] == truth[m]).mean()),
                "over_routing": float((predicted[m] > truth[m]).mean()),
                "under_routing": float((predicted[m] < truth[m]).mean()),
            })

    out = pd.DataFrame(rows)
    print(f"질의 {len(df):,}건 · 조합 {len(out)}가지\n")
    print(out.to_string(index=False,
                        formatters={"agreement": "{:.1%}".format,
                                    "over_routing": "{:.1%}".format,
                                    "under_routing": "{:.1%}".format}))

    lo, hi = out.agreement.min(), out.agreement.max()
    label_agreement = float((preds.predicted_tier == preds.true_label).mean())
    print(f"\n일치율 범위 {lo:.1%} ~ {hi:.1%}  (폭 {hi-lo:.1%}p)")
    print(f"3-class 무작위 기대치 33.3% — 최저({lo:.1%})는 이보다 낮다")
    print(f"difficulty_label 과의 일치율 {label_agreement:.1%}")
    print("\n편향 방향이 규칙에 따라 뒤집힌다:")
    for rname in rules:
        sub = out[out.rule == rname]
        print(f"  {rname:10} 과대 {sub.over_routing.mean():.1%}  과소 {sub.under_routing.mean():.1%}")

    args.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / "results.csv", index=False, encoding="utf-8")
    print(f"\n저장: {args.out}/results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
