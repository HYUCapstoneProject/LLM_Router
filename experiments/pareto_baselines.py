#!/usr/bin/env python3
"""RouterBench 로 라우팅 정책들의 비용-성능 파레토를 오프라인 계산한다.

2026-08-03 미팅에서 "토큰을 아끼는 기법이 이미 여럿인데 그 대비 성능을 비교하라"는
지적이 있었다. Always-Opus 대비 절감률만으로는 "우리 라우터가 좋다"가 아니라
"싼 모델이 싸다"를 보여줄 뿐이다.

RouterBench 에는 11개 모델이 36,497개 질의에 대해 낸 성능과 비용이 이미 들어 있다.
따라서 **LLM 을 한 번도 호출하지 않고** 정책 함수만 갈아끼워 전부 계산할 수 있다.

가장 중요한 기준선은 개별 모델이 아니라 **비적응 프런티어**다. 두 고정 모델을
확률적으로 섞기만 해도 비용-성능 곡선이 공짜로 나온다. 라우터는 그 선 위로
올라가야만 값어치가 있다. 개별 모델만 찍어놓고 비교하면 이 사실이 가려진다.

사용법:
    python experiments/pareto_baselines.py --data <routerbench_0shot.pkl>
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Callable

import pandas as pd

MODELS = [
    "mistralai/mistral-7b-chat",
    "WizardLM/WizardLM-13B-V1.2",
    "mistralai/mixtral-8x7b-chat",
    "meta/code-llama-instruct-34b-chat",
    "zero-one-ai/Yi-34B-Chat",
    "meta/llama-2-70b-chat",
    "claude-instant-v1",
    "gpt-3.5-turbo-1106",
    "claude-v1",
    "claude-v2",
    "gpt-4-1106-preview",
]

# 정책 = (DataFrame) -> 행마다 고를 모델 이름 Series
Policy = Callable[[pd.DataFrame], pd.Series]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_pickle(path)
    missing = [m for m in MODELS if m not in df.columns or f"{m}|total_cost" not in df.columns]
    if missing:
        raise SystemExit(f"데이터에 없는 모델: {missing}")
    # 성능은 0/1 이 아니라 0~1 연속값이다(mtbench 등은 부분점수를 준다).
    for m in MODELS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    return df


def score(df: pd.DataFrame, chosen: pd.Series) -> dict[str, float]:
    """고른 모델 기준의 질의당 평균 비용과 평균 성능."""
    idx = pd.RangeIndex(len(df))
    perf = pd.Series(
        [df.at[i, chosen.iat[i]] for i in idx], index=idx, dtype="float64"
    )
    cost = pd.Series(
        [df.at[i, f"{chosen.iat[i]}|total_cost"] for i in idx], index=idx, dtype="float64"
    )
    return {
        "performance": float(perf.mean()),
        "cost_per_query": float(cost.mean()),
        "total_cost": float(cost.sum()),
    }


# --- 정책들 ---------------------------------------------------------------


def always(model: str) -> Policy:
    return lambda df: pd.Series([model] * len(df))


def random_uniform(seed: int) -> Policy:
    """무작위 라우팅. 분류기가 배운 게 있는지 증명하는 최소 조건이다.

    Haiku 가 Opus 보다 싸므로 아무렇게나 보내도 비용은 내려간다. 우리 라우터가
    이 점을 못 이기면 학습이 아무것도 하지 않았다는 뜻이다.
    """
    def policy(df: pd.DataFrame) -> pd.Series:
        rng = random.Random(seed)
        return pd.Series([rng.choice(MODELS) for _ in range(len(df))])
    return policy


def oracle() -> Policy:
    """정답을 맞히는 모델 중 가장 싼 것. RouterBench §4.4 의 상한선이다.

    아무 모델도 못 맞히면 그중 가장 싼 것을 고른다 — 어차피 틀릴 거라면
    비용이라도 아끼는 것이 상한선의 정의에 맞다.
    """
    def policy(df: pd.DataFrame) -> pd.Series:
        chosen = []
        for i in range(len(df)):
            best, best_cost = None, None
            cheapest, cheapest_cost = None, None
            for m in MODELS:
                cost = df.at[i, f"{m}|total_cost"]
                if cheapest_cost is None or cost < cheapest_cost:
                    cheapest, cheapest_cost = m, cost
                if df.at[i, m] >= 1.0 and (best_cost is None or cost < best_cost):
                    best, best_cost = m, cost
            chosen.append(best if best is not None else cheapest)
        return pd.Series(chosen)
    return policy


def rule_based(short: str, medium: str, long: str, short_chars: int, long_chars: int) -> Policy:
    """프롬프트 길이만 보는 if 문. ModernBERT 학습이 값어치를 하는지 재는 기준선이다.

    라우팅에 학습이 필요하다는 주장을 하려면, 학습 없이 얻을 수 있는 것을
    먼저 보여줘야 한다.
    """
    def policy(df: pd.DataFrame) -> pd.Series:
        lengths = df["prompt"].astype(str).str.len()
        return pd.Series(
            [short if n < short_chars else (long if n > long_chars else medium) for n in lengths]
        )
    return policy


def pareto_frontier(points: list[dict]) -> list[dict]:
    """지배당하지 않는 점들. 더 싸면서 성능도 높은 점이 있으면 지배당한 것이다.

    비용 오름차순으로 훑으며 지금까지의 최고 성능을 넘는 점만 남긴다.
    싼 쪽 점은 성능이 낮아도 프런티어다 — 예산이 적을 때의 최선이기 때문이다.
    """
    ordered = sorted(points, key=lambda p: (p["cost_per_query"], -p["performance"]))
    frontier: list[dict] = []
    best = float("-inf")
    for point in ordered:
        if point["performance"] > best:
            frontier.append(point)
            best = point["performance"]
    return frontier


def upper_convex_hull(frontier: list[dict]) -> list[dict]:
    """비적응 프런티어. 두 고정 모델을 확률적으로 섞어 도달 가능한 상단 경계다.

    이 선 아래에 있는 라우터는 존재 이유가 없다. 같은 비용에서 더 나은 성능을
    모델 두 개를 확률적으로 섞는 것만으로 얻을 수 있기 때문이다. 개별 모델 점만
    찍어놓고 비교하면 이 사실이 가려진다.
    """
    hull: list[dict] = []
    for point in frontier:
        # 직전 점이 a→point 직선 아래에 있으면 혼합으로 대체되므로 버린다.
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            slope_ab = ((b["performance"] - a["performance"])
                        / (b["cost_per_query"] - a["cost_per_query"]))
            slope_ap = ((point["performance"] - a["performance"])
                        / (point["cost_per_query"] - a["cost_per_query"]))
            if slope_ap >= slope_ab:
                hull.pop()
            else:
                break
        hull.append(point)
    return hull


def tier_ladder(mapping: dict[str, str], tiers: pd.Series) -> Policy:
    """티어 예측을 RouterBench 모델로 옮긴다.

    운영에서는 easy/medium/hard 를 Haiku/Sonnet/Opus 로 보내지만 RouterBench 에는
    그 모델들이 없다. 그래서 대응하는 3단 사다리를 고른다.

    사다리는 **비적응 프런티어 위의 모델**로 잡는다. 프런티어에서 벗어난 모델을
    쓰면 우리 라우터가 나쁜 모델을 배정받은 채로 좋은 모델들이 만든 프런티어와
    비교되어, 방법이 아니라 모델 선택 때문에 지는 그림이 된다.
    """
    def policy(df: pd.DataFrame) -> pd.Series:
        return pd.Series([mapping[t] for t in tiers])
    return policy


def frontier_performance_at(hull: list[dict], cost: float) -> float | None:
    """주어진 비용에서 고정 모델 혼합이 낼 수 있는 성능.

    프런티어의 두 꼭짓점 사이는 확률적 혼합으로 직선을 그리며 도달할 수 있다.
    라우터의 값어치는 이 직선을 얼마나 넘어서느냐다.
    """
    if not hull:
        return None
    if cost <= hull[0]["cost_per_query"]:
        return hull[0]["performance"] if cost == hull[0]["cost_per_query"] else None
    if cost >= hull[-1]["cost_per_query"]:
        return hull[-1]["performance"]
    for a, b in zip(hull, hull[1:]):
        if a["cost_per_query"] <= cost <= b["cost_per_query"]:
            span = b["cost_per_query"] - a["cost_per_query"]
            ratio = 0.0 if span == 0 else (cost - a["cost_per_query"]) / span
            return a["performance"] + ratio * (b["performance"] - a["performance"])
    return None


def sweep_ladders(df: pd.DataFrame, preds: pd.DataFrame, hull: list[dict],
                  top: int = 8) -> list[dict]:
    """3단 사다리 조합을 전부 돌려 우리 라우터가 가장 잘 서는 배정을 찾는다.

    사다리를 임의로 고르면 결론이 사다리에 좌우된다. gpt-4 는 Yi-34B 보다 17배
    비싼데 성능은 0.14 밖에 안 높아서, hard 를 gpt-4 로 보내는 순간 비용이 터진다.
    그건 라우터의 실패가 아니라 배정의 실패다.

    운영에서도 어차피 답해야 하는 질문이다 — 티어를 어느 Claude 모델에 붙일 것인가.

    ⚠️ 이 탐색을 돌린 split 에서 그대로 결과를 보고하면 안 된다. 조합을 다 보고
    가장 좋은 것을 고르는 건 그 split 에 대한 과적합이다. validation 에서 고르고
    test 에서 보고한다.
    """
    cost_of = {m: df[f"{m}|total_cost"].mean() for m in MODELS}
    ordered = sorted(MODELS, key=lambda m: cost_of[m])

    rows = []
    for easy in ordered:
        for medium in ordered:
            for hard in ordered:
                # 사다리는 비용 오름차순이어야 의미가 있다. 아니면 티어 개념이 무너진다.
                if not (cost_of[easy] <= cost_of[medium] <= cost_of[hard]):
                    continue
                ladder = {"easy": easy, "medium": medium, "hard": hard}
                metrics = score(df, tier_ladder(ladder, preds.predicted_tier)(df))
                mixed = frontier_performance_at(hull, metrics["cost_per_query"])
                if mixed is None:
                    continue
                rows.append({
                    "easy": easy, "medium": medium, "hard": hard,
                    "performance": metrics["performance"],
                    "cost_per_query": metrics["cost_per_query"],
                    "frontier_performance": mixed,
                    "frontier_gap": metrics["performance"] - mixed,
                })
    rows.sort(key=lambda r: r["frontier_gap"], reverse=True)
    return rows[:top]


def plot(results: list[dict], hull: list[dict], out: Path, n_queries: int,
         ladder: dict[str, str]) -> None:
    """비용-성능 파레토 한 장. 발표에서 이 그림이 질문에 대한 답이다.

    라벨은 영어로 둔다. 한글 글꼴이 없는 환경에서 그리면 글자가 전부 두부로
    깨지는데, 그건 조용히 일어나서 발표 직전에야 발견된다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    to_1k = lambda c: c * 1000  # noqa: E731

    hull_x = [to_1k(p["cost_per_query"]) for p in hull]
    hull_y = [p["performance"] for p in hull]
    ax.plot(hull_x, hull_y, "-", color="#888", linewidth=1.5, zorder=1,
            label="Non-adaptive frontier (mixing fixed models)")
    ax.fill_between(hull_x, hull_y, min(hull_y) - 0.05, color="#888", alpha=0.06, zorder=0)

    styles = {
        "fixed": dict(marker="o", color="#bbb", s=40, label="Fixed model"),
        "baseline": dict(marker="s", color="#d1495b", s=90, label="Baseline"),
        "ours": dict(marker="*", color="#2e86ab", s=320, label="Ours (ModernBERT router)"),
        "upper-bound": dict(marker="^", color="#3c9a5f", s=110, label="Upper bound"),
    }
    seen = set()
    for r in results:
        style = dict(styles[r["kind"]])
        label = style.pop("label")
        ax.scatter(to_1k(r["cost_per_query"]), r["performance"], zorder=3,
                   label=None if label in seen else label, **style)
        seen.add(label)
        if r["kind"] != "fixed":
            # ours 와 tier-oracle 은 비용·성능이 가까워 기본 오프셋이면 글자가 겹친다.
            offsets = {"ours:modernbert": (10, -14),
                       "tier-oracle:perfect-label": (-52, 10)}
            ax.annotate(r["policy"].split(":")[0],
                        (to_1k(r["cost_per_query"]), r["performance"]),
                        textcoords="offset points",
                        xytext=offsets.get(r["policy"], (8, 5)), fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Cost per 1,000 queries (USD, log scale)")
    ax.set_ylabel("RouterBench performance")
    ax.set_title(f"Cost-performance Pareto on RouterBench "
                 f"({n_queries:,} held-out queries, {len(MODELS)} models)")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc="lower right", fontsize=9)
    # 어떤 배정으로 낸 점인지 모르면 "ours" 를 해석할 수 없다.
    ladder_note = "\n".join([
        "Tier ladder (chosen on validation):",
        f"  easy → {ladder['easy']}",
        f"  medium → {ladder['medium']}",
        f"  hard → {ladder['hard']}",
    ])
    ax.text(0.02, 0.97, ladder_note,
            transform=ax.transAxes, fontsize=8, va="top", color="#555")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path,
                        default=Path(".venv-experiments/data/routerbench_0shot.pkl"))
    parser.add_argument("--out", type=Path, default=Path("experiments/results/pareto"))
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--predictions", type=Path,
                        default=Path("experiments/results/pareto/router_predictions_test.csv"),
                        help="보고용 test split 예측")
    parser.add_argument("--val-predictions", type=Path,
                        default=Path("experiments/results/pareto/router_predictions_validation.csv"),
                        help="사다리 선택용 validation 예측. 없으면 사다리를 기본값으로 쓴다")
    # 비적응 프런티어 위의 3단 사다리. 프런티어 밖 모델을 쓰면 우리 라우터가
    # 모델 선택 때문에 지는 그림이 되어 비교가 방법 대 방법이 아니게 된다.
    parser.add_argument("--tier-easy", default="WizardLM/WizardLM-13B-V1.2")
    parser.add_argument("--tier-medium", default="zero-one-ai/Yi-34B-Chat")
    parser.add_argument("--tier-hard", default="gpt-4-1106-preview")
    args = parser.parse_args()

    df = load(args.data).reset_index(drop=True)

    ladder = {"easy": args.tier_easy, "medium": args.tier_medium, "hard": args.tier_hard}
    ladder_source = "기본값(인자)"

    # 사다리는 validation 에서 고른다. test 를 보고 고르면 그 숫자는 과적합이다.
    if args.val_predictions.exists():
        val_preds = pd.read_csv(args.val_predictions)
        val_df = df[df.sample_id.isin(set(val_preds.sample_id))].reset_index(drop=True)
        val_preds = val_preds.set_index("sample_id").loc[val_df.sample_id].reset_index()
        val_fixed = [{"policy": f"always:{m}", "kind": "fixed",
                      **score(val_df, always(m)(val_df))} for m in MODELS]
        val_hull = upper_convex_hull(pareto_frontier(val_fixed))
        best = sweep_ladders(val_df, val_preds, val_hull, top=1)[0]
        ladder = {"easy": best["easy"], "medium": best["medium"], "hard": best["hard"]}
        ladder_source = f"validation {len(val_df):,}건에서 선택 (격차 {best['frontier_gap']:+.4f})"

    preds = None
    if args.predictions.exists():
        preds = pd.read_csv(args.predictions)
        # 우리 라우터를 표본에서만 재고 기준선을 전량에서 재면 비교가 성립하지 않는다.
        # 같은 행으로 모두 잘라낸다.
        df = df[df.sample_id.isin(set(preds.sample_id))].reset_index(drop=True)
        preds = preds.set_index("sample_id").loc[df.sample_id].reset_index()
        print(f"라우터 예측 {len(preds):,}건 — 모든 정책을 이 부분집합에서 평가한다")
        print(f"3단 사다리: easy={ladder['easy']}  medium={ladder['medium']}  hard={ladder['hard']}")
        print(f"사다리 출처: {ladder_source}")

    print(f"질의 {len(df):,}개 · 모델 {len(MODELS)}개\n")

    policies: dict[str, tuple[str, Policy]] = {
        **{f"always:{m}": ("fixed", always(m)) for m in MODELS},
        "random": ("baseline", random_uniform(args.seed)),
        "rule-based:length": ("baseline",
                              rule_based("mistralai/mixtral-8x7b-chat", "claude-instant-v1",
                                         "gpt-4-1106-preview", 300, 1200)),
        "oracle": ("upper-bound", oracle()),
    }
    if preds is not None:
        policies["ours:modernbert"] = ("ours", tier_ladder(ladder, preds.predicted_tier))
        # 분류기가 완벽했다면 어디까지 갈 수 있었는지. 3티어 설계 자체의 천장이라
        # 분류 성능과 설계 한계를 분리해서 볼 수 있다.
        policies["tier-oracle:perfect-label"] = ("upper-bound",
                                                 tier_ladder(ladder, preds.true_label))

    results = []
    for name, (kind, policy) in policies.items():
        metrics = score(df, policy(df))
        results.append({"policy": name, "kind": kind, **metrics})
        print(f"  {name:42} perf={metrics['performance']:.4f}  "
              f"${metrics['cost_per_query']*1000:7.4f}/1k질의")

    fixed = [r for r in results if r["kind"] == "fixed"]
    frontier = pareto_frontier(fixed)
    hull = upper_convex_hull(frontier)
    print("\n=== 비적응 프런티어 (고정 모델 혼합으로 도달 가능한 상단) ===")
    for point in hull:
        print(f"  {point['policy']:42} perf={point['performance']:.4f}  "
              f"${point['cost_per_query']*1000:7.4f}/1k질의")

    # 각 정책이 프런티어 위인지 아래인지. 아래면 모델 두 개를 섞는 것만 못하다.
    print("\n=== 프런티어 대비 (같은 비용에서 혼합이 내는 성능과 비교) ===")
    for r in results:
        if r["kind"] == "fixed":
            continue
        mixed = frontier_performance_at(hull, r["cost_per_query"])
        if mixed is None:
            print(f"  {r['policy']:42} 프런티어 비용 범위 밖")
            continue
        gap = r["performance"] - mixed
        verdict = "위" if gap > 0 else "아래"
        print(f"  {r['policy']:42} perf={r['performance']:.4f}  "
              f"혼합={mixed:.4f}  차이={gap:+.4f}  ({verdict})")
        r["frontier_performance"] = mixed
        r["frontier_gap"] = gap

    ladder_sweep: list[dict] = []
    if preds is not None:
        ladder_sweep = sweep_ladders(df, preds, hull)
        print("\n=== 3단 사다리 탐색 (우리 라우터 예측 고정, 배정만 교체) ===")
        print(f"  {'easy':28}{'medium':28}{'hard':24}  {'perf':>7} {'격차':>8}")
        for row in ladder_sweep:
            print(f"  {row['easy'][:26]:28}{row['medium'][:26]:28}{row['hard'][:22]:24}"
                  f"  {row['performance']:7.4f} {row['frontier_gap']:+8.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": {"path": str(args.data), "queries": len(df), "models": MODELS},
        "seed": args.seed,
        "results": results,
        "non_adaptive_frontier": [p["policy"] for p in hull],
        "tier_ladder": ladder,
        "tier_ladder_source": ladder_source,
        "ladder_sweep": ladder_sweep,
    }
    (args.out / "baselines.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(args.out / "baselines.csv", index=False, encoding="utf-8")
    plot(results, hull, args.out / "pareto.png", len(df), ladder)
    print(f"\n저장: {args.out}/baselines.json, baselines.csv, pareto.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
