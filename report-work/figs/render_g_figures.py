"""Render the report figures that use the first-attempt gain G (2026-10-21).

Router comparisons drop the variants trained only on RouterBench/xRouteBench,
the operational R0/RT-08 reference rows and R1-BGE (user decisions 2026-10-21).
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / ".worktrees/gain-metric/experiments/results"
EV10 = RESULTS / "ev10-gain-20261021"
R0R3 = RESULTS / "gain-metric-r0r3-20261021"
LATENCY = ROOT / (
    ".worktrees/router-final-common/experiments/results/"
    "ev10-router-final-20261018/report"
)
OUT = ROOT / "report-work/figs/report-figures"


def checked(base, relative, key=None):
    path = base / relative
    manifest = json.loads((base / "manifest.json").read_text())
    outputs = manifest.get("outputs", manifest.get("files", {}))
    expected = outputs.get(key or relative)
    if isinstance(expected, dict):
        expected = expected.get("sha256")
    if expected is not None:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    return pd.read_csv(path)


font_dir = ROOT / "report-work/ppt-typography-revision-20261020/fonts"
for name in ("Pretendard-Regular.ttf", "Pretendard-Bold.ttf"):
    font_manager.fontManager.addfont(font_dir / name)
plt.rcParams.update(
    {"font.family": "Pretendard", "font.size": 12, "axes.unicode_minus": False,
     "svg.fonttype": "none"}
)
purple, gray, pale, ink, grid = "#4c43c4", "#686d78", "#b8bdc8", "#25272c", "#e2e4e9"


def signed(value, digits=3):
    return f"{value:+.{digits}f}".replace("-", "−")


def style(ax):
    ax.tick_params(axis="y", length=0, pad=12)
    ax.tick_params(axis="x", length=0, pad=8, colors=gray, labelsize=11)
    ax.grid(axis="x", color=grid, linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def check_bounds(fig, texts):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in texts:
        box = item.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0, item.get_text()
        assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, item.get_text()


def save(fig, name):
    for suffix in (".png", ".svg"):
        fig.savefig((OUT / name).with_suffix(suffix), dpi=220, facecolor="white")
    plt.close(fig)
    print(f"Saved {(OUT / name).with_suffix('.png')}")


def dot_rows(ax, rows, ours, xmax):
    """rows: list of (label, method, G, lo, hi)."""
    notes = []
    for i, (_, method, value, lo, hi) in enumerate(rows):
        color = purple if method in ours else gray
        ax.hlines(i, lo, hi, color=color, linewidth=1.7)
        ax.plot(value, i, "o", ms=7.5, color=color)
        notes.append(ax.text(
            hi + xmax * 0.015, i, signed(value), va="center", color=color,
            fontsize=11.5, fontweight="bold" if method in ours else "normal",
        ))
    ax.set_yticks(range(len(rows)), [row[0] for row in rows])
    for label, row in zip(ax.get_yticklabels(), rows):
        label.set_color(ink)
        label.set_fontweight("bold" if row[1] in ours else "normal")
    ax.set_ylim(len(rows) - 0.5, -0.6)
    return notes


# ---------------------------------------------------------------- R0~R3 G
def r0r3():
    """Four training objectives in one panel (user decision 2026-10-21).

    R0-R2 share training data and the 7,202-item final test. R3 was trained only on the
    split that holds some task types out, so its value comes from that split's 2,314-item
    selection set; it is drawn apart in gray and footnoted, not compared directly.
    """
    data = checked(R0R3, "macro-summary.csv")
    data = data[data.criterion == "lambda0"]
    pick = [("A", "test", "R0", "난이도 분류"), ("A", "test", "R1", "후보별 성공 확률 예측"),
            ("A", "test", "R2", "후보 간 쌍별 순위 학습"),
            ("G1", "selection", "R3-NATURAL", "비용·품질 효용 가중 분류*")]
    rows = []
    for experiment, split, arm, label in pick:
        row = data[(data.experiment == experiment) & (data.arm == arm) & (data.split == split)].iloc[0]
        rows.append((label, arm, row.G, row.G_lo, row.G_hi))
    fig, ax = plt.subplots(figsize=(13, 5.9))
    fig.subplots_adjust(left=0.30, right=0.92, top=0.79, bottom=0.25)
    texts = dot_rows(ax, rows, {"R0", "R1", "R2"}, 0.04)
    # Separate R3 from the directly comparable three.
    ax.axhline(2.5, color=grid, linewidth=1.2, linestyle=(0, (4, 4)))
    ax.set_xlim(-0.03, 0.04)
    ax.axvline(0, color=ink, linewidth=1)
    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: signed(v, 2)))
    style(ax)
    texts += ax.get_yticklabels()
    ax.set_xlabel("첫 시도 이득 G (모델 출력 그대로, 두 원천 평균)", fontsize=12.5,
                  color=ink, labelpad=12)
    texts.append(fig.text(0.03, 0.94, "성공 확률 예측·쌍별 순위 학습이 난이도 분류보다 높다",
                          fontsize=20, fontweight="bold", color=ink, va="top"))
    texts.append(fig.text(
        0.03, 0.86,
        "최종 평가 세트 7,202문항 · 선 = 95% 구간 (문항 bootstrap 2,000회) · 0 = 같은 비용의 고정 단계 혼합과 같음",
        fontsize=12, color=gray, va="top"))
    texts.append(fig.text(
        0.03, 0.05,
        "* 세 방식을 비교한 뒤 일반화 실험에서 추가로 학습한 값(일부 문제 유형을 학습에서 뺀 분할, "
        "모델 선정용 2,314문항). 학습·평가 문항이 달라 위 셋과 직접 비교하지 않는다.",
        fontsize=11, color=gray))
    check_bounds(fig, texts)
    save(fig, "g-5-05-r0r3")


# ---------------------------------------------------------------- EV-10 G ranking
NAMES = {
    "knn": "kNN",
    "r1_group": "후보별 성공 확률 예측",
    "r0_group": "난이도 분류",
    "irt_router": "IRT-Router",
    "graphrouter": "GraphRouter",
    "svm": "SVM",
    "routellm_retrained": "RouteLLM 재학습본",
    "mlp": "MLP",
    "routerdc": "RouterDC",
    "mf": "MF",
    "hybrid_llm": "Hybrid LLM",
    "routellm_released": "RouteLLM 공개본",
    "avengers_pro": "Avengers-Pro",
}
OURS = {"r1_group", "r0_group"}


def ev10_main():
    data = checked(EV10, "tables/gain-macro.csv")
    main = data[(data.basis == "main") & (data.scope == "macro") & (data.rule == "tau")
                & (data.param.round(2) == 0.8)]
    rows = main[main.method.isin(NAMES)].sort_values("G", ascending=False)
    assert len(rows) == len(NAMES)
    needed = main[main.method == "needed_tier"].iloc[0]
    fig, ax = plt.subplots(figsize=(13, 9.4))
    fig.subplots_adjust(left=0.27, right=0.86, top=0.85, bottom=0.17)
    texts = dot_rows(
        ax, [(NAMES[r.method], r.method, r.G, r.G_lo, r.G_hi) for r in rows.itertuples()],
        OURS, 0.03,
    )
    ax.set_xlim(0, 0.03)
    ax.xaxis.set_major_locator(MultipleLocator(0.005))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: signed(v, 3)))
    style(ax)
    ax.set_xlabel("첫 시도 이득 G (τ = .8 규칙, 원천 4개 평균)", fontsize=12.5, color=ink,
                  labelpad=14)
    texts += ax.get_yticklabels()
    texts.append(fig.text(0.03, 0.955, "상위 라우터의 첫 시도 이득은 +0.02 수준이다",
                          fontsize=20, fontweight="bold", color=ink, va="top"))
    texts.append(fig.text(
        0.03, 0.91,
        f"test 17,662문항 · 정답 풀 확률로 고른 단계(참고 상한)는 {signed(needed.G)}",
        fontsize=12, color=gray, va="top"))
    handles = [Line2D([], [], marker="o", linestyle="none", color=purple, markersize=8, label="본 프로젝트"),
               Line2D([], [], marker="o", linestyle="none", color=gray, markersize=8, label="논문 라우터")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.265, 0.06), ncol=2,
               frameon=False, fontsize=12, columnspacing=2.0)
    texts.append(fig.text(0.27, 0.045, "선 = 95% 구간 (질문 묶음 bootstrap 10,000회)",
                          fontsize=11, color=gray))
    check_bounds(fig, texts)
    save(fig, "g-5-08-mean-gain")


def ev10_strata():
    data = checked(EV10, "tables/by-stratum.csv")
    data = data[(data.rule == "tau") & (data.param.round(2) == 0.8)]
    macro = data.groupby(["stratum", "method"]).G.mean().unstack(0)
    order = macro.loc[list(NAMES)].sort_values("solvable", ascending=False).index
    panels = [("solvable", "어느 단계든 80% 이상 푸는 문항"),
              ("beyond_partial", "일부만 풀리는 문항")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8.6), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.95, top=0.82, bottom=0.12, wspace=0.12)
    texts = []
    for ax, (stratum, title) in zip(axes, panels):
        values = macro.loc[order, stratum]
        colors = [purple if m in OURS else gray for m in order]
        ax.barh(range(len(order)), values, color=colors, height=0.55, zorder=3)
        for i, value in enumerate(values):
            texts.append(ax.annotate(
                signed(value), xy=(value, i), xytext=(6 if value >= 0 else -6, 0),
                textcoords="offset points", va="center", fontsize=10.5,
                ha="left" if value >= 0 else "right", color=ink))
        ax.axvline(0, color=ink, linewidth=1)
        style(ax)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: signed(v, 2)))
        texts.append(ax.set_title(title, loc="left", fontsize=13, fontweight="bold",
                                  color=ink, pad=10))
    axes[0].set_xlim(-0.015, 0.03)
    axes[0].xaxis.set_major_locator(MultipleLocator(0.01))
    axes[1].set_xlim(-0.14, 0.02)
    axes[1].xaxis.set_major_locator(MultipleLocator(0.04))
    axes[0].set_yticks(range(len(order)), [NAMES[m] for m in order])
    axes[0].set_ylim(len(order) - 0.5, -0.6)
    for label, method in zip(axes[0].get_yticklabels(), order):
        label.set_fontweight("bold" if method in OURS else "normal")
    texts += axes[0].get_yticklabels()
    texts.append(fig.text(0.03, 0.955, "풀리는 문항 안에서도 첫 시도 이득은 대부분 양수다",
                          fontsize=20, fontweight="bold", color=ink, va="top"))
    texts.append(fig.text(
        0.03, 0.905,
        "τ = .8 규칙 · 층 안에서 기준선을 다시 만듦 · 원천 4개 평균 · 아무도 못 푸는 문항은 정의상 G = 0이라 생략",
        fontsize=12, color=gray, va="top"))
    check_bounds(fig, texts)
    save(fig, "g-5-09-strata")


def latency():
    data = pd.read_csv(LATENCY / "tables/latency_cpu.csv")
    manifest = json.loads((LATENCY / "manifest.json").read_text())
    assert hashlib.sha256((LATENCY / "tables/latency_cpu.csv").read_bytes()).hexdigest() \
        == manifest["outputs"]["tables/latency_cpu.csv"]
    selected = data[data.router.isin(NAMES) & data.stage.isin(["added_given_bge", "total"])]
    assert len(selected) == len(NAMES) and selected.router.is_unique
    selected = selected.sort_values("median_ms")
    fig, ax = plt.subplots(figsize=(14, 8.8))
    fig.subplots_adjust(left=0.24, right=0.97, top=0.97, bottom=0.15)
    baseline = 0.1
    colors = [purple if m in OURS else gray for m in selected.router]
    ax.barh(range(len(selected)), selected.median_ms - baseline, left=baseline,
            height=0.53, color=colors, zorder=3)
    ax.set_yticks(range(len(selected)), [NAMES[m] for m in selected.router], fontsize=12)
    for label, method in zip(ax.get_yticklabels(), selected.router):
        label.set_fontweight("bold" if method in OURS else "normal")
    ax.set_ylim(len(selected) - 0.45, -0.55)
    ax.set_xscale("log")
    ax.set_xlim(baseline, 1000)
    ax.set_xticks([0.1, 1, 10, 100, 1000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    style(ax)
    ax.grid(axis="x", which="minor", color="#f0f1f4", linewidth=0.6, zorder=0)
    texts = []
    for i, value in enumerate(selected.median_ms):
        label = f"{value:.2f} ms" if value < 10 else f"{value:.1f} ms"
        texts.append(ax.annotate(label, xy=(value, i), xytext=(8, 0),
                                 textcoords="offset points", va="center", fontsize=11.5,
                                 fontweight="bold", color=ink))
    ax.set_xlabel("추가 라우팅 지연 (CPU 중앙값, ms · 로그 스케일)\nBGE 임베딩이 이미 계산된 조건",
                  fontsize=12, labelpad=14, linespacing=1.5)
    texts += [*ax.get_yticklabels(), ax.xaxis.label]
    check_bounds(fig, texts)
    save(fig, "g-5-10-latency")


def retrain():
    arms = ["재학습 안 함", "새 피드백만", "과거 데이터 섞기", "easy 보충"]
    panels = [("easy로 보낸 비율", [60.1, 0.0, 0.7, 24.7], "{:.1f}%", 70),
              ("질문당 비용 (재학습 안 함 = 1)", [1.0, 1.77, 1.75, 1.63], "{:.2f}", 2.0),
              ("첫 시도 해결률", [59.2, 79.2, 78.8, 74.6], "{:.1f}%", 100)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.2))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.76, bottom=0.2, wspace=0.25)
    texts = []
    for ax, (title, values, fmt, ymax) in zip(axes, panels):
        colors = [purple] + [gray] * 3
        ax.bar(range(4), values, color=colors, width=0.6, zorder=3)
        for i, value in enumerate(values):
            texts.append(ax.text(i, value + ymax * 0.02, fmt.format(value), ha="center",
                                 fontsize=11.5, color=ink,
                                 fontweight="bold" if i == 0 else "normal"))
        ax.set_ylim(0, ymax * 1.12)
        ax.set_xticks(range(4), arms, fontsize=11)
        ax.set_yticks([])
        ax.tick_params(axis="x", length=0, pad=8)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(pale)
        texts.append(ax.set_title(title, loc="left", fontsize=13, fontweight="bold",
                                  color=ink, pad=12))
        texts += ax.get_xticklabels()
    texts.append(fig.text(0.04, 0.955, "피드백으로 재학습하면 easy가 사라지고 비용이 63~77% 늘었다",
                          fontsize=19, fontweight="bold", color=ink, va="top"))
    texts.append(fig.text(0.04, 0.89, "피드백 시뮬레이션 6구간 · 구간마다 다음 구간 질문으로 평가",
                          fontsize=12, color=gray, va="top"))
    check_bounds(fig, texts)
    save(fig, "g-5-11-retrain")


if __name__ == "__main__":
    r0r3()
    ev10_main()
    ev10_strata()
    latency()
    retrain()
