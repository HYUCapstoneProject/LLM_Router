"""Render the report figures 6-5 to 6-8 with the R2/R3 group routers (2026-10-22).

12a/12b trained R2-group (pairwise) and R3-group (utility-gap-weighted CE) on the
same EV-10 blended rows, groups and source weights as R0/R1-group. 13_r23_gain
appended them to the frozen gain lock, so every router now comes from one table.

Router comparisons keep the earlier exclusions: the variants trained only on
RouterBench/xRouteBench, the operational R0/RT-08 reference rows and R1-BGE
(user decisions 2026-10-21). The main rule stays tau = .8 with 1:3:5 costs.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator, NullFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
R23 = (Path.home() / "Library/CloudStorage/GoogleDrive-alexyoo0506@gmail.com/내 드라이브"
       / "LLM_Router/router-final/runs/run001/r23-extension/gain-test-local-mac-20261022")
LATENCY = ROOT / (
    ".worktrees/router-final-common/experiments/results/"
    "ev10-router-final-20261018/report"
)
OUT = ROOT / "report-work/figs/report-figures"


def r23(name):
    """Read one r23 gain-test table, checking its SHA when the manifest lists it."""
    path = R23 / name
    manifest = json.loads((R23 / "manifest.json").read_text())
    outputs = manifest.get("outputs", manifest.get("files", {}))
    expected = outputs.get(name) if isinstance(outputs, dict) else None
    if isinstance(expected, dict):
        expected = expected.get("sha256")
    if expected is not None:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    return pd.read_parquet(path)


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
            max(hi, 0) + xmax * 0.015, i, signed(value), va="center", color=color,
            fontsize=11.5, fontweight="bold" if method in ours else "normal",
        ))
    ax.set_yticks(range(len(rows)), [row[0] for row in rows])
    for label, row in zip(ax.get_yticklabels(), rows):
        label.set_color(ink)
        label.set_fontweight("bold" if row[1] in ours else "normal")
    ax.set_ylim(len(rows) - 0.5, -0.6)
    return notes


NAMES = {
    "knn": "kNN",
    "r1_group": "후보별 성공 확률 예측",
    "r0_group": "난이도 분류",
    "r2_group": "후보 간 쌍별 순위 학습",
    "r3_group": "비용·품질 효용 가중 분류",
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
OURS = {"r0_group", "r1_group", "r2_group", "r3_group"}
OBJECTIVES = ["r0_group", "r1_group", "r2_group", "r3_group"]


def main_tau(data, basis="main"):
    return data[(data.basis == basis) & (data.scope == "macro") & (data.rule == "tau")
                & (data.param.round(2) == 0.8)]


# ---------------------------------------------------------------- R0~R3 G (6-5)
COST_LABEL = {"main": "단계 비용 1:3:5", "group_price": "단계 비용 = 집단 평균 가격"}


def r0r3(title, basis="group_price"):
    """Four training objectives on the same rows, groups, rule and test set.
    Main basis since 2026-10-22: group-mean data price (user decision); 1:3:5 is the sub view."""
    main = main_tau(r23("gain-macro.parquet"), basis).set_index("method")
    rows = [(NAMES[m], m, main.at[m, "G"], main.at[m, "G_lo"], main.at[m, "G_hi"])
            for m in OBJECTIVES]
    fig, ax = plt.subplots(figsize=(13, 5.6))
    fig.subplots_adjust(left=0.30, right=0.92, top=0.78, bottom=0.2)
    texts = dot_rows(ax, rows, OURS, 0.025)
    ax.set_xlim(-0.005, 0.02)
    ax.axvline(0, color=ink, linewidth=1)
    ax.xaxis.set_major_locator(MultipleLocator(0.005))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: signed(v, 3)))
    style(ax)
    texts += ax.get_yticklabels()
    ax.set_xlabel(f"첫 시도 이득 G (τ = .8 규칙, {COST_LABEL[basis]}, 원천 4개 평균)", fontsize=12.5,
                  color=ink, labelpad=12)
    texts.append(fig.text(0.03, 0.94, title, fontsize=20, fontweight="bold", color=ink,
                          va="top"))
    texts.append(fig.text(
        0.03, 0.86,
        "같은 학습 데이터 34,573행 · 같은 모델 집단 · test 17,662문항 · "
        "선 = 95% 구간 (질문 묶음 bootstrap 10,000회)",
        fontsize=12, color=gray, va="top"))
    check_bounds(fig, texts)
    save(fig, "g-6-05-r0r3-group")


# ---------------------------------------------------------------- G ranking (6-6)
def ev10_main(title, basis="main", name="g-6-06-mean-gain", xlim=(0, 0.03),
              cost_note=None):
    """G ranking. basis="group_price" keeps the same tau-rule decisions and only prices the
    tiers at each source's group-mean data price (high = 1), so C and H(C) change."""
    main = main_tau(r23("gain-macro.parquet"), basis)
    rows = main[main.method.isin(NAMES)].sort_values("G", ascending=False)
    assert len(rows) == len(NAMES)
    needed = main[main.method == "needed_tier"].iloc[0]
    fig, ax = plt.subplots(figsize=(13, 10.4))
    fig.subplots_adjust(left=0.27, right=0.86, top=0.86, bottom=0.15)
    texts = dot_rows(
        ax, [(NAMES[r.method], r.method, r.G, r.G_lo, r.G_hi) for r in rows.itertuples()],
        OURS, xlim[1] - xlim[0],
    )
    ax.set_xlim(*xlim)
    if xlim[0] < 0:
        ax.axvline(0, color=ink, linewidth=1)
    ax.xaxis.set_major_locator(MultipleLocator(0.005))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: signed(v, 3)))
    style(ax)
    cost = "단계 비용 1:3:5" if basis == "main" else "단계 비용 = 집단 평균 가격"
    ax.set_xlabel(f"첫 시도 이득 G (τ = .8 규칙, {cost}, 원천 4개 평균)", fontsize=12.5,
                  color=ink, labelpad=14)
    texts += ax.get_yticklabels()
    texts.append(fig.text(0.03, 0.955, title, fontsize=20, fontweight="bold", color=ink,
                          va="top"))
    texts.append(fig.text(
        0.03, 0.915,
        cost_note or f"test 17,662문항 · 정답 풀 확률로 고른 단계(참고 상한)는 {signed(needed.G)}",
        fontsize=12, color=gray, va="top"))
    handles = [Line2D([], [], marker="o", linestyle="none", color=purple, markersize=8, label="본 프로젝트"),
               Line2D([], [], marker="o", linestyle="none", color=gray, markersize=8, label="논문 라우터")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.265, 0.05), ncol=2,
               frameon=False, fontsize=12, columnspacing=2.0)
    texts.append(fig.text(0.27, 0.035, "선 = 95% 구간 (질문 묶음 bootstrap 10,000회)",
                          fontsize=11, color=gray))
    check_bounds(fig, texts)
    save(fig, name)


# ---------------------------------------------------------------- strata (6-7)
STRATA_BASES = R23.parent / "by-stratum-bases-local-mac-20261022"


def ev10_strata(title, basis="main", name="g-6-07-strata"):
    """basis="group_price" reads the strata table recomputed for both cost bases
    (strata_group_price.py; its main rows equal gain-test's by-stratum exactly)."""
    if basis == "main":
        data = r23("by-stratum.parquet")
    else:
        path = STRATA_BASES / "by-stratum-bases.parquet"
        manifest = json.loads((STRATA_BASES / "manifest.json").read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest() \
            == manifest["outputs"]["by-stratum-bases.parquet"]
        data = pd.read_parquet(path)
        data = data[data.basis == basis]
    data = data[(data.rule == "tau") & (data.param.round(2) == 0.8)]
    macro = data.groupby(["stratum", "method"]).G.mean().unstack(0)
    order = macro.loc[list(NAMES)].sort_values("solvable", ascending=False).index
    panels = [("solvable", "어느 단계든 80% 이상 푸는 문항"),
              ("beyond_partial", "일부만 풀리는 문항")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 9.6), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.95, top=0.83, bottom=0.1, wspace=0.12)
    texts = []
    lows = []
    for ax, (stratum, panel_title) in zip(axes, panels):
        values = macro.loc[order, stratum]
        lows.append(values.min())
        colors = [purple if m in OURS else gray for m in order]
        ax.barh(range(len(order)), values, color=colors, height=0.55, zorder=3)
        for i, value in enumerate(values):
            texts.append(ax.annotate(
                "0.000" if round(value, 3) == 0 else signed(value), xy=(value, i),
                xytext=(6 if value >= 0 else -6, 0),
                textcoords="offset points", va="center", fontsize=10.5,
                ha="left" if value >= 0 else "right", color=ink))
        ax.axvline(0, color=ink, linewidth=1)
        style(ax)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: signed(v, 2)))
        texts.append(ax.set_title(panel_title, loc="left", fontsize=13, fontweight="bold",
                                  color=ink, pad=10))
    axes[0].set_xlim(min(-0.015, lows[0] - 0.01), 0.03)
    axes[0].xaxis.set_major_locator(MultipleLocator(0.01))
    axes[1].set_xlim(min(-0.14, lows[1] - 0.03), 0.02)
    axes[1].xaxis.set_major_locator(MultipleLocator(0.04))
    axes[0].set_yticks(range(len(order)), [NAMES[m] for m in order])
    axes[0].set_ylim(len(order) - 0.5, -0.6)
    for label, method in zip(axes[0].get_yticklabels(), order):
        label.set_fontweight("bold" if method in OURS else "normal")
    texts += axes[0].get_yticklabels()
    texts.append(fig.text(0.03, 0.955, title, fontsize=20, fontweight="bold", color=ink,
                          va="top"))
    cost = "단계 비용 1:3:5" if basis == "main" else "단계 비용 = 집단 평균 가격"
    texts.append(fig.text(
        0.03, 0.91,
        f"τ = .8 규칙 · {cost} · 층 안에서 기준선을 다시 만듦 · 원천 4개 평균 · "
        "아무도 못 푸는 문항은 정의상 G = 0이라 생략",
        fontsize=12, color=gray, va="top"))
    check_bounds(fig, texts)
    save(fig, name)


# ---------------------------------------------------------------- latency (6-8)
SAME_AS_R1 = {"r2_group", "r3_group"}


def latency():
    """R2/R3 share R1-group's architecture (ModernBERT-base, 512 tokens, three outputs)
    and were not timed separately; their bars reuse R1-group's measurement, hatched."""
    data = pd.read_csv(LATENCY / "tables/latency_cpu.csv")
    manifest = json.loads((LATENCY / "manifest.json").read_text())
    assert hashlib.sha256((LATENCY / "tables/latency_cpu.csv").read_bytes()).hexdigest() \
        == manifest["outputs"]["tables/latency_cpu.csv"]
    measured = set(NAMES) - SAME_AS_R1
    selected = data[data.router.isin(measured) & data.stage.isin(["added_given_bge", "total"])]
    assert len(selected) == len(measured) and selected.router.is_unique
    r1 = selected[selected.router == "r1_group"].iloc[0]
    copies = pd.DataFrame([dict(r1, router=m) for m in sorted(SAME_AS_R1)])
    selected = pd.concat([selected, copies]).sort_values(["median_ms", "router"])
    fig, ax = plt.subplots(figsize=(14, 9.6))
    fig.subplots_adjust(left=0.24, right=0.97, top=0.97, bottom=0.17)
    baseline = 0.1
    colors = [purple if m in OURS else gray for m in selected.router]
    bars = ax.barh(range(len(selected)), selected.median_ms - baseline, left=baseline,
                   height=0.53, color=colors, zorder=3)
    for bar, method in zip(bars, selected.router):
        if method in SAME_AS_R1:
            bar.set_facecolor("white")
            bar.set_edgecolor(purple)
            bar.set_hatch("///")
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
    for i, (value, method) in enumerate(zip(selected.median_ms, selected.router)):
        label = f"{value:.2f} ms" if value < 10 else f"{value:.1f} ms"
        if method in SAME_AS_R1:
            label += " (같은 구조)"
        texts.append(ax.annotate(label, xy=(value, i), xytext=(8, 0),
                                 textcoords="offset points", va="center", fontsize=11.5,
                                 fontweight="bold", color=ink))
    ax.set_xlabel("추가 라우팅 지연 (CPU 중앙값, ms · 로그 스케일)\nBGE 임베딩이 이미 계산된 조건",
                  fontsize=12, labelpad=14, linespacing=1.5)
    handles = [Patch(facecolor="white", edgecolor=purple, hatch="///",
                     label="따로 재지 않음: 후보별 성공 확률 예측과 같은 구조(ModernBERT-base, "
                           "512토큰, 출력 3개)라 그 측정값으로 표시")]
    legend = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.03, 0.01),
                        frameon=False, fontsize=11)
    texts += [*ax.get_yticklabels(), ax.xaxis.label, *legend.get_texts()]
    check_bounds(fig, texts)
    save(fig, "g-6-08-latency")


# ---------------------------------------------------------------- task types (6-9)
TASKS = R23.parent / "gain-test-tasks-local-mac-20261022"
TYPES = ["수학", "코드", "지식 객관식", "상식·추론", "전문가 과학·난제", "사실 회상", "열린 대화·지시"]


TASK_BASES = R23.parent / "by-task-bases-local-mac-20261022"


def task_types(basis="main", name="g-6-09-task-types"):
    """Per-task G on the extension lock; rows follow the overall G ranking on the same basis.

    main comes from the 11b by-task table; group_price from the point-estimate rerun of the
    same cells (its main rows match 11b exactly). One colour scale for both, so they compare.
    """
    folder, file = (TASKS, "by-task.parquet") if basis == "main" else (TASK_BASES, "by-task-bases.parquet")
    data = pd.read_parquet(folder / file)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert hashlib.sha256((folder / file).read_bytes()).hexdigest() == manifest["outputs"][file]
    if basis != "main":
        data = data[data.basis.eq(basis)]
    data = data[data.source.eq("all") & data.rule.eq("tau") & (data.param.round(2) == 0.8)]
    wide = data.pivot_table(index="method", columns="task_type", values="G")[TYPES]
    order = main_tau(r23("gain-macro.parquet"), basis)
    order = order[order.method.isin(NAMES)].sort_values("G", ascending=False).method.tolist()
    wide = wide.loc[order]
    fig, ax = plt.subplots(figsize=(11.5, 9.4))
    fig.subplots_adjust(left=0.25, right=0.86, top=0.98, bottom=0.07)
    image = ax.imshow(wide.to_numpy(), cmap="RdBu", vmin=-0.04, vmax=0.04, aspect="auto")
    texts = []
    for i, method in enumerate(order):
        for j, task in enumerate(TYPES):
            value = wide.at[method, task]
            label = "0.000" if round(value, 3) == 0 else signed(value)
            texts.append(ax.text(j, i, label, ha="center", va="center", fontsize=11,
                                 color="white" if abs(value) > 0.025 else ink))
    ax.set_xticks(range(len(TYPES)), TYPES, fontsize=11.5)
    ax.set_yticks(range(len(order)), [NAMES[m] for m in order], fontsize=12)
    for label, method in zip(ax.get_yticklabels(), order):
        label.set_fontweight("bold" if method in OURS else "normal")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes([0.89, 0.25, 0.018, 0.5])
    bar = fig.colorbar(image, cax=cax)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=10.5, length=3)
    texts.append(cax.set_title("첫 시도 이득 G", fontsize=11.5, loc="left", pad=8))
    texts += [*ax.get_yticklabels(), *ax.get_xticklabels()]
    check_bounds(fig, texts)
    save(fig, name)


if __name__ == "__main__":
    import sys
    titles = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    r0r3(titles.get("r0r3", "같은 데이터로 학습한 네 학습 목표의 첫 시도 이득"))
    # Main = group-mean data price (6-6-1, 6-7-1); sub = Claude 1:3:5 (6-6-2, 6-7-2).
    group = main_tau(r23("gain-macro.parquet"), "group_price")
    needed = group[group.method == "needed_tier"].G.iloc[0]
    ev10_main(
        titles.get("group_price", "집단 평균 가격 기준 상위권은 kNN과 난이도 분류다"),
        basis="group_price", name="g-6-06-1-group-price", xlim=(-0.01, 0.02),
        cost_note=("test 17,662문항 · low:medium:high = 1:2.8:45(RouterBench) · 1:2.8:17(xRouteBench) · "
                   f"1:9.7:60(LLMRouterBench) · 1:4.5:88(SPROUT) · 참고 상한 {signed(needed)}"))
    claude = main_tau(r23("gain-macro.parquet"))
    needed = claude[claude.method == "needed_tier"].G.iloc[0]
    ev10_main(
        titles.get("main", "Claude 가격 비율 1:3:5로 옮기면 이득이 커지고 순위가 바뀐다"),
        name="g-6-06-2-claude-135",
        cost_note=f"test 17,662문항 · Claude Haiku : Sonnet : Opus = 1 : 3 : 5 · 참고 상한 {signed(needed)}")
    ev10_strata(titles.get("strata_group_price",
                           "집단 평균 가격 기준 풀리는 문항 안의 이득은 kNN과 난이도 분류에만 남는다"),
                basis="group_price", name="g-6-07-1-strata-group-price")
    ev10_strata(titles.get("strata", "1:3:5로 옮기면 풀리는 문항 안의 이득이 대부분 양수가 된다"),
                name="g-6-07-2-strata-claude-135")
    latency()
    if TASK_BASES.exists():
        task_types("group_price", "g-6-09-1-task-types-group-price")
    if TASKS.exists():
        task_types("main", "g-6-09-2-task-types-claude-135")
