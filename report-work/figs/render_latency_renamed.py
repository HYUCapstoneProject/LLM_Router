"""Render all 16 comparison routers on a logarithmic latency axis."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / (
    ".worktrees/router-final-common/experiments/results/"
    "ev10-router-final-20261018/report"
)
DATA = REPORT / "tables/latency_cpu.csv"
manifest = json.loads((REPORT / "manifest.json").read_text())
assert hashlib.sha256(DATA.read_bytes()).hexdigest() == manifest["outputs"][
    "tables/latency_cpu.csv"
]

font_dir = ROOT / "report-work/ppt-typography-revision-20261020/fonts"
for name in ("Pretendard-Regular.ttf", "Pretendard-Bold.ttf"):
    font_manager.fontManager.addfont(font_dir / name)
plt.rcParams.update(
    {
        "font.family": "Pretendard",
        "font.size": 12,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
    }
)
data = pd.read_csv(DATA)
methods = {
    "r1_group": "후보별 성공 확률 예측 (가격대 묶음)",
    "r1_group_a": "후보별 성공 확률 예측\n(가격대 묶음, 기존 데이터만)",
    "r0_group": "난이도 분류 (가격대 묶음)",
    "graphrouter": "GraphRouter",
    "routerdc": "RouterDC",
    "routellm_released": "RouteLLM 공개본",
    "routellm_retrained": "RouteLLM 재학습본",
    "mf": "MF",
    "knn": "kNN",
    "knn_a": "kNN (기존 데이터만)",
    "svm": "SVM",
    "mlp": "MLP",
    "irt_router": "IRT-Router",
    "avengers_pro": "Avengers-Pro",
    "hybrid_llm": "Hybrid LLM",
}
# Compare added latency with the cache's BGE embedding already available.
# Routers with their own encoder retain their complete measured latency.
# R1-BGE is left out of the report (2026-10-21); the source table still has it.
selected = data[
    data.router.isin(methods)
    & data.stage.isin(["added_given_bge", "total"])
].copy()
assert len(selected) == 15 and selected.router.is_unique
assert set(selected.router) == set(methods)
assert selected.median_ms.notna().all() and selected.median_ms.gt(0).all()
selected = selected.sort_values("median_ms")
rows = [
    (
        row.router,
        methods[row.router],
        row.median_ms,
        f"{row.median_ms:.2f} ms" if row.median_ms < 10 else f"{row.median_ms:.1f} ms",
    )
    for row in selected.itertuples()
]

fig, ax = plt.subplots(figsize=(15, 10.2))
fig.subplots_adjust(left=0.34, right=0.98, top=0.985, bottom=0.13)
colors = [
    "#4c43c4" if method in {"r1_group", "r1_group_a", "r0_group"}
    else "#686d78"
    for method, *_ in rows
]
# A log axis cannot start at zero; keep each bar's endpoint at its measured value.
baseline = 0.1
ax.barh(
    range(len(rows)), [row[2] - baseline for row in rows],
    left=baseline, height=0.53, color=colors, zorder=3,
)
ax.set_yticks(range(len(rows)), [row[1] for row in rows], fontsize=12)
ax.set_ylim(len(rows) - 0.45, -0.55)
ax.set_xscale("log")
ax.set_xlim(baseline, 1000)
ax.set_xticks([0.1, 1, 10, 100, 1000])
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.tick_params(axis="y", length=0, pad=15)
ax.tick_params(axis="x", which="both", length=0, pad=9, colors="#686d78")
ax.grid(axis="x", color="#e2e4e9", linewidth=0.8, zorder=0)
ax.grid(axis="x", which="minor", color="#f0f1f4", linewidth=0.6, zorder=0)
for spine in ax.spines.values():
    spine.set_visible(False)
for i, (_, _, value, text) in enumerate(rows):
    ax.annotate(
        text, xy=(value, i), xytext=(8, 0), textcoords="offset points",
        va="center", fontsize=11.5,
        fontweight="bold", color="#25272c",
    )
ax.set_xlabel(
    "추가 라우팅 지연 (CPU 중앙값, ms · 로그 스케일)\nBGE 임베딩이 이미 계산된 조건",
    fontsize=12, labelpad=14, linespacing=1.5,
)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for text in [*ax.get_yticklabels(), *ax.get_xticklabels(), *ax.texts, ax.xaxis.label]:
    box = text.get_window_extent(renderer)
    assert box.x0 >= 0 and box.y0 >= 0, text.get_text()
    assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, text.get_text()

output = ROOT / "report-work/figs/report-figures/5-10-latency-log"
fig.savefig(output.with_suffix(".png"), dpi=240, facecolor="white")
fig.savefig(output.with_suffix(".svg"), facecolor="white")
print(f"Saved {output.with_suffix('.png')}")
print("Verified source checksum; all 15 comparison routers shown individually (R1-BGE excluded).")
