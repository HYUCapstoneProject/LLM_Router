"""Update report method names using the existing EV-10 mean-gain estimates."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / (
    ".worktrees/router-final-common/experiments/results/"
    "ev10-router-final-20261018/report"
)
DATA = REPORT / "tables/mean_gain_macro_ci.csv"
manifest = json.loads((REPORT / "manifest.json").read_text())
assert hashlib.sha256(DATA.read_bytes()).hexdigest() == manifest["outputs"][
    "tables/mean_gain_macro_ci.csv"
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

names = {
    "oracle": "oracle (정답을 아는 상한)",
    "graphrouter": "GraphRouter",
    "r1_group": "후보별 성공 확률 예측 (가격대 묶음)",
    "r1_group_a": "후보별 성공 확률 예측\n(가격대 묶음, 기존 데이터만)",
    "r0_group": "난이도 분류 (가격대 묶음)",
    "knn": "kNN",
    "knn_a": "kNN (기존 데이터만)",
    "svm": "SVM",
    "routerdc": "RouterDC",
    "irt_router": "IRT-Router",
    "routellm_retrained": "RouteLLM 재학습본",
    "routellm_released": "RouteLLM 공개본",
    "mf": "MF",
    "mlp": "MLP",
    "avengers_pro": "Avengers-Pro",
    "hybrid_llm": "Hybrid LLM",
}
ours = {"r1_group", "r1_group_a", "r0_group"}
purple, gray, pale, ink = "#4c43c4", "#686d78", "#b8bdc8", "#25272c"
frame = pd.read_csv(DATA)
assert len(frame) == 17 and "r1_bge" in set(frame.method)
# R1-BGE is left out of the report (2026-10-21); the source table still has it.
frame = frame[frame.method != "r1_bge"].reset_index(drop=True)
assert set(frame.method) == set(names)

fig, ax = plt.subplots(figsize=(15, 12.6))
fig.subplots_adjust(left=0.325, right=0.80, top=0.865, bottom=0.215)
ax.set_xlim(0, 0.053)
ax.set_ylim(len(frame) - 0.5, -0.65)
ax.set_yticks(range(len(frame)), [names[m] for m in frame.method])
ax.tick_params(axis="y", length=0, pad=14, labelsize=12)
ax.tick_params(axis="x", length=0, pad=8, colors=gray, labelsize=11)
ax.xaxis.set_major_locator(MultipleLocator(0.01))
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"+{value:.2f}"))
ax.grid(axis="x", color="#e2e4e9", linewidth=0.9)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.add_patch(
    Rectangle((0.036, 0.5), 0.013, 3, facecolor=purple, alpha=0.06, linewidth=0)
)
annotations = []
for i, row in frame.iterrows():
    if row.method == "oracle":
        ax.hlines(i, 0, 0.051, color=pale, linestyles=(0, (3, 4)), linewidth=1)
        annotations.append(
            ax.text(
                0.054, i, f"+{row.mean_gain:.3f} (축 밖)",
                va="center", color=gray, fontsize=12, clip_on=False
            )
        )
        continue
    color = purple if row.method in ours else gray
    ax.hlines(i, row.lo, row.hi, color=color, linewidth=1.7)
    ax.plot(row.mean_gain, i, "o", ms=7.5, color=color)
    annotations.append(
        ax.text(
            row.hi + 0.0009, i, f"+{row.mean_gain:.3f}",
            va="center", color=color, fontsize=11.5,
            fontweight="bold" if row.method in ours else "normal",
            clip_on=False,
        )
    )
for label, method in zip(ax.get_yticklabels(), frame.method):
    label.set_color(ink)
    label.set_fontweight("bold" if method in ours else "normal")

ax.text(
    1.08, 2.2, "상위 3개 방법의\n95% 신뢰구간이 겹침",
    transform=ax.get_yaxis_transform(),
    va="center", color=purple, fontsize=12, linespacing=1.6,
)
ax.set_xlabel(
    "같은 비용의 고정 시작 정책 대비 평균 효용 이득",
    fontsize=13, color=ink, labelpad=18,
)
fig.text(
    0.045, 0.955, "상위권 라우터의 평균 효용 이득은 +0.04 수준이다",
    fontsize=22, fontweight="bold", color=ink, va="top",
)
fig.text(
    0.045, 0.914,
    "원천 4개 평균 · 실패 비용 가중치 κ 21개 × 비용 수준 3개 평균 · 논문 라우터 비교",
    fontsize=13, color=gray, va="top",
)
handles = [
    Line2D([], [], marker="o", linestyle="none", color=purple, markersize=8, label="본 프로젝트"),
    Line2D([], [], marker="o", linestyle="none", color=gray, markersize=8, label="논문 라우터"),
]
fig.legend(
    handles=handles, loc="lower left", bbox_to_anchor=(0.32, 0.108),
    ncol=2, frameon=False, fontsize=12, columnspacing=2.0,
)
fig.text(
    0.325, 0.088, "선 = 95% 신뢰구간 (질문 묶음 bootstrap 10,000회)",
    fontsize=11, color=gray,
)
fig.text(
    0.325, 0.060,
    "가격대 묶음: 모델을 원천별 가격대 집단으로 묶음   |   기존 데이터만: RouterBench·xRouteBench로만 학습",
    fontsize=10, color=gray,
)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for text in [*ax.get_yticklabels(), *fig.texts, *annotations]:
    box = text.get_window_extent(renderer)
    assert box.x0 >= 0 and box.y0 >= 0, text.get_text()
    assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, text.get_text()

output = ROOT / "report-work/figs/report-figures/fig5-9_mean_gain"
fig.savefig(output.with_suffix(".png"), dpi=220, facecolor="white")
fig.savefig(output.with_suffix(".svg"), facecolor="white")
print(f"Saved {output.with_suffix('.png')}")
print("Verified source checksum; kept 16 estimates and confidence intervals (R1-BGE excluded).")
