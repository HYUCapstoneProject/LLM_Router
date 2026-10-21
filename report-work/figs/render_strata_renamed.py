"""Render the existing strata results with concise Korean labels."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / (
    ".worktrees/router-final-common/experiments/results/"
    "ev10-router-final-20261018/report"
)
DATA = REPORT / "tables/strata_gain.csv"
manifest = json.loads((REPORT / "manifest.json").read_text())
assert hashlib.sha256(DATA.read_bytes()).hexdigest() == manifest["outputs"][
    "tables/strata_gain.csv"
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

methods = [
    ("graphrouter", "GraphRouter", "#686d78"),
    ("r1_group", "후보별 성공 확률 예측 (가격대 묶음)", "#4c43c4"),
    ("r0_group", "난이도 분류 (가격대 묶음)", "#938ed6"),
    ("knn", "kNN", "#686d78"),
    ("oracle", "oracle", "#b8bdc8"),
]
strata = ["solvable", "beyond_partial", "beyond_zero"]
labels = ["풀리는 문항", "일부만 풀림", "아무도 못 품"]
frame = pd.read_csv(DATA).set_index("method")

fig, ax = plt.subplots(figsize=(13, 7.5))
fig.subplots_adjust(left=0.09, right=0.985, top=0.965, bottom=0.26)
positions = np.arange(3)
for i, (method, label, color) in enumerate(methods):
    values = frame.loc[method, strata].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    ax.bar(
        positions + (i - 2) * 0.145,
        values,
        width=0.13,
        color=color,
        label=label,
        zorder=3,
    )

ax.set_ylim(-0.21, 0.17)
ax.set_xticks(positions, labels, fontsize=14)
ax.tick_params(axis="x", length=0, pad=10)
ax.tick_params(axis="y", length=0, pad=8, colors="#686d78")
ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+.2f}"))
ax.grid(axis="y", color="#e2e4e9", linewidth=0.8, zorder=0)
ax.axhline(0, color="#969ba5", linewidth=1, zorder=2)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_ylabel("집단 내 평균 효용 이득", fontsize=14, labelpad=16)
ax.set_xlabel("문항 집단", fontsize=13, labelpad=16)

legend = fig.legend(
    *ax.get_legend_handles_labels(),
    loc="lower center",
    bbox_to_anchor=(0.53, 0.025),
    ncol=3,
    frameon=False,
    fontsize=11.5,
    columnspacing=2.2,
    handlelength=1.3,
    labelspacing=1.2,
)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for text in [*legend.get_texts(), *ax.get_xticklabels(), ax.xaxis.label, ax.yaxis.label]:
    box = text.get_window_extent(renderer)
    assert box.x0 >= 0 and box.y0 >= 0, text.get_text()
    assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, text.get_text()
legend_box = legend.get_window_extent(renderer)
assert legend_box.y1 < ax.xaxis.label.get_window_extent(renderer).y0

output = ROOT / "report-work/figs/report-figures/5-09-strata-renamed"
fig.savefig(output.with_suffix(".png"), dpi=240, facecolor="white")
fig.savefig(output.with_suffix(".svg"), facecolor="white")
print(f"Saved {output.with_suffix('.png')}")
print("Verified source checksum; retained all 15 plotted values.")
