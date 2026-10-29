"""Render the task-type heatmap with the agreed routing method names."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / (
    ".worktrees/router-final-common/experiments/results/"
    "ev10-router-final-20261018/report"
)
DATA = REPORT / "tables/task_types.csv"
manifest = json.loads((REPORT / "manifest.json").read_text())
assert hashlib.sha256(DATA.read_bytes()).hexdigest() == manifest["outputs"][
    "tables/task_types.csv"
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
    ("graphrouter", "GraphRouter"),
    ("r1_group", "후보별 성공 확률 예측 (가격대 묶음)"),
    ("r0_group", "난이도 분류 (가격대 묶음)"),
    ("knn", "kNN"),
    ("routerdc", "RouterDC"),
    ("routellm_released", "RouteLLM 공개본"),
]
categories = [
    "수학",
    "코드",
    "지식 객관식",
    "상식·추론",
    "전문가 과학·난제",
    "사실 회상",
    "열린 대화·지시",
]
frame = pd.read_csv(DATA).set_index("method")
values = frame.loc[[method for method, _ in methods], categories].to_numpy()
assert values.shape == (6, 7) and np.isfinite(values).all()

# Retain the reference image's rounded values and symmetric color scale.
reference = np.array(
    [
        [.024, -.004, .042, .012, -.012, .005, .016],
        [.060, .019, .042, .009, -.006, .015, .016],
        [.032, -.003, .027, .006, -.011, .009, .016],
        [.005, -.002, .042, .015, -.019, .007, .016],
        [.019, -.059, .041, -.002, -.033, .000, .012],
        [-.007, -.016, .031, .000, -.015, .000, .009],
    ]
)
np.testing.assert_allclose(values.round(3), reference, atol=1e-12)

fig = plt.figure(figsize=(15, 5.5))
ax = fig.add_axes([0.28, 0.10, 0.65, 0.84])
color_axis = fig.add_axes([0.945, 0.10, 0.015, 0.84])
im = ax.imshow(values, cmap="RdBu", vmin=-0.06, vmax=0.06, aspect="auto")
ax.set_xticks(range(len(categories)), categories, fontsize=12)
ax.set_yticks(range(len(methods)), [label for _, label in methods], fontsize=12)
ax.tick_params(axis="both", length=0, pad=7)
for spine in ax.spines.values():
    spine.set_visible(False)

annotations = []
for row in range(values.shape[0]):
    for col in range(values.shape[1]):
        value = values[row, col]
        label = "0.000" if abs(value) < 0.0005 else f"{value:+.3f}"
        annotations.append(
            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=11,
                color="white" if abs(value) > 0.05 else "#222222",
            )
        )
colorbar = fig.colorbar(im, cax=color_axis, ticks=np.arange(-0.06, 0.061, 0.02))
colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
colorbar.ax.tick_params(length=0, labelsize=11, pad=5)
colorbar.outline.set_visible(False)
colorbar.ax.set_title("평균 이득", fontsize=11, pad=7)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for text in [
    *ax.get_xticklabels(),
    *ax.get_yticklabels(),
    *colorbar.ax.get_yticklabels(),
    colorbar.ax.title,
    *annotations,
]:
    box = text.get_window_extent(renderer)
    assert box.x0 >= 0 and box.y0 >= 0, text.get_text()
    assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height, text.get_text()

output = ROOT / "report-work/figs/report-figures/task-types-renamed"
fig.savefig(output.with_suffix(".png"), dpi=240, facecolor="white")
fig.savefig(output.with_suffix(".svg"), facecolor="white")
print(f"Saved {output.with_suffix('.png')}")
print("Verified source checksum and all 42 values against the reference image.")
