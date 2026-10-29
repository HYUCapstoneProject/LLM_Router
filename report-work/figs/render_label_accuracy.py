"""Render overall and task-category label accuracy for the same 4,434 questions."""

import ast
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "report-work/figs/data/label-accuracy"
OUT = ROOT / "report-work/figs/report-figures"
METHODS = {
    "R0": "난이도 분류 (배포)",
    "R2": "후보 간 쌍별 순위 학습",
    "R3": "비용·품질 효용 기반 가중 분류",
    "R1": "후보별 성공 확률 예측",
    "F3": "재학습: easy 보충",
    "F1": "재학습: 새 피드백만",
    "F2": "재학습: 과거 데이터 섞기",
}
COLOR = "#4c43c4"
GRAY = "#6b707b"
TEXT = "#25272c"
FONT = ROOT / "report-work/ppt-typography-revision-20261020/fonts"
for name in ("Pretendard-Regular.ttf", "Pretendard-Bold.ttf"):
    font_manager.fontManager.addfont(FONT / name)
plt.rcParams.update({
    "font.family": "Pretendard",
    "font.size": 14,
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "text.color": TEXT,
})

parser = argparse.ArgumentParser()
parser.add_argument("--overall-only", action="store_true")
args = parser.parse_args()
manifest = json.loads((DATA / "manifest.json").read_text())
if args.overall_only:
    # The archived execution contains the complete overall accuracy table.
    # Per-question recovery is necessary only for the category breakdown.
    overall = pd.DataFrame.from_dict(
        manifest["historical_summary_percent"], orient="index",
        columns=["accuracy"],
    ).loc[list(METHODS)]
else:
    overall = pd.read_csv(DATA / "overall.csv").set_index("method").loc[list(METHODS)]
    assert overall.n.eq(4434).all()
for method, expected in manifest["historical_summary_percent"].items():
    assert round(overall.loc[method, "accuracy"], 3) == expected


def save_figure(fig, basename):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= -0.1 and box.y0 >= -0.1, text.get_text()
        assert box.x1 <= fig.bbox.width + .1, text.get_text()
        assert box.y1 <= fig.bbox.height + .1, text.get_text()
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"{basename}.{extension}", dpi=220, facecolor="white")
    plt.close(fig)
    print(f"Saved {basename}.png and .svg")


fig, ax = plt.subplots(figsize=(14.5, 6.2))
fig.subplots_adjust(left=.335, right=.97, top=.97, bottom=.15)
values = overall.accuracy.to_numpy()
positions = np.arange(len(METHODS))
ax.barh(
    positions, values, height=.49,
    color=[COLOR] + [GRAY] * (len(METHODS) - 1), zorder=3,
)
ax.set_yticks(positions, METHODS.values(), fontsize=15)
ax.invert_yaxis()
ax.get_yticklabels()[0].set_fontweight("bold")
ax.set_xlim(0, 100)
ax.set_xticks(np.arange(0, 101, 20))
ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
ax.set_xlabel("난이도 라벨 정확도", labelpad=15, fontsize=15)
ax.tick_params(axis="both", length=0, pad=10, labelcolor=TEXT)
ax.grid(axis="x", color="#e4e5ec", linewidth=.8, zorder=0)
for spine in ax.spines.values():
    spine.set_visible(False)
for i, value in enumerate(values):
    ax.text(
        value + 1, i, f"{value:.1f}%", va="center",
        color=COLOR if i == 0 else TEXT, weight="bold", fontsize=15,
    )
save_figure(fig, "5-05b-label-accuracy")
if args.overall_only:
    raise SystemExit(0)

frame = pd.read_parquet(DATA / "per-question-predictions.parquet")
assert set(frame.method) == set(METHODS)
assert frame.groupby(["method", "seed"]).sample_id.nunique().eq(4434).all()
# Reuse the report's existing shared category definition without importing its
# experiment modules. No new categories or post-hoc accuracy-based groups.
category_source = ROOT / (
    ".worktrees/router-final-common/ml/notebooks/router_final/categories.py"
)
definitions = {}
for statement in ast.parse(category_source.read_text()).body:
    if isinstance(statement, ast.Assign):
        for target in statement.targets:
            if isinstance(target, ast.Name) and target.id in {"TASK_TYPE", "TYPES"}:
                definitions[target.id] = ast.literal_eval(statement.value)
mapping = definitions["TASK_TYPE"]
frame["task_type"] = [
    mapping[(source, category)]
    for source, category in zip(frame.source_dataset, frame.category)
]
categories = [c for c in definitions["TYPES"] if c in set(frame.task_type)]
assert len(categories) == 6
counts = (
    frame[frame.method.eq("R0")].groupby("task_type").sample_id.nunique()
    .reindex(categories)
)
assert counts.sum() == 4434
by_seed = frame.groupby(["method", "seed", "task_type"]).correct.mean().mul(100)
category_accuracy = by_seed.groupby(["method", "task_type"]).mean().unstack()
category_accuracy = category_accuracy.loc[list(METHODS), categories]
assert np.isfinite(category_accuracy.to_numpy()).all()
for method in METHODS:
    weighted = (category_accuracy.loc[method] * counts).sum() / counts.sum()
    np.testing.assert_allclose(weighted, overall.loc[method, "accuracy"], atol=1e-10)
category_accuracy.to_csv(DATA / "by-category.csv")
counts.rename("n").to_csv(DATA / "category-counts.csv")
source_map = pd.DataFrame([
    {"source_dataset": s, "source_category": c, "task_type": mapping[(s, c)]}
    for s, c in frame[["source_dataset", "category"]].drop_duplicates().itertuples(index=False, name=None)
])
source_map.to_csv(DATA / "category-mapping.csv", index=False)

fig = plt.figure(figsize=(15.5, 6.6))
ax = fig.add_axes([.315, .15, .585, .81])
cax = fig.add_axes([.915, .15, .014, .81])
values = category_accuracy.to_numpy()
cmap = LinearSegmentedColormap.from_list(
    "accuracy_purple", ["#faf9fe", "#e4e0f5", "#a39bdc", COLOR, "#302883"]
)
im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=100)
ax.set_yticks(positions, METHODS.values(), fontsize=14)
ax.get_yticklabels()[0].set_fontweight("bold")
ax.set_xticks(
    np.arange(len(categories)),
    [f"{category}\n({counts[category]:,}문항)" for category in categories],
    fontsize=13.5,
)
ax.tick_params(axis="both", length=0, pad=12)
for spine in ax.spines.values():
    spine.set_visible(False)
for row in range(values.shape[0]):
    for col in range(values.shape[1]):
        value = values[row, col]
        ax.text(
            col, row, f"{value:.1f}", va="center", ha="center",
            color="white" if value >= 60 else TEXT, fontsize=14,
            weight="bold" if row == 0 else "normal",
        )
bar = fig.colorbar(im, cax=cax, ticks=np.arange(0, 101, 20))
bar.ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
bar.ax.tick_params(length=0, labelsize=12, pad=6)
bar.outline.set_visible(False)
bar.set_label("난이도 라벨 정확도", fontsize=13, labelpad=10)
save_figure(fig, "5-05c-label-accuracy-by-category")

manifest["category_definition"] = {
    "path": str(category_source),
    "sha256": hashlib.sha256(category_source.read_bytes()).hexdigest(),
    "empty_categories_omitted": [c for c in definitions["TYPES"] if c not in categories],
}
manifest["derived_files"] = {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [
        DATA / "per-question-predictions.parquet",
        DATA / "overall.csv", DATA / "by-category.csv",
        DATA / "category-counts.csv", DATA / "category-mapping.csv",
    ]
}
(DATA / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print("Verified: all seven methods match the historical summary to 0.001 percentage point.")
print("Verified: six categories account for every one of the 4,434 questions.")
