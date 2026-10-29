"""Render the existing EV-10 kappa curves as a 2-by-2 report figure."""

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = (
    Path.home()
    / "Library/CloudStorage/GoogleDrive-alexyoo0506@gmail.com"
    / "내 드라이브/LLM_Router/router-final/runs/run001/test/track1.parquet"
)
EXPECTED_SHA256 = "e1e4b887e9daabf578311a7c7a45d6aa4efa1bd87b63fce5194f88bb50666b02"
assert hashlib.sha256(DATA.read_bytes()).hexdigest() == EXPECTED_SHA256

font_dir = ROOT / "report-work/ppt-typography-revision-20261020/fonts"
for name in ("Pretendard-Regular.ttf", "Pretendard-Bold.ttf"):
    font_manager.fontManager.addfont(font_dir / name)

plt.rcParams.update(
    {
        "font.family": "Pretendard",
        "font.size": 12,
        "axes.titlesize": 17,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
    }
)

methods = [
    ("graphrouter", "GraphRouter", "#2a6fdb"),
    ("r1_group", "후보별 성공 확률 예측 (가격대 묶음)", "#e0582b"),
    ("r0_group", "난이도 분류 (가격대 묶음)", "#1d9a6c"),
    ("knn", "kNN", "#9b4fd1"),
    ("routerdc", "RouterDC", "#c9a227"),
    ("routellm_released", "RouteLLM 공개본", "#5b6b7a"),
]
sources = [
    ("routerbench", "RouterBench"),
    ("xroutebench", "xRouteBench"),
    ("llmrouterbench", "LLMRouterBench"),
    ("sprout", "SPROUT"),
]

data = pd.read_parquet(DATA)
selected = data[
    data.basis.eq("main") & data.rule.eq("escalation") & data.level.eq(0.5)
]
fig, axes = plt.subplots(2, 2, figsize=(12, 10.4), sharey=False)
for ax, (source, title) in zip(axes.flat, sources):
    for method, label, color in methods:
        part = selected[
            selected.source.eq(source) & selected.method.eq(method)
        ].sort_values("kappa")
        assert len(part) == 21 and part.kappa.is_unique
        assert part[["kappa", "gain", "gain_lo", "gain_hi"]].notna().all().all()
        ax.plot(part.kappa, part.gain, color=color, lw=2, label=label)
        ax.fill_between(
            part.kappa, part.gain_lo, part.gain_hi, color=color, alpha=0.12, lw=0
        )
    ax.axhline(0, color="#888", lw=0.8)
    ax.axhline(0.01, color="#bbb", lw=0.8, ls="--")
    ax.set_title(title, pad=12)
    ax.set_xlabel("kappa (weight of one failed attempt)", labelpad=8)
    ax.tick_params(labelsize=11)

for ax in axes[:, 0]:
    ax.set_ylabel("비용 수준 50%에서의 효용 이득", labelpad=10)

fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.18, hspace=0.36, wspace=0.24)
handles, labels = axes[0, 0].get_legend_handles_labels()
legend = fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.53, 0.025),
    ncol=2,
    frameon=False,
    fontsize=12,
    columnspacing=3.0,
    handlelength=2.5,
    labelspacing=0.8,
)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for text in [*legend.get_texts(), *(ax.title for ax in axes.flat)]:
    box = text.get_window_extent(renderer)
    assert box.x0 >= 0 and box.y0 >= 0
    assert box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height

output = ROOT / "report-work/figs/report-figures/5-07-gain-by-kappa-2x2"
fig.savefig(output.with_suffix(".png"), dpi=240, facecolor="white")
fig.savefig(output.with_suffix(".svg"), facecolor="white")
print(f"Saved {output.with_suffix('.png')}")
print(f"Verified original data SHA-256; rendered 24 curves and 24 confidence bands.")
