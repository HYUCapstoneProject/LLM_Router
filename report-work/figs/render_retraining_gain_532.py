"""Render section 5.3.2 from the completed Colab output, without recomputing G."""

import hashlib
import json
import shutil
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, MultipleLocator
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DRIVE = next(
    p / "LLM_Router"
    for p in Path("/Users/yoocheolmin/Google Drive").iterdir()
    if unicodedata.normalize("NFC", p.name) == "내 드라이브"
)
SOURCE = DRIVE / "retraining-gain-20261021"
OUT = ROOT / "report-work/figs/retraining-gain-20261021"
OUT.mkdir(exist_ok=True)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((SOURCE / "manifest.json").read_text())
for filename in ("gain-summary.csv", "delta-vs-f0.csv"):
    assert sha256(SOURCE / filename) == manifest["output_sha256"][filename]
    shutil.copy2(SOURCE / filename, OUT / filename)

summary = pd.read_csv(OUT / "gain-summary.csv")
summary = summary[(summary["set"] == "test") & summary.arm.isin(["F0", "F4", "F5"])]
summary = summary.set_index("arm").loc[["F0", "F4", "F5"]]
delta = pd.read_csv(OUT / "delta-vs-f0.csv")
delta = delta[delta["set"] == "test"].set_index("arm").loc[["F4", "F5"]]
assert manifest["sets"]["test"] == 7202
assert list(summary.seeds) == [1, 3, 3]
for arm in delta.index:
    assert abs(delta.loc[arm, "delta_G"] - (summary.loc[arm, "G"] - summary.loc["F0", "G"])) < 1e-12
    assert delta.loc[arm, "lo"] < 0 < delta.loc[arm, "hi"]

for weight in ("Regular", "Bold"):
    font_manager.fontManager.addfont(Path.home() / "Library/Fonts" / f"Pretendard-{weight}.ttf")
plt.rcParams.update({
    "font.family": "Pretendard", "font.size": 12,
    "axes.unicode_minus": False, "svg.fonttype": "path",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
INK, GRAY, GRID = "#25272c", "#686d78", "#e2e4e9"
COLORS = {"F0": GRAY, "F4": "#746bb8", "F5": "#4c43c4"}
LABELS = ["재학습 안 함", "양방향 교정\n(오답만)", "양방향 교정\n+ 비율 맞춤"]
Y = {"F0": 2, "F4": 1, "F5": 0}

fig = plt.figure(figsize=(14, 6.7))
left = fig.add_axes([0.235, 0.315, 0.295, 0.455])
right = fig.add_axes([0.645, 0.315, 0.305, 0.455])
fig.text(0.055, 0.93, "양방향 교정의 첫 시도 이득 G", color=INK, fontsize=23, weight="bold")
fig.text(0.055, 0.875, "최종 평가 7,202문항 · RouterBench와 xRouteBench를 같은 비중으로 평균 · 재학습 모델은 seed 3개 평균",
         color=GRAY, fontsize=12)

def axis_style(ax):
    ax.set_ylim(-0.6, 2.65)
    ax.tick_params(axis="both", length=0, pad=10, colors=GRAY, labelsize=11)
    ax.xaxis.set_major_locator(MultipleLocator(0.005))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if abs(x) < 1e-10 else f"{x:+.3f}"))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.axvline(0, color="#989daa", ls=(0, (4, 4)), lw=1.2)
    for s in ax.spines.values():
        s.set_visible(False)

for ax in (left, right):
    axis_style(ax)
left.set_xlim(-0.013, 0.009)
right.set_xlim(-0.0075, 0.0165)
left.set_yticks([2, 1, 0], LABELS, fontsize=12, color=INK)
left.tick_params(axis="y", pad=19)
right.set_yticks([])
left.set_title("(a) 같은 비용의 고정 정책 대비", loc="left", pad=17, color=INK, fontsize=14, weight="bold")
right.set_title("(b) 재학습하지 않은 모델 대비", loc="left", pad=17, color=INK, fontsize=14, weight="bold")
left.set_xlabel("G = A − H(C)", labelpad=13, color=INK, fontsize=13)
right.set_xlabel("ΔG = 재학습 모델의 G − 재학습 안 한 모델의 G", labelpad=13, color=INK, fontsize=13)

for arm, row in summary.iterrows():
    yy = Y[arm]
    color = COLORS[arm]
    left.hlines(yy, row.G_lo, row.G_hi, lw=2.3, color=color)
    left.plot(row.G, yy, "o", ms=8, color=color, zorder=3)
    left.text(row.G, yy + 0.22, f"{row.G:+.4f}", ha="center", va="bottom", color=color, fontsize=12, weight="bold")

right.text(0, Y["F0"], "기준 모델 (ΔG = 0)", ha="center", va="center", color=GRAY, fontsize=11,
           bbox={"facecolor": "white", "edgecolor": "none", "pad": 4})
for arm, row in delta.iterrows():
    yy = Y[arm]
    color = COLORS[arm]
    right.hlines(yy, row.lo, row.hi, lw=2.3, color=color)
    right.plot(row.delta_G, yy, "o", ms=8, color=color, zorder=3)
    right.text(row.delta_G, yy + 0.22, f"{row.delta_G:+.4f}", ha="center", va="bottom", color=color, fontsize=12, weight="bold")

fig.text(0.055, 0.14, "두 방식 모두 ΔG의 95% 구간이 0을 포함해, 기존 모델보다 개선됐다고 판단하기 어렵다.",
         fontsize=14, weight="bold", color=INK)
fig.text(0.055, 0.09, "점: 평균 추정치   선: 95% bootstrap 구간   점선: 이득 0   |   0.01 = 첫 시도 해결률 1%p 차이",
         color=GRAY, fontsize=10.8)
fig.text(0.055, 0.052, "질문 묶음 단위 2,000회 재표집 · 원천별 재표집을 모든 모델·seed가 공유 · 구간은 문항 표본의 불확실성을 반영",
         color=GRAY, fontsize=10.2)

stem = "fig5-11-bidirectional-gain"
for suffix in ("png", "svg"):
    fig.savefig(OUT / f"{stem}.{suffix}", dpi=240)
plt.close(fig)

rows = summary.reset_index()[["arm", "G", "G_lo", "G_hi", "A", "C", "H", "G_seed_min", "G_seed_max", "seeds"]]
rows = rows.merge(delta.reset_index()[["arm", "delta_G", "lo", "hi"]], how="left", on="arm")
rows.to_csv(OUT / "figure-data.csv", index=False)
provenance = {
    "source": str(SOURCE), "source_manifest_sha256": sha256(SOURCE / "manifest.json"),
    "source_sha256": {n: sha256(SOURCE / n) for n in ("gain-summary.csv", "delta-vs-f0.csv")},
    "selection": {"set": "test", "arms": ["F0", "F4", "F5"], "n": 7202},
    "metric_recomputed": False, "bootstrap": manifest["bootstrap"],
    "outputs": {p.name: sha256(p) for p in OUT.iterdir() if p.suffix in (".png", ".svg", ".csv")},
}
(OUT / "figure-manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
print(OUT / f"{stem}.png")
print(rows.to_string(index=False))
