"""Render the all-method retraining comparison by first-try gain (2026-10-21).

Reads notebook 05's output (F0-F5 identical to notebook 04, plus the training-free
arms). Two evaluation sets are shown separately because the methods were evaluated
on different items: per-window methods on the next-window 4,473 items, final models
on the 7,202-item final evaluation set. No F codes appear in the figure.
"""

import hashlib
import json
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DRIVE = next(p / "LLM_Router" for p in Path.home().joinpath("Google Drive").iterdir()
             if unicodedata.normalize("NFC", p.name) == "내 드라이브")
SOURCE = DRIVE / "retraining-gain-f6-20261021"
OUT = ROOT / "report-work/figs/retraining-gain-20261021"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((SOURCE / "manifest.json").read_text())
for name in ("gain-summary.csv", "delta-vs-f0.csv"):
    assert sha(SOURCE / name) == manifest["output_sha256"][name], name
summary = pd.read_csv(SOURCE / "gain-summary.csv").set_index(["set", "arm"])
delta = pd.read_csv(SOURCE / "delta-vs-f0.csv").set_index(["set", "arm"])

UPWARD, BIDIR, FREE = "#4c43c4", "#b5651d", "#1f8a70"
PANELS = [
    ("prequential", "다음 구간 4,473문항 (구간마다 갱신한 모델로 다음 구간 예측)",
     [("F1", "새 피드백만", UPWARD), ("F2", "과거 데이터 섞기", UPWARD), ("F3", "easy 보충", UPWARD),
      ("F6", "학습 없이 갱신", FREE), ("F6-esc", "학습 없이 갱신 (상향 기록만)", FREE)]),
    ("test", "최종 평가 세트 7,202문항 (마지막 모델)",
     [("F4", "양방향 교정 (오답만)", BIDIR), ("F5", "양방향 교정 + 비율 맞춤", BIDIR),
      ("F6", "학습 없이 갱신", FREE), ("F6-esc", "학습 없이 갱신 (상향 기록만)", FREE)]),
]

for weight in ("Regular", "Bold"):
    font_manager.fontManager.addfont(Path.home() / "Library/Fonts" / f"Pretendard-{weight}.ttf")
plt.rcParams.update({"font.family": "Pretendard", "font.size": 12, "axes.unicode_minus": False,
                     "svg.fonttype": "path", "figure.facecolor": "white", "savefig.facecolor": "white"})
INK, GRAY, GRID = "#25272c", "#686d78", "#e2e4e9"

fig = plt.figure(figsize=(14, 9.6))
fig.text(0.055, 0.945, "G가 오른 재학습은 모두 비싼 단계로 옮겨 간 경우였다", color=INK, fontsize=22, weight="bold")
fig.text(0.055, 0.905, "재학습 안 한 모델 대비 첫 시도 이득 변화 ΔG · 두 원천을 같은 비중으로 평균 · 본체 재학습은 seed 3개 평균",
         color=GRAY, fontsize=12)
heights = [len(rows) for _, _, rows in PANELS]
bottoms = [0.505, 0.155]
for (set_name, title, rows), bottom, n in zip(PANELS, bottoms, heights):
    ax = fig.add_axes([0.30, bottom, 0.40, 0.065 * n])
    base_c = summary.loc[(set_name, "F0"), "C"]
    for i, (arm, label, color) in enumerate(rows):
        y = n - 1 - i
        row = delta.loc[(set_name, arm)]
        ax.hlines(y, row.lo, row.hi, color=color, lw=2.3)
        ax.plot(row.delta_G, y, "o", ms=8, color=color, zorder=3)
        ax.text(row.hi + 0.0015, y, f"{row.delta_G:+.3f}", va="center", color=color, fontsize=12, weight="bold")
        ax.text(-0.026, y, label, ha="right", va="center", color=INK, fontsize=12.5, transform=ax.transData)
        cost = summary.loc[(set_name, arm), "C"]
        ax.text(1.06, y, f"{base_c:.2f} → {cost:.2f}", ha="left", va="center", color=INK if cost > base_c * 1.2 else GRAY,
                fontsize=12, weight="bold" if cost > base_c * 1.2 else "normal", transform=ax.get_yaxis_transform())
    ax.text(1.06, n - 0.35, "평균 단계 비용", ha="left", va="bottom", color=GRAY, fontsize=11,
            transform=ax.get_yaxis_transform())
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(-0.02, 0.06)
    ax.axvline(0, color="#989daa", ls=(0, (4, 4)), lw=1.2)
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if abs(x) < 1e-10 else f"{x:+.2f}"))
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=8, colors=GRAY, labelsize=11)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", x=-0.6, pad=14, color=INK, fontsize=13.5, weight="bold")
ax.set_xlabel("ΔG = 각 방식의 G − 재학습 안 한 모델의 G", labelpad=12, color=INK, fontsize=12.5)
handles = [Line2D([], [], marker="o", color=c, lw=2.3, ms=8, label=l)
           for c, l in ((UPWARD, "상향 피드백으로 본체 재학습"), (BIDIR, "양방향 정답으로 본체 재학습"),
                        (FREE, "본체는 고정, 문턱만 갱신"))]
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.05, 0.03), ncol=3, frameon=False, fontsize=11.5,
           columnspacing=2.2)
fig.text(0.055, 0.015, "점: 추정치 · 선: 질문 묶음 bootstrap 2,000회 95% 구간 · 두 평가는 문항이 달라 위아래 값을 직접 비교하지 않는다. "
         "재학습 안 한 모델의 평균 단계 비용은 화살표 왼쪽 값",
         color=GRAY, fontsize=10)
stem = "fig5-13-all-retraining-gain"
for ext in ("png", "svg"):
    fig.savefig(OUT / f"{stem}.{ext}", dpi=240)
plt.close(fig)
(OUT / "all-retraining-figure-manifest.json").write_text(json.dumps({
    "source": str(SOURCE), "source_manifest_sha256": sha(SOURCE / "manifest.json"),
    "panels": {s: [a for a, _, _ in rows] for s, _, rows in PANELS},
    "outputs": {f"{stem}.{ext}": sha(OUT / f"{stem}.{ext}") for ext in ("png", "svg")}},
    ensure_ascii=False, indent=2))
print(OUT / f"{stem}.png")
