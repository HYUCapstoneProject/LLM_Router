"""Render section 5.3.1 from the completed Colab estimates and paired intervals."""

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
DRIVE = next(p / "LLM_Router" for p in Path.home().joinpath("Google Drive").iterdir()
             if unicodedata.normalize("NFC", p.name) == "내 드라이브")
SOURCE = DRIVE / "retraining-gain-20261021"
OUT = ROOT / "report-work/figs/retraining-gain-20261021"
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((SOURCE / "manifest.json").read_text())
for name in ("gain-summary.csv", "delta-vs-f0.csv"):
    assert sha(SOURCE / name) == manifest["output_sha256"][name]
    shutil.copy2(SOURCE / name, OUT / name)

summary = pd.read_csv(OUT / "gain-summary.csv")
summary = summary[summary["set"] == "prequential"].set_index("arm").loc[["F0", "F1", "F2", "F3"]]
delta = pd.read_csv(OUT / "delta-vs-f0.csv")
delta = delta[delta["set"] == "prequential"].set_index("arm").loc[["F1", "F2", "F3"]]
assert manifest["sets"]["prequential"] == 4473
assert list(summary.seeds) == [1, 3, 3, 3]
for arm in delta.index:
    assert abs(delta.loc[arm, "delta_G"] - (summary.loc[arm, "G"] - summary.loc["F0", "G"])) < 1e-12
    assert 0 < delta.loc[arm, "lo"] < delta.loc[arm, "hi"]

for weight in ("Regular", "Bold"):
    font_manager.fontManager.addfont(Path.home() / "Library/Fonts" / f"Pretendard-{weight}.ttf")
plt.rcParams.update({"font.family": "Pretendard", "font.size": 12, "axes.unicode_minus": False,
                     "svg.fonttype": "path", "figure.facecolor": "white", "savefig.facecolor": "white"})
INK, GRAY, GRID = "#25272c", "#686d78", "#e2e4e9"
COLORS = {"F0": GRAY, "F1": "#4c43c4", "F2": "#746bb8", "F3": "#81739e"}
LABELS = ["재학습 안 함", "새 피드백만", "과거 데이터 섞기", "easy 보충"]
Y = {"F0": 3, "F1": 2, "F2": 1, "F3": 0}
fig = plt.figure(figsize=(14, 7.5))
left = fig.add_axes([0.235, 0.31, 0.295, 0.46])
right = fig.add_axes([0.645, 0.31, 0.305, 0.46])
fig.text(0.055, 0.93, "상향 피드백 재학습의 첫 시도 이득 G", color=INK, fontsize=23, weight="bold")
fig.text(0.055, 0.877, "다음 구간 평가 4,473문항 (구간 2~6) · 두 원천을 같은 비중으로 평균 · 재학습 모델은 seed 3개 평균",
         color=GRAY, fontsize=12)
for ax in (left, right):
    ax.set_ylim(-0.55, 3.65)
    ax.tick_params(axis="both", length=0, pad=10, colors=GRAY, labelsize=11)
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: "0" if abs(x) < 1e-10 else f"{x:+.2f}"))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.axvline(0, color="#989daa", ls=(0, (4, 4)), lw=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
left.set_xlim(-0.025, 0.037)
right.set_xlim(-0.007, 0.055)
left.set_yticks([3, 2, 1, 0], LABELS, fontsize=12, color=INK)
left.tick_params(axis="y", pad=19)
right.set_yticks([])
left.set_title("(a) 같은 비용의 고정 정책 대비", loc="left", pad=17, color=INK, fontsize=14, weight="bold")
right.set_title("(b) 재학습하지 않은 모델 대비", loc="left", pad=17, color=INK, fontsize=14, weight="bold")
left.set_xlabel("G = A − H(C)", labelpad=13, color=INK, fontsize=13)
right.set_xlabel("ΔG = 재학습 모델의 G − 재학습 안 한 모델의 G", labelpad=13, color=INK, fontsize=13)
for arm, row in summary.iterrows():
    yy, color = Y[arm], COLORS[arm]
    left.hlines(yy, row.G_lo, row.G_hi, lw=2.3, color=color)
    left.plot(row.G, yy, "o", ms=8, color=color, zorder=3)
    left.text(row.G, yy + 0.22, f"{row.G:+.4f}", ha="center", va="bottom", color=color, fontsize=12, weight="bold")
right.text(0.015, 3, "기준 모델 (ΔG = 0)", ha="center", va="center", color=GRAY, fontsize=11)
for arm, row in delta.iterrows():
    yy, color = Y[arm], COLORS[arm]
    right.hlines(yy, row.lo, row.hi, lw=2.3, color=color)
    right.plot(row.delta_G, yy, "o", ms=8, color=color, zorder=3)
    right.text(row.delta_G, yy + 0.22, f"{row.delta_G:+.4f}", ha="center", va="bottom", color=color, fontsize=12, weight="bold")
fig.text(0.055, 0.173, "세 방식 모두 재학습 안 한 모델보다 G가 개선됐지만, 별도의 망각 기준을 통과하지 못해 채택하지 않았다.",
         fontsize=14, weight="bold", color=INK)
fig.text(0.055, 0.115, "점: 평균 추정치   선: 95% bootstrap 구간   점선: 이득 0   |   0.01 = 첫 시도 해결률 1%p 차이",
         color=GRAY, fontsize=10.8)
fig.text(0.055, 0.077, "구간 j에서 학습한 모델로 구간 j+1을 먼저 예측 · 질문 묶음 단위 2,000회 재표집 · 모든 모델·seed가 재표집 공유",
         color=GRAY, fontsize=10.2)
fig.text(0.055, 0.042, "구간은 문항 표본의 불확실성을 반영한다. easy 보충의 G 구간은 0을 포함하지만 재학습 안 한 모델 대비 ΔG 구간은 양수다.",
         color=GRAY, fontsize=10.2)
stem = "fig5-10-feedback-gain"
for ext in ("png", "svg"):
    fig.savefig(OUT / f"{stem}.{ext}", dpi=240)
plt.close(fig)
rows = summary.reset_index()[["arm", "G", "G_lo", "G_hi", "A", "C", "H", "G_seed_min", "G_seed_max", "seeds"]]
rows = rows.merge(delta.reset_index()[["arm", "delta_G", "lo", "hi"]], how="left", on="arm")
rows.to_csv(OUT / "feedback-figure-data.csv", index=False)
provenance = {"source": str(SOURCE), "source_manifest_sha256": sha(SOURCE / "manifest.json"),
              "source_sha256": {n: sha(SOURCE / n) for n in ("gain-summary.csv", "delta-vs-f0.csv")},
              "selection": {"set": "prequential", "arms": ["F0", "F1", "F2", "F3"], "n": 4473},
              "metric_recomputed": False, "bootstrap": manifest["bootstrap"],
              "outputs": {f"{stem}.{ext}": sha(OUT / f"{stem}.{ext}") for ext in ("png", "svg")}}
(OUT / "feedback-figure-manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
print(OUT / f"{stem}.png")
