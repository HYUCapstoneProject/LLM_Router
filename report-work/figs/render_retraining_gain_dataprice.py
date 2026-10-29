"""Render figures 6-11, 6-13 and 6-14 from the data-price retraining G (2026-10-22).

Reads notebook 06's output (tier price = mean training-data cost of each source's representative
models, high = 1). The 1:3:5 figures from 2026-10-21 are left untouched in retraining-gain-20261021/.
No F codes appear in the figures.
"""

import hashlib
import json
import shutil
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
SOURCE = DRIVE / "retraining-gain-f6-dataprice-20261022"
OUT = ROOT / "report-work/figs/retraining-gain-dataprice-20261022"
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((SOURCE / "manifest.json").read_text())
assert manifest["price_basis"] == "data"
for name in ("gain-summary.csv", "delta-vs-f0.csv"):
    assert sha(SOURCE / name) == manifest["output_sha256"][name], name
    shutil.copy2(SOURCE / name, OUT / name)
summary = pd.read_csv(OUT / "gain-summary.csv").set_index(["set", "arm"])
delta = pd.read_csv(OUT / "delta-vs-f0.csv").set_index(["set", "arm"])
for (set_name, arm), row in delta.iterrows():
    assert abs(row.delta_G - (summary.loc[(set_name, arm), "G"] - summary.loc[(set_name, "F0"), "G"])) < 1e-12

for weight in ("Regular", "Bold"):
    font_manager.fontManager.addfont(Path.home() / "Library/Fonts" / f"Pretendard-{weight}.ttf")
plt.rcParams.update({"font.family": "Pretendard", "font.size": 12, "axes.unicode_minus": False,
                     "svg.fonttype": "path", "figure.facecolor": "white", "savefig.facecolor": "white"})
INK, GRAY, GRID = "#25272c", "#686d78", "#e2e4e9"
PRICE_NOTE = "단계 비용 = 대표 모델의 학습 데이터 평균 가격 (RouterBench 1 : 4.0 : 56.7, xRouteBench 1 : 3.4 : 30.5)"
fmt = FuncFormatter(lambda x, _: "0" if abs(x) < 1e-10 else f"{x:+.2f}")


def style(ax, ymax):
    ax.set_ylim(-0.55, ymax + 0.65)
    ax.tick_params(axis="both", length=0, pad=10, colors=GRAY, labelsize=11)
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.xaxis.set_major_formatter(fmt)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.axvline(0, color="#989daa", ls=(0, (4, 4)), lw=1.2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def two_panel(set_name, arms, labels, colors, title, subtitle, conclusion, notes, xlim_left, xlim_right, stem, height):
    rows = summary.loc[set_name].loc[arms]
    ys = {arm: len(arms) - 1 - i for i, arm in enumerate(arms)}
    fig = plt.figure(figsize=(14, height))
    left = fig.add_axes([0.235, 0.33, 0.295, 0.44])
    right = fig.add_axes([0.645, 0.33, 0.305, 0.44])
    fig.text(0.055, 0.93, title, color=INK, fontsize=23, weight="bold")
    fig.text(0.055, 0.882, subtitle, color=GRAY, fontsize=12)
    fig.text(0.055, 0.848, PRICE_NOTE, color=GRAY, fontsize=12)
    for ax in (left, right):
        style(ax, len(arms) - 1)
    left.set_xlim(*xlim_left)
    right.set_xlim(*xlim_right)
    left.set_yticks([ys[a] for a in arms], labels, fontsize=12, color=INK)
    left.tick_params(axis="y", pad=19)
    right.set_yticks([])
    left.set_title("(a) 같은 비용의 고정 정책 대비", loc="left", pad=17, color=INK, fontsize=14, weight="bold")
    right.set_title("(b) 재학습하지 않은 모델 대비", loc="left", pad=17, color=INK, fontsize=14, weight="bold")
    left.set_xlabel("G = A − H(C)", labelpad=13, color=INK, fontsize=13)
    right.set_xlabel("ΔG = 재학습 모델의 G − 재학습 안 한 모델의 G", labelpad=13, color=INK, fontsize=13)
    for arm, row in rows.iterrows():
        yy, color = ys[arm], colors[arm]
        left.hlines(yy, row.G_lo, row.G_hi, lw=2.3, color=color)
        left.plot(row.G, yy, "o", ms=8, color=color, zorder=3)
        left.text(row.G, yy + 0.22, f"{row.G:+.3f}", ha="center", va="bottom", color=color, fontsize=12, weight="bold")
    right.text(sum(xlim_right) / 2, ys[arms[0]], "기준 모델 (ΔG = 0)", ha="center", va="center", color=GRAY, fontsize=11)
    for arm in arms[1:]:
        row = delta.loc[(set_name, arm)]
        yy, color = ys[arm], colors[arm]
        right.hlines(yy, row.lo, row.hi, lw=2.3, color=color)
        right.plot(row.delta_G, yy, "o", ms=8, color=color, zorder=3)
        right.text(row.delta_G, yy + 0.22, f"{row.delta_G:+.3f}", ha="center", va="bottom", color=color, fontsize=12,
                   weight="bold")
    fig.text(0.055, 0.19, conclusion, fontsize=14, weight="bold", color=INK)
    for i, note in enumerate(notes):
        fig.text(0.055, 0.13 - 0.038 * i, note, color=GRAY, fontsize=10.5)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=240)
    plt.close(fig)
    return stem


NOTE_MARKS = "점: 평균 추정치   선: 95% bootstrap 구간   점선: 이득 0   |   0.01 = 첫 시도 해결률 1%p 차이"
stems = []
stems.append(two_panel(
    "prequential", ["F0", "F1", "F2", "F3"], ["재학습 안 함", "새 피드백만", "과거 데이터 섞기", "easy 보충"],
    {"F0": GRAY, "F1": "#4c43c4", "F2": "#746bb8", "F3": "#81739e"},
    "상향 피드백 재학습의 첫 시도 이득 G",
    "다음 구간 평가 4,473문항 (구간 2~6) · 두 원천을 같은 비중으로 평균 · 재학습 모델은 seed 3개 평균",
    "low로 보내는 비율을 줄인 만큼 G가 올랐지만, 별도의 망각 기준을 통과하지 못해 채택하지 않았다.",
    [NOTE_MARKS,
     "low로 보낸 비율: 재학습 안 함 66% · 새 피드백만 0% · 과거 데이터 섞기 1% · easy 보충 38%",
     "구간 j에서 학습한 모델로 구간 j+1을 먼저 예측 · 질문 묶음 단위 2,000회 재표집 · 모든 모델·seed가 재표집 공유"],
    (-0.115, 0.045), (-0.01, 0.145), "fig6-11-feedback-gain-dataprice", 7.8))
stems.append(two_panel(
    "test", ["F0", "F4", "F5"], ["재학습 안 함", "양방향 교정\n(오답만)", "양방향 교정\n+ 비율 맞춤"],
    {"F0": GRAY, "F4": "#746bb8", "F5": "#4c43c4"},
    "양방향 교정의 첫 시도 이득 G",
    "최종 평가 7,202문항 · RouterBench와 xRouteBench를 같은 비중으로 평균 · 재학습 모델은 seed 3개 평균",
    "두 방식 모두 G가 올랐지만, 난이도 판단 정확도는 재학습하지 않은 모델보다 낮았다.",
    [NOTE_MARKS,
     "low로 보낸 비율: 재학습 안 함 64% · 오답만 교정 13% · 비율까지 맞춤 62%",
     "질문 묶음 단위 2,000회 재표집 · 원천별 재표집을 모든 모델·seed가 공유 · 구간은 문항 표본의 불확실성을 반영"],
    (-0.105, 0.015), (-0.01, 0.09), "fig6-13-bidirectional-gain-dataprice", 7.2))

# 6-14: every retraining method on one ΔG axis, with the low-tier share instead of the mean cost.
UPWARD, BIDIR, FREE = "#4c43c4", "#b5651d", "#1f8a70"
PANELS = [
    ("prequential", "다음 구간 4,473문항 (구간마다 갱신한 모델로 다음 구간 예측)",
     [("F1", "새 피드백만", UPWARD), ("F2", "과거 데이터 섞기", UPWARD), ("F3", "easy 보충", UPWARD),
      ("F6", "학습 없이 갱신", FREE), ("F6-esc", "학습 없이 갱신 (상향 기록만)", FREE)]),
    ("test", "최종 평가 세트 7,202문항 (마지막 모델)",
     [("F4", "양방향 교정 (오답만)", BIDIR), ("F5", "양방향 교정 + 비율 맞춤", BIDIR),
      ("F6", "학습 없이 갱신", FREE), ("F6-esc", "학습 없이 갱신 (상향 기록만)", FREE)]),
]
fig = plt.figure(figsize=(14, 9.8))
fig.text(0.055, 0.945, "G가 크게 오른 재학습은 모두 low 단계를 덜 쓰게 된 경우였다", color=INK, fontsize=22, weight="bold")
fig.text(0.055, 0.908, "재학습 안 한 모델 대비 첫 시도 이득 변화 ΔG · 두 원천을 같은 비중으로 평균 · 본체 재학습은 seed 3개 평균",
         color=GRAY, fontsize=12)
fig.text(0.055, 0.878, PRICE_NOTE, color=GRAY, fontsize=12)
for (set_name, title, rows), bottom in zip(PANELS, [0.49, 0.14]):
    n = len(rows)
    ax = fig.add_axes([0.30, bottom, 0.40, 0.065 * n])
    base_low = summary.loc[(set_name, "F0"), "low_rate"]
    for i, (arm, label, color) in enumerate(rows):
        y = n - 1 - i
        row = delta.loc[(set_name, arm)]
        ax.hlines(y, row.lo, row.hi, color=color, lw=2.3)
        ax.plot(row.delta_G, y, "o", ms=8, color=color, zorder=3)
        ax.text(row.hi + 0.003, y, f"{row.delta_G:+.3f}", va="center", color=color, fontsize=12, weight="bold")
        ax.text(-0.04, y, label, ha="right", va="center", color=INK, fontsize=12.5, transform=ax.transData)
        low = summary.loc[(set_name, arm), "low_rate"]
        strong = base_low - low > 0.2
        ax.text(1.06, y, f"{base_low:.0%} → {low:.0%}", ha="left", va="center", color=INK if strong else GRAY,
                fontsize=12, weight="bold" if strong else "normal", transform=ax.get_yaxis_transform())
    ax.text(1.06, n - 0.35, "low로 보낸 비율", ha="left", va="bottom", color=GRAY, fontsize=11,
            transform=ax.get_yaxis_transform())
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(-0.03, 0.15)
    ax.axvline(0, color="#989daa", ls=(0, (4, 4)), lw=1.2)
    ax.xaxis.set_major_locator(MultipleLocator(0.03))
    ax.xaxis.set_major_formatter(fmt)
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
fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.05, 0.035), ncol=3, frameon=False, fontsize=11.5,
           columnspacing=2.2)
fig.text(0.055, 0.015, "점: 추정치 · 선: 질문 묶음 bootstrap 2,000회 95% 구간 · 두 평가는 문항이 달라 위아래 값을 직접 비교하지 않는다. "
         "low 비율은 재학습 안 한 모델 → 각 방식", color=GRAY, fontsize=10)
stem = "fig6-14-all-retraining-gain-dataprice"
for ext in ("png", "svg"):
    fig.savefig(OUT / f"{stem}.{ext}", dpi=240)
plt.close(fig)
stems.append(stem)

(OUT / "figure-manifest.json").write_text(json.dumps({
    "source": str(SOURCE), "source_manifest_sha256": sha(SOURCE / "manifest.json"),
    "price_basis": manifest["price_basis"], "prices": manifest["prices"],
    "outputs": {f"{s}.{ext}": sha(OUT / f"{s}.{ext}") for s in stems for ext in ("png", "svg")}},
    ensure_ascii=False, indent=2))
for s in stems:
    print(OUT / f"{s}.png")
