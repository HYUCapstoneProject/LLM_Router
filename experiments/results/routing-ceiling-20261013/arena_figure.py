"""RouterArena: agreement with the official difficulty labels (no per-model outcomes or costs exist).

Reads the committed F5 final metrics (F0 and F1-F5 on the same 8,379 rows) and the official label
distribution; R1/R2 come from the 2026-09-24 diagnostic (lambda=0 action projected to a label).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
arena = pd.read_parquet(sys.argv[1])
arena = arena[arena.evaluation_split.eq("full")]
INK, MUTED, GRID, BLUE = "#0b0b0b", "#6f6e69", "#e4e3df", "#2a78d6"
SHADES = {"easy": "#a9c9ef", "medium": "#5b98e0", "hard": "#1d4f91"}  # one hue, light -> dark
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})

metrics = pd.read_csv(HERE.parent / "retraining-effect-f5-20261010/report-final/final-metrics.csv")
metrics = metrics[metrics.role.eq("routerarena")]
names = {"f0_frozen": "F0 (R0 1.0)", "f5_matched_prior": "F5 matched prior", "f4_bidirectional": "F4 bidirectional",
         "f3_balanced_easy": "F3 escalation + easy", "f2_replay": "F2 escalation + replay",
         "f1_cycle_only": "F1 escalation only"}
rows = []
for arm, label in names.items():
    g = metrics[metrics.arm.eq(arm)]
    rows.append({"method": label, **{f"{k}_{s}": getattr(g[k], s)() for k in ("accuracy", "macro_f1") for s in ("mean", "min", "max")},
                 **{f"predicted_{l}": g[f"predicted_{l}_share"].mean() for l in ("easy", "medium", "hard")}})
for label, acc, f1 in (("R1* (lambda=0 action)", .2280, .1754), ("R2* (lambda=0 action)", .2136, .1927)):
    rows.append({"method": label, **{f"accuracy_{s}": acc for s in ("mean", "min", "max")},
                 **{f"macro_f1_{s}": f1 for s in ("mean", "min", "max")}})
table = pd.DataFrame(rows)
official = arena.difficulty_label.value_counts(normalize=True).reindex(["easy", "medium", "hard"])
majority_acc = float(official.max())
easy_precision = official["easy"]
majority_f1 = 2 * easy_precision / (easy_precision + 1) / 3
table.to_csv(HERE / "arena-alignment.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.2]})
order = list(range(len(table)))[::-1]
for ax, key, ref, ref_name in ((axes[0], "accuracy", majority_acc, "always 'easy'"),
                              (axes[1], "macro_f1", majority_f1, "always 'easy'")):
    for y, (_, r) in zip(order, table.iterrows()):
        color = MUTED if r.method.endswith("action)") else BLUE
        ax.hlines(y, r[f"{key}_min"], r[f"{key}_max"], color=color, lw=3, alpha=.4)
        ax.scatter(r[f"{key}_mean"], y, color=color, s=46, zorder=3)
        ax.annotate(f"{r[f'{key}_mean']:.3f}", (r[f"{key}_max"], y), textcoords="offset points", xytext=(6, -3),
                    fontsize=8, color=INK)
    ax.axvline(ref, color=INK, ls="--", lw=1.2)
    ax.annotate(f"{ref_name} {ref:.3f}", (ref, len(table) - 0.45), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=INK, va="center")
    ax.set_ylim(-0.6, len(table) - 0.2)
    ax.set_yticks(order, table.method)
    ax.set_xlim(0, 0.75)
    ax.set_xlabel("Accuracy vs official label" if key == "accuracy" else "Macro-F1 vs official label")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
axes[1].set_yticklabels([])
share = table.dropna(subset=["predicted_easy"])
labels = ["Official labels"] + list(share.method)
data = [official.to_numpy()] + [r[["predicted_easy", "predicted_medium", "predicted_hard"]].to_numpy(float)
                                for _, r in share.iterrows()]
for y, (name, values) in enumerate(zip(labels, data)):
    left = 0
    for level, v in zip(("easy", "medium", "hard"), values):
        axes[2].barh(len(labels) - 1 - y, v, left=left, color=SHADES[level], edgecolor="white", linewidth=2,
                     label=level if y == 0 else None, height=0.7)
        if v >= 0.08:
            axes[2].text(left + v / 2, len(labels) - 1 - y, f"{v:.0%}", ha="center", va="center", fontsize=7.5,
                         color="white" if level != "easy" else INK)
        left += v
axes[2].set_yticks(range(len(labels))[::-1], labels)
axes[2].set_xlim(0, 1)
axes[2].set_xlabel("Share of the 8,379 questions")
axes[2].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8)
fig.suptitle("RouterArena (8,379, official difficulty labels; no outcome or cost records, so only label agreement)"
             "\nNo retraining arm beats F0, and none beats always answering 'easy' on accuracy", fontsize=10, color=INK)
fig.text(0.01, 0.01, "Bars: min-max over 3 seeds. *R1/R2: cost-free candidate choice projected to a label "
         "(2026-09-24 diagnostic), not a difficulty classifier.", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(HERE / "arena-alignment.png", dpi=170)
print(HERE / "arena-alignment.png")
