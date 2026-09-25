"""Create a compact, reproducible figure from verified qualification summaries."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).parent
r0 = json.loads((base / "r0/scores/r0-comparison.json").read_text())
a = json.loads((base / "r1/a-analysis/analysis.json").read_text())
g1 = json.loads((base / "r1/g1-analysis/analysis.json").read_text())
models = [("R0", r0["arena"]), ("A R1", a["datasets"]["arena"]),
          ("G1 R1-natural", g1["datasets"]["arena"])]
rows = []
for label, report in models:
    for scope, caption in [("evaluation_split=full", "All originals, n=8,379"),
                           ("evaluation_split=full/over512", ">512 tokens, n=748")]:
        result = report["slices"][scope]
        rows.append((label + "\n" + caption, result))
fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.25, 1]}, constrained_layout=True)
y = np.arange(len(rows))
for length, shift, color in [("512", -.15, "#64748b"), ("8192", .15, "#0f766e")]:
    vals = [100*r[1]["lengths"][length]["difficulty_slot_accuracy"] for r in rows]
    axes[0].barh(y + shift, vals, height=.28, color=color, label=f"{int(length):,} tokens")
    for pos, val in zip(y+shift, vals):
        axes[0].text(val+.5, pos, f"{val:.2f}%", va="center", fontsize=9)
axes[0].set_yticks(y, [r[0] for r in rows], fontsize=9)
axes[0].set_xlim(0, 70)
axes[0].set_xlabel("Historical difficulty-label agreement (%)")
axes[0].set_title("RouterArena original prompts only", loc="left", weight="bold")
axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2, frameon=False)
for i, (_, result) in enumerate(rows):
    ci = np.array(result["uncertainty"]["paired_delta_95_ci"])*100
    delta = result["primary_quality_delta"]*100
    axes[1].plot(ci, [i, i], color="#334155", linewidth=2)
    axes[1].scatter([delta], [i], color="#0f766e", s=35, zorder=3)
    axes[1].annotate(f"{delta:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]", (delta,i),
                     xytext=(0,10), textcoords="offset points", ha="center", fontsize=8)
axes[1].axvline(0, color="#94a3b8", linestyle="--")
axes[1].set_yticks(y, [""]*len(rows))
axes[1].set_xlabel("8,192 minus 512 (percentage points)")
axes[1].set_title("Paired change and 95% sampling CI", loc="left", weight="bold")
axes[1].margins(x=.25)
for ax in axes:
    ax.set_ylim(len(rows)-.5, -.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
fig.suptitle("Same checkpoints, longer inference context; no training or policy retuning", fontsize=13)
out = base / "report"
out.mkdir(exist_ok=True)
fig.savefig(out / "arena-length-comparison.png", dpi=180)
fig.savefig(out / "arena-length-comparison.svg")
plt.close(fig)
print(out / "arena-length-comparison.png")
