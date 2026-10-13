"""Figures for the routing-ceiling analysis (reads the CSVs written by ceiling_analysis.py)."""
import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
INK, MUTED, GRID = "#0b0b0b", "#6f6e69", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # reference categorical slots 1-2 (validated light mode)
SOURCES = {"routerbench": "RouterBench test (3,473)", "xroutebench": "xRouteBench test (3,729)"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})

curves = pd.read_csv(HERE / "curves.csv")
methods = pd.read_csv(HERE / "methods-resolution-cost.csv")
labels = pd.read_csv(HERE / "label-accuracy-ceiling.csv")


def frontier(points, xs):
    out = []
    for x in xs:
        best = max([q for c, q in points if c <= x + 1e-12])
        for (c1, q1), (c2, q2) in itertools.combinations(sorted(points), 2):
            if c1 <= x <= c2 and c2 > c1:
                best = max(best, q1 + (q2 - q1) * (x - c1) / (c2 - c1))
        out.append(best)
    return out


def upper(df, extend=False):
    """Upper envelope of (cost, quality) policy points; `extend` carries the best quality to cost 1."""
    df = df.sort_values("relative_cost")
    x, y = list(df.relative_cost), list(df.quality.cummax())
    if extend:
        x, y = x + [1.0], y + [y[-1]]
    return x, y


# Figure 1: first-try quality vs cost — where every learned router stops.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, (source, title) in zip(axes, SOURCES.items()):
    m = methods[methods.source.eq(source)].set_index("method")
    tiers = [(m.loc[f"always-{t}", "first_try_relative_cost"], m.loc[f"always-{t}", "first_try_quality"])
             for t in ("low", "medium", "high")]
    xs = [i / 200 for i in range(201)]
    ax.plot(xs, frontier(tiers, xs), color=MUTED, lw=2, label="mix of always-tier policies (no routing)")
    c = curves[curves.source.eq(source)]
    style = {"oracle (knows every tier's outcome)": dict(color=INK, ls="--", lw=2, label="oracle: knows each tier's outcome"),
             "peer bound (8 other LLMs' outcomes + task)": dict(color=INK, ls=":", lw=2, label="bound: knows 8 other LLMs' outcomes + task"),
             "R1 success logits": dict(color=BLUE, lw=2, label="R1 (lambda sweep)"),
             "R0-tau (cumulative threshold)": dict(color=ORANGE, lw=2, label="R0-tau (threshold sweep)")}
    for name, kw in style.items():
        part = c[c.method.eq(name)]
        if len(part):
            ax.plot(*upper(part, extend=name.startswith("oracle")), **kw)
    for name, marker, dx in (("F0 (operational R0 1.0)", "o", 0.02), ("F5 matched prior", "s", 0.02),
                             ("F4 bidirectional", "^", 0.02), ("R1 locked policy", "D", 0.02)):
        row = m.loc[name]
        ax.scatter(row.first_try_relative_cost, row.first_try_quality, s=64, marker=marker, color=INK,
                   edgecolor="white", linewidth=1.5, zorder=5)
        ax.annotate(name.replace(" (operational R0 1.0)", "").replace(" policy", ""),
                    (row.first_try_relative_cost + dx, row.first_try_quality - 0.006), fontsize=8, color=INK)
    for (x, y), t in zip(tiers, ("always-low", "always-medium", "always-high")):
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(4, -11), fontsize=8, color=MUTED)
    ax.set(title=title, xlabel="Relative cost of the first routed tier (always-low 0, always-high 1)",
           ylabel="First-try quality", xlim=(-0.02, 1.02))
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
axes[0].legend(loc="lower right", fontsize=8, frameon=False)
fig.suptitle("Learned routers stop just above the no-routing line; the oracle gap is not reachable from text",
             fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(HERE / "ceiling-routing.png", dpi=170)
plt.close(fig)

# Figure 2: label accuracy vs the label-noise ceiling.
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharex=True)
order = ["always the majority label", "task majority label (train)", "F4 bidirectional", "F5 matched prior",
         "F0 (operational R0 1.0)"]
for ax, (source, title) in zip(axes, SOURCES.items()):
    part = labels[labels.source.eq(source)].set_index("method")
    ceiling = part.loc["label-noise ceiling (Bayes, knows true solve rate)", "accuracy"]
    values = [part.loc[k, "accuracy"] for k in order]
    ax.hlines(range(len(order)), 0, values, color=GRID, lw=2)
    ax.scatter(values, range(len(order)), color=BLUE, s=46, zorder=3)
    for i, v in enumerate(values):
        ax.annotate(f"{v:.3f}", (v, i), textcoords="offset points", xytext=(6, -3), fontsize=8, color=INK)
    ax.axvline(ceiling, color=INK, ls="--", lw=1.5)
    ax.set_ylim(-0.5, len(order) - 0.2)
    ax.annotate(f"label-noise ceiling {ceiling:.3f}", (ceiling, len(order) - 0.45), xytext=(-4, 0),
                textcoords="offset points", ha="right", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(order)), [o.replace(" (operational R0 1.0)", " (R0 1.0)") for o in order])
    ax.set(title=title, xlabel="Accuracy against the original 3-class difficulty label", xlim=(0, 1))
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
fig.suptitle("Difficulty-label accuracy: R0 sits near the task-only baseline, about 0.2 below the label-noise ceiling",
             fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(HERE / "ceiling-label-accuracy.png", dpi=170)
plt.close(fig)

# Figure 3: first-try success, final resolution and the total cost of the escalation chain.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
show = ["always-low", "always-medium", "always-high", "F0 (operational R0 1.0)", "F5 matched prior",
        "F4 bidirectional", "R1 locked policy", "oracle direct (cheapest solving tier)"]
short = {"F0 (operational R0 1.0)": "F0", "F5 matched prior": "F5", "F4 bidirectional": "F4",
         "R1 locked policy": "R1", "oracle direct (cheapest solving tier)": "oracle",
         "always-low": "start low", "always-medium": "start medium", "always-high": "high only"}
for ax, (source, title) in zip(axes, SOURCES.items()):
    m = methods[methods.source.eq(source)].set_index("method")
    for name in show:
        row = m.loc[name]
        color = MUTED if name.startswith("always") else (INK if name.startswith("oracle") else BLUE)
        x = row.cascade_cost_vs_always_high
        ax.vlines(x, row.first_try_success, row.final_resolution, color=color, lw=1.5)
        ax.scatter(x, row.final_resolution, color=color, s=46, zorder=3)
        ax.scatter(x, row.first_try_success, facecolor="white", edgecolor=color, s=46, lw=1.5, zorder=3)
        # Labels sit beside the hollow first-try point, where methods are well separated.
        if name.startswith("oracle"):
            ax.annotate(short[name], (x, row.final_resolution), textcoords="offset points", xytext=(-7, 5),
                        ha="right", fontsize=8, color=INK)
        else:
            left = (source, name) in {("xroutebench", "F5 matched prior"), ("routerbench", "always-medium")}
            ax.annotate(short[name], (x, row.first_try_success), textcoords="offset points",
                        xytext=(-8, -3) if left else (7, -3), ha="right" if left else "left", fontsize=8, color=INK)
    ax.set(title=title, xlabel="Total cost of all calls, relative to sending everything to the high tier",
           xlim=(0.2, 1.05), ylim=(0.45, 0.95))
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Rate (filled: final resolution, hollow: first try)")
handles = [plt.Line2D([], [], marker="o", ls="", color=MUTED, label="final resolution after escalation"),
           plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=MUTED, label="first-try success"),
           plt.Line2D([], [], color=MUTED, label="no routing (fixed start tier)"),
           plt.Line2D([], [], color=BLUE, label="learned router"),
           plt.Line2D([], [], color=INK, label="oracle (cheapest solving tier)")]
axes[1].legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
fig.suptitle("With escalation, starting low already reaches oracle-level resolution; routing only buys first-try success",
             fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(HERE / "resolution-and-cost.png", dpi=170)
plt.close(fig)
print(HERE)
