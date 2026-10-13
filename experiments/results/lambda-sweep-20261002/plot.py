"""Render λ-sweep figures from curves.csv / chosen-points.csv."""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mpl-cache"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
EVAL, TUNE = "#2a78d6", "#9a9890"
CRITERIA = {  # label, color, marker
    "lambda0": ("λ=0", INK, "o"),
    "budget0.5": ("budget ≤ .5", "#eb6834", "s"),
    "retain95": ("95% of always-high", "#1baf7a", "D"),
    "retain98": ("98% of always-high", "#4a3aa7", "^"),
}
TIER_RAMP = ("#b7d3f2", "#5b9be3", "#1d4f91")  # low → high, one hue light→dark
SOURCES = ("routerbench", "xroutebench")

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def tradeoff(curves: pd.DataFrame, points: pd.DataFrame, experiment: str, eval_split: str) -> Path:
    arms = list(dict.fromkeys(curves.loc[curves.experiment.eq(experiment), "arm"]))
    fig, axes = plt.subplots(len(arms), 2, figsize=(10, 3.1 * len(arms)), squeeze=False)
    for r, arm in enumerate(arms):
        for c, source in enumerate(SOURCES):
            ax = axes[r][c]
            sel = curves[(curves.experiment == experiment) & (curves.arm == arm) & (curves.source == source)]
            tune = sel[sel.role == "tune"].sort_values("relative_cost")
            ev = sel[sel.split == eval_split].sort_values("relative_cost")
            ax.plot(tune.relative_cost, tune.quality, color=TUNE, lw=1.5, ls="--", label=f"tuning ({tune.split.iloc[0]})")
            ax.plot(ev.relative_cost, ev.quality, color=EVAL, lw=2, label=f"eval ({eval_split})")
            pts = points[(points.experiment == experiment) & (points.arm == arm) & (points.source == source) & (points.split == eval_split)]
            ref = pts.iloc[0]
            for x, y, name in (
                (0.0, ref.always_low_quality, "low"),
                (ref.always_medium_relative_cost, ref.always_medium_quality, "med"),
                (1.0, ref.always_high_quality, "high"),
            ):
                ax.scatter([x], [y], marker="x", s=36, color=MUTED, zorder=3)
                ax.annotate(f"always-{name}", (x, y), textcoords="offset points", xytext=(4, -10), fontsize=7, color=MUTED)
            for crit, (label, color, marker) in CRITERIA.items():
                row = pts[pts.criterion == crit]
                if row.empty or pd.isna(row.iloc[0]["lambda"]):
                    continue
                row = row.iloc[0]
                ax.scatter([row.at_relative_cost], [row.at_quality], s=70, marker=marker, color=color,
                           edgecolor=SURFACE, linewidth=2, zorder=4, label=f"{label}")
            ax.axvline(0.5, color=GRID, lw=1, zorder=0)
            ax.set_title(f"{arm} · {source}", fontsize=10, loc="left")
            ax.set_xlim(-0.03, 1.05)
            ax.set_xlabel("relative cost (0=always-low, 1=always-high)")
            ax.set_ylabel("mean quality")
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=TUNE, lw=1.5, ls="--", label="tuning split"),
               Line2D([], [], color=EVAL, lw=2, label=f"eval ({eval_split})"),
               Line2D([], [], color=MUTED, marker="x", ls="", label="always-low/med/high")]
    handles += [Line2D([], [], color=col, marker=m, ls="", markersize=7, label=lab)
                for lab, col, m in CRITERIA.values()]
    top = 1 - 0.32 / fig.get_figheight()
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, top), ncol=7, fontsize=7.5, frameon=False)
    fig.suptitle(f"{experiment}: quality vs cost as λ varies (λ chosen on tuning, markers on {eval_split})", x=0.01, y=0.998, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.62 / fig.get_figheight()))
    out = HERE / "figures" / f"{experiment.lower()}-{eval_split}-tradeoff.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def shares(curves: pd.DataFrame, points: pd.DataFrame, experiment: str, eval_split: str) -> Path:
    arms = list(dict.fromkeys(curves.loc[curves.experiment.eq(experiment), "arm"]))
    fig, axes = plt.subplots(len(arms), 2, figsize=(10, 2.6 * len(arms)), squeeze=False)
    for r, arm in enumerate(arms):
        for c, source in enumerate(SOURCES):
            ax = axes[r][c]
            ev = curves[(curves.experiment == experiment) & (curves.arm == arm) & (curves.source == source)
                        & (curves.split == eval_split) & (curves["lambda"] > 0)].sort_values("lambda")
            ax.stackplot(ev["lambda"], ev.low_rate, ev.medium_rate, ev.high_rate, colors=TIER_RAMP,
                         labels=["low→easy", "medium→medium", "high→hard"], edgecolor=SURFACE, linewidth=0.5)
            pts = points[(points.experiment == experiment) & (points.arm == arm) & (points.source == source)
                         & (points.split == eval_split)]
            for crit, (label, color, _) in CRITERIA.items():
                row = pts[pts.criterion == crit]
                if crit == "lambda0" or row.empty or pd.isna(row.iloc[0]["lambda"]) or row.iloc[0]["lambda"] <= 0:
                    continue
                ax.axvline(row.iloc[0]["lambda"], color=color, lw=2, label=label)
            ax.set_xscale("log")
            ax.set_ylim(0, 1)
            ax.set_xlim(1e-4, 100)
            ax.grid(False)
            ax.set_title(f"{arm} · {source}", fontsize=10, loc="left")
            ax.set_xlabel("λ (log)")
            ax.set_ylabel("share of routes")
            if r == 0 and c == 0:
                ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(0, 0.5))
    fig.suptitle(f"{experiment}: tier share vs λ on {eval_split}", x=0.01, ha="left")
    fig.tight_layout()
    out = HERE / "figures" / f"{experiment.lower()}-{eval_split}-tier-share.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    curves = pd.read_csv(HERE / "curves.csv")
    points = pd.read_csv(HERE / "chosen-points.csv")
    for experiment, split in (("A", "test"), ("G1", "selection"), ("G1", "task_holdout")):
        print(tradeoff(curves, points, experiment, split))
        print(shares(curves, points, experiment, split))


if __name__ == "__main__":
    main()
