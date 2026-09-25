#!/usr/bin/env python3
"""Render sealed G1 metrics as PNG/SVG scientific figures without inventing missing values."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ARMS = ("r1-natural", "r1-balanced", "r3-natural", "r3-balanced")
COLORS = dict(zip(ARMS, ("#0072B2", "#56B4E9", "#D55E00", "#CC79A7")))
METRIC = "source_macro_locked_policy_quality_auc"
BUDGET = "0.5000"
MARKERS = ("o", "s", "^", "D", "v", "P")


def read_optional(path):
    return json.loads(path.read_text()) if path.is_file() else None


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def arm_label(arm, selected):
    return arm.upper().replace("-", " · ") + (" *" if arm == selected else "")


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#b7bec8",
            "grid.color": "#e1e6ed",
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def save(fig, destination, name, description, outputs):
    for extension in ("png", "svg"):
        path = destination / f"{name}.{extension}"
        fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=0.2)
        outputs.append(
            {"file": path.name, "sha256": sha(path), "description": description}
        )
    plt.close(fig)


def missing(ax, title):
    ax.set_title(title, loc="left", fontweight="bold")
    ax.text(
        0.5,
        0.5,
        "N/A — evaluation not available",
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="#687384",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def overview(stages, selected, destination, outputs):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.1))
    columns = (
        (METRIC, "Locked-policy quality AUC", "Nominal budget integral"),
        (
            "default_source_macro_quality",
            "Quality at nominal budget 0.5",
            "Source-macro quality",
        ),
        (
            "default_source_macro_relative_cost",
            "Actual relative cost at budget 0.5",
            "Source-macro relative cost",
        ),
    )
    for row, (stage, reports) in enumerate(stages.items()):
        for col, (key, title, xlabel) in enumerate(columns):
            ax = axes[row, col]
            if not reports:
                missing(ax, f"{stage}\n{title}")
                continue
            values = [number(reports.get(arm, {}).get(key)) for arm in ARMS]
            finite = [v for v in values if np.isfinite(v)]
            limit = max(1.0, max(finite, default=1.0) * 1.18)
            for y, (arm, value) in enumerate(zip(ARMS, values)):
                feasible = reports.get(arm, {}).get("default_budget_feasible")
                if not np.isfinite(value):
                    ax.text(0.02, y, "N/A", va="center", color="#687384")
                    continue
                bar = ax.barh(
                    y,
                    value,
                    color=COLORS[arm],
                    height=0.6,
                    edgecolor="#182736" if arm == selected else "white",
                    linewidth=1.5 if arm == selected else 0.8,
                )
                if feasible is False:
                    bar[0].set_hatch("///")
                ax.text(
                    value + 0.015 * limit, y, f"{value:.3f}", va="center", fontsize=10
                )
            ax.set_yticks(range(len(ARMS)), [arm_label(a, selected) for a in ARMS])
            ax.set_ylim(len(ARMS) - 0.4, -0.6)
            ax.set_xlim(min(0, min(finite, default=0) * 1.1), limit)
            ax.set_title(f"{stage}\n{title}", loc="left", fontweight="bold")
            ax.set_xlabel(xlabel)
            ax.grid(axis="x", alpha=0.7)
            ax.set_axisbelow(True)
            if col == 2:
                ax.axvline(0.5, color="#b2182b", linestyle="--", linewidth=1.3)
                for y, arm in enumerate(ARMS):
                    if reports.get(arm, {}).get("default_budget_feasible") is False:
                        ax.text(
                            0.98,
                            y + 0.30,
                            "SOURCE BUDGET FAIL",
                            transform=ax.get_yaxis_transform(),
                            ha="right",
                            va="center",
                            fontsize=7.5,
                            color="#a41429",
                            bbox={
                                "facecolor": "white",
                                "alpha": 0.9,
                                "edgecolor": "none",
                            },
                        )
    fig.suptitle(
        "G1: locked policy performance and cost",
        x=0.04,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.015,
        "* Selection-locked arm; holdout never reselects. Hatching: default budget fails in at least one source.\n"
        "AUC integrates nominal budgets, not realized cost. Holdout AUC is N/A: xroutebench high quality is below low.\n"
        "Dashed 0.5 line shows the macro reference; feasibility is checked per source.",
        fontsize=10,
        color="#445164",
    )
    fig.tight_layout(rect=(0, 0.085, 1, 0.94), h_pad=2.3, w_pad=2.0)
    save(
        fig,
        destination,
        "01-performance-and-budget",
        "Selection and holdout metrics with budget failure flags",
        outputs,
    )


def cost_quality(stages, selected, destination, outputs):
    available = [(name, report) for name, report in stages.items() if report]
    if not available:
        return
    sources = sorted(
        {
            source
            for _, reports in available
            for report in reports.values()
            for source in report.get("per_source", {})
        }
    )
    fig, axes = plt.subplots(
        len(available),
        len(sources),
        figsize=(7.3 * len(sources), 4.9 * len(available)),
        squeeze=False,
    )
    for r, (stage, reports) in enumerate(available):
        for c, source in enumerate(sources):
            ax = axes[r, c]
            baseline = None
            budget_keys = set()
            for arm in ARMS:
                values = reports.get(arm, {}).get("per_source", {}).get(source)
                if not values:
                    continue
                baseline = baseline or values.get("always")
                points = sorted(
                    values.get("budgets", {}).items(), key=lambda item: float(item[0])
                )
                budget_keys.update(key for key, _ in points)
                xs = [number(point.get("mean_actual_cost")) for _, point in points]
                ys = [number(point.get("mean_quality")) for _, point in points]
                ax.plot(
                    xs,
                    ys,
                    color=COLORS[arm],
                    alpha=0.8,
                    linewidth=2 if arm == selected else 1.3,
                )
                for i, ((key, point), x, y) in enumerate(zip(points, xs, ys)):
                    if not np.isfinite(x) or not np.isfinite(y):
                        continue
                    violation = point.get("budget_exceeded") is True
                    ax.scatter(
                        x,
                        y,
                        s=90 if key == BUDGET else 48,
                        marker=MARKERS[i % len(MARKERS)],
                        facecolor=COLORS[arm],
                        edgecolor="#b2182b" if violation else "white",
                        linewidth=2.1 if violation else 0.6,
                        zorder=4,
                    )
            if baseline:
                for tier, point in baseline.items():
                    x, y = (
                        number(point.get("actual_cost")),
                        number(point.get("quality")),
                    )
                    if np.isfinite(x) and np.isfinite(y):
                        ax.scatter(x, y, marker="X", s=95, color="#303945", zorder=5)
                        ax.annotate(
                            f"always-{tier}",
                            (x, y),
                            xytext=(6, 8),
                            textcoords="offset points",
                            fontsize=9,
                        )
            ax.set_title(f"{stage} · {source}", loc="left", fontweight="bold")
            ax.set_xlabel("Mean actual benchmark cost per request")
            ax.set_ylabel("Mean quality (benchmark score)")
            ax.ticklabel_format(
                axis="x", style="sci", scilimits=(-2, 2), useMathText=True
            )
            ax.grid(alpha=0.65)
            ax.margins(x=0.16, y=0.2)
    arm_handles = [
        Line2D([0], [0], color=COLORS[a], lw=2, label=arm_label(a, selected))
        for a in ARMS
    ]
    keys = sorted(budget_keys, key=float)
    budget_handles = [
        Line2D(
            [0],
            [0],
            color="#596575",
            marker=MARKERS[i % len(MARKERS)],
            linestyle="none",
            label=f"b={float(key):g}",
        )
        for i, key in enumerate(keys)
    ]
    fig.legend(
        handles=arm_handles + budget_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.suptitle(
        "Realized cost–quality points under calibration-locked policies",
        x=0.04,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.005,
        "Lines join nominal-budget settings; they are not reoptimized Pareto frontiers. Red outline: realized cost exceeds that point's nominal budget.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.93), h_pad=2.4, w_pad=2)
    save(
        fig,
        destination,
        "02-realized-cost-quality",
        "Per-source actual cost and quality with fixed budgets and always baselines",
        outputs,
    )


def task_heatmap(holdout, selected, destination, outputs):
    if not holdout:
        return
    tasks = sorted(
        {
            (source, task)
            for report in holdout.values()
            for source, values in report.get("per_source", {}).items()
            for task in values.get("budgets", {}).get(BUDGET, {}).get("per_task", {})
        }
    )
    if not tasks:
        return
    matrix = np.full((len(tasks), len(ARMS)), np.nan)
    counts = {}
    for row, (source, task) in enumerate(tasks):
        for col, arm in enumerate(ARMS):
            point = (
                holdout.get(arm, {})
                .get("per_source", {})
                .get(source, {})
                .get("budgets", {})
                .get(BUDGET, {})
                .get("per_task", {})
                .get(task, {})
            )
            matrix[row, col] = number(point.get("mean_quality"))
            if point.get("rows") is not None:
                counts[(source, task)] = point["rows"]
    fig, ax = plt.subplots(figsize=(11.4, max(4.5, len(tasks) * 0.66 + 2.4)))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#e8ebef")
    im = ax.imshow(
        np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=1, aspect="auto"
    )
    ax.set_xticks(range(len(ARMS)), [arm_label(a, selected) for a in ARMS])
    ax.set_yticks(
        range(len(tasks)),
        [
            f"{source} / {task}  (n={counts.get((source, task), 'N/A')})"
            for source, task in tasks
        ],
    )
    for row in range(len(tasks)):
        for col in range(len(ARMS)):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                f"{value:.3f}" if np.isfinite(value) else "N/A",
                ha="center",
                va="center",
                color="white" if np.isfinite(value) and value > 0.6 else "#182736",
                fontweight="bold",
            )
    ax.set_title(
        "Task-disjoint holdout quality at nominal budget 0.5",
        loc="left",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="Mean benchmark quality")
    fig.text(
        0.04,
        0.025,
        "Task/source scores retain their original benchmark meaning. Quality alone does not establish equal-cost superiority.\n* Selection-locked arm; no holdout reselection. Missing cells remain N/A.",
        fontsize=9,
        color="#445164",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(
        fig,
        destination,
        "03-holdout-task-quality",
        "Holdout task quality heatmap, fixed nominal budget 0.5",
        outputs,
    )


def pool_heatmap(pool, destination, outputs):
    if not pool:
        return
    rows = []
    for source, source_data in pool.get("sources", {}).items():
        rows.append((source, "ALL", source_data["overall"]))
        rows.extend(
            (source, task, values)
            for task, values in source_data.get("tasks", {}).items()
        )
    if not rows:
        return
    slots = ("low", "medium", "high")
    matrix = np.array(
        [
            [
                number(
                    values.get("candidates", {})
                    .get(slot, {})
                    .get("quality", {})
                    .get("mean")
                )
                for slot in slots
            ]
            for _, _, values in rows
        ]
    )
    fig, ax = plt.subplots(figsize=(12.5, 7.6))
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#e8ebef")
    im = ax.imshow(
        np.ma.masked_invalid(matrix), cmap=cmap, vmin=0, vmax=1, aspect="auto"
    )
    ax.set_xticks(range(3), ["Low candidate", "Medium candidate", "High candidate"])
    ax.set_yticks(
        range(len(rows)),
        [f"{source} / {task}  (n={values['rows']})" for source, task, values in rows],
    )
    for row in range(len(rows)):
        for col in range(3):
            value = matrix[row, col]
            ax.text(
                col,
                row,
                f"{value:.4f}" if np.isfinite(value) else "N/A",
                ha="center",
                va="center",
                color="white" if np.isfinite(value) and value > 0.6 else "#182736",
                fontweight="bold",
            )
        if row and rows[row][0] != rows[row - 1][0]:
            ax.axhline(row - 0.5, color="white", linewidth=4)
    ax.set_title(
        "Actual candidate quality on the task holdout",
        loc="left",
        fontsize=17,
        fontweight="bold",
        pad=18,
    )
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="Mean benchmark quality")
    fig.text(
        0.04,
        0.025,
        "Candidate slots use different model identities in each source. High does not guarantee higher held-out quality.\nPost-selection diagnostic: no candidate-pool changes or policy reselection. Values are benchmark outcomes, not predicted route shares.",
        fontsize=9,
        color="#445164",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save(
        fig,
        destination,
        "05-holdout-candidate-quality",
        "Actual source/task candidate quality; post-selection diagnostic",
        outputs,
    )


def arena_plot(arena, selected, destination, outputs):
    if not arena:
        return
    reports = arena.get("reports", {})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (section, key, title) in zip(
        axes,
        (
            ("full", "accuracy", "Difficulty-label alignment: accuracy"),
            (
                "full",
                "macro_f1_fixed_three_labels",
                "Difficulty-label alignment: macro-F1",
            ),
            (
                "paired_robustness",
                "same_action_rate",
                "Paired perturbations: same action",
            ),
        ),
    ):
        for y, arm in enumerate(ARMS):
            value = number(reports.get(arm, {}).get(section, {}).get(key))
            if np.isfinite(value):
                ax.barh(y, value, color=COLORS[arm], height=0.58)
                ax.text(value + 0.015, y, f"{value:.3f}", va="center")
            else:
                ax.text(0.03, y, "N/A", va="center")
        ax.set_yticks(range(4), [arm_label(a, selected) for a in ARMS])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.6)
        ax.set_axisbelow(True)
    fig.suptitle(
        "RouterArena: action-slot stress diagnostic (fixed lambda = 0)",
        x=0.03,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.025,
        "Not routing quality/cost, not an Arena Score, and not a new selection rule. Same-action stability can be inflated by constant predictions.\n* Winner remains locked from G1 selection; these are action-slot projections, not trained difficulty-class probabilities.",
        fontsize=9,
        color="#445164",
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.88), w_pad=2)
    save(
        fig,
        destination,
        "04-arena-stress",
        "Arena action-slot alignment and paired stability; no routing-quality claim",
        outputs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_dir.resolve()
    paths = {
        "selection": root / "selection-lock/manifest.json",
        "holdout": root / "holdout-report/manifest.json",
        "arena": root / "arena-stress/results/summary.json",
        "pool": root / "evidence/holdout-pool-analysis.json",
    }
    data = {key: read_optional(path) for key, path in paths.items()}
    if data["selection"] is None:
        parser.error("selection-lock/manifest.json is required")
    selected = data["selection"]["selected_arm"]
    if data["holdout"] and data["holdout"]["selected_arm"] != selected:
        raise ValueError("Holdout and selection identify different locked winners")
    if (
        data["arena"]
        and data["arena"].get("protocol", {}).get("selected_arm_unchanged") != selected
    ):
        raise ValueError("Arena diagnostic identifies a different locked winner")
    stages = {
        "Selection": data["selection"]["selection_reports"],
        "Task holdout": data["holdout"]["reports"] if data["holdout"] else None,
    }
    destination = root / "figures"
    destination.mkdir(exist_ok=True)
    style()
    outputs = []
    overview(stages, selected, destination, outputs)
    cost_quality(stages, selected, destination, outputs)
    task_heatmap(stages["Task holdout"], selected, destination, outputs)
    arena_plot(data["arena"], selected, destination, outputs)
    pool_heatmap(data["pool"], destination, outputs)
    manifest = {
        "schema_version": "g1-report-figures-v1",
        "selected_arm": selected,
        "inputs": {
            key: {"path": str(path), "sha256": sha(path)} if path.exists() else None
            for key, path in paths.items()
        },
        "missing_values_rendered_as": "N/A or omitted figure; never zero",
        "medium_share_analysis_included": False,
        "figures": outputs,
    }
    (destination / "render-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "figures": len(outputs),
                "directory": str(destination),
                "holdout_available": data["holdout"] is not None,
                "arena_available": data["arena"] is not None,
            }
        )
    )


if __name__ == "__main__":
    main()
