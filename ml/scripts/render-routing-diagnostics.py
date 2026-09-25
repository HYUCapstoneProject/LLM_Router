"""Render reproducible, descriptive train/dev routing diagnostics from frozen scores.

No model inference, policy selection, held-out outcomes, or RouterArena predictions
are used. The saved validation policies remain unchanged. Run with a Python
environment containing numpy, pandas, pyarrow and matplotlib.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/llm-router-routing-diagnostics-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

METHODS = ("ordinal", "pointwise", "pairwise")
METHOD_NAMES = {
    "ordinal": "R0 · CE",
    "pointwise": "R1 · BCE",
    "pairwise": "R2 · RankNet",
}
TIERS = ("low", "medium", "high")
LABELS = ("easy", "medium", "hard")
COLORS = ("#2c84b8", "#e7a339", "#ca5363")
METHOD_COLORS = ("#64748b", "#087f8c", "#8b5fbf")
SOURCE_NAMES = {"routerbench": "RouterBench", "xroutebench": "xRouteBench"}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def verify(path, record, provenance):
    actual = {"sha256": digest(path), "size_bytes": Path(path).stat().st_size}
    expected = record if isinstance(record, dict) else {"sha256": record}
    for field in ("sha256", "size_bytes"):
        if field in expected and actual[field] != expected[field]:
            raise ValueError(f"Input checksum mismatch: {path} ({field})")
    provenance[str(Path(path).resolve())] = actual


def paired(reference, candidate, name):
    """Strict one-to-one ID join; every overlapping input/outcome must agree."""
    for frame in (reference, candidate):
        if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
            raise ValueError(f"Missing/duplicate sample IDs: {name}")
    if set(reference.sample_id) != set(candidate.sample_id):
        raise ValueError(f"Sample ID coverage differs: {name}")
    aligned = candidate.set_index("sample_id").loc[reference.sample_id].reset_index()
    for column in set(reference.columns) & set(aligned.columns):
        if not reference[column].reset_index(drop=True).equals(aligned[column]):
            raise ValueError(f"Joined metadata/outcome mismatch: {name}.{column}")
    return aligned


def grouped_counts(frame, group_columns, value_column, values):
    rows = []
    for key, group in frame.groupby(group_columns, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        counts = group[value_column].value_counts().reindex(values, fill_value=0)
        for value in values:
            rows.append(
                {
                    **dict(zip(group_columns, key, strict=True)),
                    value_column: value,
                    "count": int(counts[value]),
                    "denominator": len(group),
                    "fraction": float(counts[value] / len(group)),
                    "small_sample_n_lt_30": len(group) < 30,
                }
            )
    return pd.DataFrame(rows)


def performance(rows, actions):
    index = np.arange(len(rows))
    quality = rows[[f"quality_{tier}" for tier in TIERS]].to_numpy(float)
    cost = rows[[f"actual_cost_{tier}" for tier in TIERS]].to_numpy(float)
    low, high = cost[:, 0].mean(), cost[:, 2].mean()
    chosen_cost = float(cost[index, actions].mean())
    return {
        "n": len(rows),
        "mean_quality": float(quality[index, actions].mean()),
        "mean_actual_cost_source_units": chosen_cost,
        "relative_cost": float((chosen_cost - low) / (high - low))
        if high > low
        else None,
        **{
            tier + "_rate": float(np.mean(actions == i)) for i, tier in enumerate(TIERS)
        },
    }


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#334155",
            "text.color": "#172b45",
            "axes.edgecolor": "#bac7d5",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig, path, title, subtitle):
    fig.suptitle(title, fontsize=17, fontweight="bold", x=0.02, ha="left", y=0.99)
    fig.text(
        0.02,
        0.99 - 0.36 / fig.get_figheight(),
        subtitle,
        fontsize=10,
        color="#52637a",
        va="top",
    )
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def stacked(ax, counts, label_column, values, order, labels, title, annotations=True):
    matrix = (
        counts.pivot(index=label_column, columns="value", values="fraction")
        .reindex(index=order, columns=values)
        .fillna(0)
    )
    left = np.zeros(len(matrix))
    y = np.arange(len(matrix))
    for value, color in zip(values, COLORS, strict=True):
        widths = matrix[value].to_numpy(float)
        ax.barh(y, widths, left=left, color=color, height=0.7, label=value)
        if annotations:
            for i, width in enumerate(widths):
                if width >= 0.095:
                    ax.text(
                        left[i] + width / 2,
                        i,
                        f"{width:.0%}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=9,
                        fontweight="bold",
                    )
        left += widths
    ax.set(yticks=y, yticklabels=labels, xlim=(0, 1), title=title)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)


def render(args):
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {}
    config = read_json(args.config)
    lock = read_json(args.comparison / "validation-lock.json")
    inputs = read_json(args.comparison / "input-manifest.json")
    verify(args.config, lock["experiment_config_sha256"], provenance)
    if lock["stage"] != "final" or lock["selection_data"] != "validation_only":
        raise ValueError("A final validation-only lock is required")
    if (
        set(lock["methods"]) != set(METHODS)
        or lock["default_relative_cost_budget"] != 0.5
    ):
        raise ValueError("Expected all three methods and frozen default budget 0.5")
    for name in ("validation-lock.json", "input-manifest.json"):
        verify(args.comparison / name, {}, provenance)
    for name in ("evaluation.py", "data.py", "objectives.py"):
        key = f"ml/src/llm_router_labeling/objective_comparison/{name}"
        verify(root / key, inputs["files"][key], provenance)
    sys.path.insert(0, str(root / "ml/src"))
    from llm_router_labeling.objective_comparison.evaluation import route_actions

    base_manifest = read_json(args.dataset / "manifest.json")
    verify(
        args.dataset / "manifest.json",
        config["base_dataset_manifest_sha256"],
        provenance,
    )
    base = {}
    metadata = None
    for split, name in (
        ("train", "train/training.parquet"),
        ("validation", "validation/validation.parquet"),
        ("metadata", "metadata.parquet"),
    ):
        path = args.dataset / name
        verify(path, base_manifest["artifacts"][name], provenance)
        frame = pd.read_parquet(path)
        if split == "metadata":
            metadata = frame.loc[frame.split.isin(["train", "validation"])].copy()
        else:
            base[split] = frame.sort_values("sample_id").reset_index(drop=True)
            if set(frame.split) != {split}:
                raise ValueError("Base split role mismatch")
    for split, base_rows in base.items():
        paired(base_rows, metadata.loc[metadata.split.eq(split)], "metadata/" + split)
    combined = pd.concat(base.values(), ignore_index=True)
    if combined.groupby("duplicate_group_id").split.nunique().gt(1).any():
        raise ValueError("A duplicate group crosses train/dev")
    thresholds = base_manifest["selected_thresholds"]["t1"]
    relabeled = np.where(
        combined.overall_solve_rate >= thresholds["easy_threshold"],
        "easy",
        np.where(
            combined.overall_solve_rate >= thresholds["hard_threshold"],
            "medium",
            "hard",
        ),
    )
    if not np.array_equal(relabeled, combined.difficulty_label.to_numpy()):
        raise ValueError("T1 threshold reconstruction disagrees with labels")

    outcome_manifest_path = args.outcomes / "manifest.json"
    outcome_manifest = read_json(outcome_manifest_path)
    verify(
        outcome_manifest_path,
        inputs["files"]["data/outcomes/manifest.json"],
        provenance,
    )
    if outcome_manifest["base_dataset_manifest_sha256"] != digest(
        args.dataset / "manifest.json"
    ):
        raise ValueError("Outcome/base dataset provenance mismatch")
    outcomes = {}
    for split, base_rows in base.items():
        path = args.outcomes / f"{split}.parquet"
        verify(path, outcome_manifest["artifacts"][path.name], provenance)
        outcomes[split] = paired(base_rows, pd.read_parquet(path), "outcomes/" + split)
    validation = outcomes["validation"]
    sources = sorted(validation.source_dataset.unique())
    joined, routes, penalties, checks = {}, [], [], {}
    budget_key = "0.5000"
    for method in METHODS:
        run_dir = args.comparison / method
        manifest = read_json(run_dir / "manifest.json")
        verify(run_dir / "manifest.json", {}, provenance)
        if manifest["experiment_config_sha256"] != digest(args.config) or manifest[
            "outcome_manifest_sha256"
        ] != digest(outcome_manifest_path):
            raise ValueError("Run configuration or dataset provenance mismatch")
        if (
            manifest["validation_data_sha256"]
            != digest(args.outcomes / "validation.parquet")
            or manifest["method"] != method
            or manifest["seed"] != 42
        ):
            raise ValueError("Run validation/seed/method contract differs")
        for name, record in manifest["artifacts"].items():
            verify(run_dir / name, record, provenance)
        frame = paired(
            validation, pd.read_parquet(run_dir / "validation-scores.parquet"), method
        )
        joined[method] = frame
        scores = frame[[f"score_{tier}" for tier in TIERS]].to_numpy(float)
        costs = frame[[f"decision_cost_{tier}" for tier in TIERS]].to_numpy(float)
        actions = np.empty(len(frame), dtype=int)
        for source in sources:
            mask = frame.source_dataset.eq(source).to_numpy()
            penalty = float(
                lock["methods"][method]["locked_budget_penalties"][source][budget_key]
            )
            actions[mask] = route_actions(method, scores[mask], costs[mask], penalty)
            penalties.append(
                {
                    "method": method,
                    "source_dataset": source,
                    "budget": 0.5,
                    "lambda": penalty,
                }
            )
        row = validation[
            [
                "sample_id",
                "source_dataset",
                "category",
                "difficulty_label",
                "duplicate_group_id",
            ]
        ].copy()
        row["method"], row["route"] = method, np.asarray(TIERS)[actions]
        for prefix in ("quality", "actual_cost"):
            row["selected_" + prefix] = validation[
                [f"{prefix}_{tier}" for tier in TIERS]
            ].to_numpy(float)[np.arange(len(row)), actions]
        routes.append(row)
        # The producer prunes equal-cost lambda points. Verify every retained
        # point, then reconstruct the complete preregistered grid below.
        stored = pd.read_parquet(run_dir / "validation-curve.parquet")
        for record in stored.to_dict("records"):
            mask = frame.source_dataset.eq(record["source_dataset"]).to_numpy()
            selected = route_actions(
                method, scores[mask], costs[mask], record["penalty"]
            )
            computed = performance(validation.loc[mask], selected)
            for field in ("low_rate", "medium_rate", "high_rate", "mean_quality"):
                if not np.isclose(
                    computed[field], record[field], atol=1e-12, rtol=1e-12
                ):
                    raise ValueError("Frozen curve reconstruction mismatch: " + field)
        checks[method] = {
            "joined_rows": len(frame),
            "overlapping_columns_verified": len(set(frame) & set(validation)),
            "stored_curve_points_reproduced": len(stored),
        }
    route_rows = pd.concat(routes, ignore_index=True)
    combined["split"] = combined["split"].map({"train": "train", "validation": "dev"})
    tables = {
        "difficulty-by-source": grouped_counts(
            combined, ["split", "source_dataset"], "difficulty_label", LABELS
        ),
        "difficulty-by-category": grouped_counts(
            combined,
            ["split", "source_dataset", "category"],
            "difficulty_label",
            LABELS,
        ),
        "routing-by-source": grouped_counts(
            route_rows, ["method", "source_dataset"], "route", TIERS
        ),
        "routing-by-category": grouped_counts(
            route_rows, ["method", "source_dataset", "category"], "route", TIERS
        ),
        "difficulty-by-route": grouped_counts(
            route_rows, ["method", "source_dataset", "difficulty_label"], "route", TIERS
        ),
        "validation-routing-rows": route_rows,
        "locked-penalties": pd.DataFrame(penalties),
    }
    lambda_rows, perf_rows, within_rows, majority_rows = [], [], [], []
    baselines = {}
    for source in sources:
        train = outcomes["train"].loc[outcomes["train"].source_dataset.eq(source)]
        quality = train[[f"quality_{tier}" for tier in TIERS]].mean().to_numpy()
        actual_cost = train[[f"actual_cost_{tier}" for tier in TIERS]].mean().to_numpy()
        span = actual_cost[2] - actual_cost[0]
        normalized = (actual_cost - actual_cost[0]) / span
        ids = train[[f"model_{tier}" for tier in TIERS]].iloc[0].tolist()
        ranking = sorted(range(3), key=lambda i: (-quality[i], actual_cost[i], ids[i]))
        budget_ranking = [i for i in ranking if normalized[i] <= 0.5 + 1e-12]
        if span <= 0 or not budget_ranking:
            raise ValueError(
                "Train source-only baseline has no valid cost normalization"
            )
        baselines[source] = {
            "train_best_single": ranking[0],
            "train_budget_fixed": budget_ranking[0],
        }
        part = validation.loc[validation.source_dataset.eq(source)]
        for name, index in baselines[source].items():
            perf_rows.append(
                {
                    "method": name,
                    "source_dataset": source,
                    "selected_train_tier": TIERS[index],
                    **performance(part, np.full(len(part), index)),
                }
            )
        majority = (
            train.difficulty_label.value_counts().reindex(LABELS, fill_value=0).idxmax()
        )
        raw = (
            joined["ordinal"]
            .loc[
                joined["ordinal"].source_dataset.eq(source),
                [f"score_{tier}" for tier in TIERS],
            ]
            .to_numpy(float)
        )
        raw_labels = np.asarray(LABELS)[np.argmax(raw, axis=1)]
        majority_rows.append(
            {
                "source_dataset": source,
                "train_majority_label": majority,
                "dev_n": len(part),
                "source_only_label_accuracy": float(
                    part.difficulty_label.eq(majority).mean()
                ),
                "r0_raw_argmax_t1_accuracy": float(
                    np.mean(raw_labels == part.difficulty_label.to_numpy())
                ),
            }
        )
    for method in METHODS:
        frame = joined[method]
        scores = frame[[f"score_{tier}" for tier in TIERS]].to_numpy(float)
        costs = frame[[f"decision_cost_{tier}" for tier in TIERS]].to_numpy(float)
        for penalty in config["evaluation"]["penalty_grid"]:
            actions = route_actions(method, scores, costs, penalty)
            for source in (*sources, "pooled_rows"):
                mask = (
                    np.ones(len(frame), dtype=bool)
                    if source == "pooled_rows"
                    else frame.source_dataset.eq(source).to_numpy()
                )
                lambda_rows.append(
                    {
                        "method": method,
                        "source_dataset": source,
                        "lambda": float(penalty),
                        "n": int(mask.sum()),
                        **{
                            tier + "_rate": float(np.mean(actions[mask] == i))
                            for i, tier in enumerate(TIERS)
                        },
                    }
                )
        selected = route_rows.loc[route_rows.method.eq(method)]
        for source in sources:
            mask = validation.source_dataset.eq(source).to_numpy()
            part = validation.loc[mask]
            actions = (
                selected.loc[mask, "route"]
                .map(dict(zip(TIERS, range(3), strict=True)))
                .to_numpy(int)
            )
            perf_rows.append(
                {
                    "method": method,
                    "source_dataset": source,
                    "selected_train_tier": None,
                    **performance(part, actions),
                }
            )
        for (source, category), group in selected.groupby(
            ["source_dataset", "category"]
        ):
            part = validation.set_index("sample_id").loc[group.sample_id]
            index = baselines[source]["train_budget_fixed"]
            within_rows.append(
                {
                    "method": method,
                    "source_dataset": source,
                    "category": category,
                    "n": len(group),
                    "mean_quality": float(group.selected_quality.mean()),
                    "train_budget_fixed_quality": float(
                        part[f"quality_{TIERS[index]}"].mean()
                    ),
                    "quality_delta_vs_train_budget_fixed": float(
                        group.selected_quality.mean()
                        - part[f"quality_{TIERS[index]}"].mean()
                    ),
                    "medium_rate": float(group.route.eq("medium").mean()),
                    "small_sample_n_lt_30": len(group) < 30,
                }
            )
    tables.update(
        {
            "lambda-grid-routing": pd.DataFrame(lambda_rows),
            "source-only-comparison": pd.DataFrame(perf_rows),
            "within-source-category-comparison": pd.DataFrame(within_rows),
            "r0-label-diagnostic": pd.DataFrame(majority_rows),
        }
    )
    zero_cost = (
        tables["lambda-grid-routing"]
        .loc[
            tables["lambda-grid-routing"]["lambda"].eq(0)
            & tables["lambda-grid-routing"].source_dataset.ne("pooled_rows"),
            ["method", "source_dataset", "medium_rate"],
        ]
        .rename(columns={"medium_rate": "lambda_zero_medium_rate"})
    )
    locked_medium = (
        tables["routing-by-source"]
        .loc[
            tables["routing-by-source"].route.eq("medium"),
            ["method", "source_dataset", "fraction"],
        ]
        .rename(columns={"fraction": "locked_medium_rate"})
    )
    tables["medium-cost-policy-effect"] = zero_cost.merge(
        locked_medium, on=["method", "source_dataset"], validate="one_to_one"
    )
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)

    style()
    category_keys = list(
        validation.groupby(["source_dataset", "category"]).size().sort_index().index
    )
    category_names = [
        f"{SOURCE_NAMES[source]} / {category}" for source, category in category_keys
    ]
    source_counts = validation.groupby("source_dataset").size()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.7))
    fig.subplots_adjust(top=0.77, bottom=0.19, left=0.14, right=0.98, wspace=0.5)
    for ax, split in zip(axes, ("train", "dev"), strict=True):
        table = (
            tables["difficulty-by-source"]
            .query("split == @split")
            .rename(columns={"difficulty_label": "value"})
        )
        ns = table.groupby("source_dataset").denominator.first()
        stacked(
            ax,
            table,
            "source_dataset",
            LABELS,
            sources,
            [f"{SOURCE_NAMES[s]}\nn={ns[s]:,}" for s in sources],
            split.upper(),
        )
    axes[-1].legend(
        loc="upper center", bbox_to_anchor=(-0.25, -0.18), ncol=3, frameon=False
    )
    save(
        fig,
        output / "01-difficulty-source.png",
        "T1 benchmark-derived difficulty · source composition",
        "Train and dev are separate partitions; these are not routing targets for R1/R2.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    fig.subplots_adjust(top=0.89, bottom=0.09, left=0.31, right=0.99, wspace=0.3)
    for i, (ax, split) in enumerate(zip(axes, ("train", "dev"), strict=True)):
        table = tables["difficulty-by-category"].query("split == @split").copy()
        table["key"] = list(zip(table.source_dataset, table.category, strict=True))
        table = table.rename(columns={"difficulty_label": "value"})
        ns = table.groupby("key").denominator.first()
        order = [str(key) for key in category_keys]
        table["key"] = table.key.astype(str)
        labels = [
            (category_names[j] + f" · n={ns[key]:,}" + (" *" if ns[key] < 30 else ""))
            if i == 0
            else f"n={ns[key]:,}" + (" *" if ns[key] < 30 else "")
            for j, key in enumerate(category_keys)
        ]
        stacked(ax, table, "key", LABELS, order, labels, split.upper())
    axes[-1].legend(
        loc="lower center", bbox_to_anchor=(-0.2, -0.09), ncol=3, frameon=False
    )
    save(
        fig,
        output / "02-difficulty-category.png",
        "T1 difficulty by source / category",
        "easy: solve rate >= 0.64 · medium: 0.27–0.64 · hard: < 0.27. * n < 30; category taxonomies differ by source.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    fig.subplots_adjust(top=0.79, bottom=0.2, left=0.12, right=0.98, wspace=0.37)
    for ax, source in zip(axes, sources, strict=True):
        table = (
            tables["routing-by-source"]
            .query("source_dataset == @source")
            .rename(columns={"route": "value"})
        )
        locked = {
            record["method"]: record["lambda"]
            for record in penalties
            if record["source_dataset"] == source
        }
        stacked(
            ax,
            table,
            "method",
            TIERS,
            METHODS,
            [f"{METHOD_NAMES[m]}\nλ={locked[m]:g}" for m in METHODS],
            f"{SOURCE_NAMES[source]} · dev n={source_counts[source]:,}",
        )
    axes[-1].legend(
        loc="upper center", bbox_to_anchor=(-0.3, -0.19), ncol=3, frameon=False
    )
    save(
        fig,
        output / "03-routing-source.png",
        "Routing actions at the frozen relative-cost budget 0.5",
        "low / medium / high identify source-local candidate models, not easy / medium / hard difficulty labels.",
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 10))
    fig.subplots_adjust(top=0.89, bottom=0.08, left=0.28, right=0.99, wspace=0.14)
    dev_ns = validation.groupby(["source_dataset", "category"]).size()
    for i, (ax, method) in enumerate(zip(axes, METHODS, strict=True)):
        table = tables["routing-by-category"].query("method == @method").copy()
        table["key"] = [
            str((s, c))
            for s, c in zip(table.source_dataset, table.category, strict=True)
        ]
        table = table.rename(columns={"route": "value"})
        labels = (
            [
                f"{name} · n={dev_ns[key]}" + (" *" if dev_ns[key] < 30 else "")
                for name, key in zip(category_names, category_keys, strict=True)
            ]
            if i == 0
            else [""] * len(category_keys)
        )
        stacked(
            ax,
            table,
            "key",
            TIERS,
            [str(key) for key in category_keys],
            labels,
            METHOD_NAMES[method],
        )
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False
    )
    save(
        fig,
        output / "04-routing-category.png",
        "Within-source category routing · frozen budget 0.5",
        "Every panel uses the same dev sample IDs. * n < 30; percentages are descriptive, not significance tests.",
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 8.2))
    fig.subplots_adjust(
        top=0.85, bottom=0.08, left=0.12, right=0.96, hspace=0.6, wspace=0.5
    )
    for i, source in enumerate(sources):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]
            table = tables["difficulty-by-route"].query(
                "source_dataset == @source and method == @method"
            )
            counts = (
                table.pivot(index="difficulty_label", columns="route", values="count")
                .reindex(index=LABELS, columns=TIERS)
                .fillna(0)
                .to_numpy(int)
            )
            fractions = counts / counts.sum(axis=1, keepdims=True)
            ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set(
                xticks=range(3),
                xticklabels=TIERS,
                yticks=range(3),
                yticklabels=[
                    f"{label}\nn={n}"
                    for label, n in zip(LABELS, counts.sum(axis=1), strict=True)
                ],
                title=f"{SOURCE_NAMES[source]} · {METHOD_NAMES[method]}",
                xlabel="Selected action",
                ylabel="T1 difficulty label" if j == 0 else "",
            )
            for y in range(3):
                for x in range(3):
                    ax.text(
                        x,
                        y,
                        f"{fractions[y, x]:.1%}\n({counts[y, x]})",
                        ha="center",
                        va="center",
                        color="white" if fractions[y, x] > 0.55 else "#172b45",
                        fontsize=9,
                    )
    save(
        fig,
        output / "05-difficulty-route-heatmaps.png",
        "T1 difficulty × chosen action · row-normalized",
        "These are association tables, not R1/R2 classification confusion matrices. Each label row sums to 100%.",
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))
    fig.subplots_adjust(top=0.79, bottom=0.2, left=0.07, right=0.98, wspace=0.23)
    for ax, source in zip(axes, (*sources, "pooled_rows"), strict=True):
        for method, color in zip(METHODS, METHOD_COLORS, strict=True):
            table = (
                tables["lambda-grid-routing"]
                .query("source_dataset == @source and method == @method")
                .sort_values("lambda")
            )
            ax.plot(
                table["lambda"],
                table.medium_rate,
                marker="o",
                markersize=4,
                lw=2,
                color=color,
                label=METHOD_NAMES[method],
            )
            if source != "pooled_rows":
                penalty = lock["methods"][method]["locked_budget_penalties"][source][
                    budget_key
                ]
                value = table.loc[table["lambda"].eq(penalty), "medium_rate"].iloc[0]
                ax.scatter(
                    [penalty],
                    [value],
                    marker="*",
                    s=170,
                    color=color,
                    edgecolor="white",
                    zorder=5,
                )
        ax.set_xscale("symlog", linthresh=0.0001)
        ax.set(
            ylim=(-0.02, 1.02),
            xlabel="Cost penalty λ (symlog; includes zero)",
            title=SOURCE_NAMES.get(source, "All dev rows · weighted by n"),
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Fraction routed to medium")
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False
    )
    save(
        fig,
        output / "06-medium-vs-lambda.png",
        "Medium routing across every preregistered λ",
        "Stars mark source-specific frozen policies at budget 0.5. λ has different score-scale meaning across objectives.",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    fig.subplots_adjust(top=0.79, bottom=0.22, left=0.08, right=0.98, wspace=0.3)
    styles = {
        **{
            method: (name, color, "o")
            for method, name, color in zip(
                METHODS, METHOD_NAMES.values(), METHOD_COLORS, strict=True
            )
        },
        "train_best_single": ("Train-best single", "#334155", "s"),
        "train_budget_fixed": ("Source-only fixed / train budget 0.5", "#ce7735", "D"),
    }
    for ax, source in zip(axes, sources, strict=True):
        table = tables["source-only-comparison"].query("source_dataset == @source")
        for record in table.to_dict("records"):
            name, color, marker = styles[record["method"]]
            ax.scatter(
                record["relative_cost"],
                record["mean_quality"],
                color=color,
                marker=marker,
                s=80,
                label=name,
            )
        ax.axvline(0.5, color="#a3adb9", ls="--", lw=1)
        ax.set(
            title=SOURCE_NAMES[source],
            xlabel="Source-normalized realized relative cost",
            ylabel="Mean recorded benchmark quality",
            xlim=(-0.03, 1.04),
        )
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    save(
        fig,
        output / "07-source-only-comparison.png",
        "Question-dependent routing versus source-only fixed choices",
        "Fixed baselines use train outcomes only. Dev points can have different realized costs; this is not a paired significance claim.",
    )

    report = {
        "schema_version": "routing-distribution-diagnostics-v1",
        "scope": "descriptive_train_and_dev_only",
        "new_model_inference": False,
        "policy_reselected": False,
        "routerarena_predictions_available_in_this_report": False,
        "seed_scope": "single_seed_42",
        "frozen_budget": 0.5,
        "t1_thresholds": thresholds,
        "row_counts": {key: len(value) for key, value in base.items()},
        "source_dev_counts": {key: int(value) for key, value in source_counts.items()},
        "difficulty_dev_counts": {
            key: int(value)
            for key, value in base["validation"].difficulty_label.value_counts().items()
        },
        "join_and_curve_checks": checks,
        "input_files": provenance,
        "source_local_candidates": {
            source: record["selected_models"]
            for source, record in outcome_manifest["arms"]["sources"].items()
        },
        "source_only_baseline_selection": {
            source: {key: TIERS[value] for key, value in values.items()}
            for source, values in baselines.items()
        },
        "tables": {
            name: json.loads(table.to_json(orient="records"))
            for name, table in tables.items()
            if name != "validation-routing-rows"
        },
        "limitations": [
            "T1 labels are benchmark-derived difficulty proxies, not human gold.",
            "Routing actions are source-local model choices, not difficulty classes.",
            "Category taxonomies differ by source; categories are not silently merged.",
            "Small cells and single-seed descriptive differences do not establish superiority.",
            "Raw costs are only compared within a source; no shared currency is assumed.",
            "No latest RouterArena inference is present; RouterArena is not substituted with other benchmarks.",
        ],
    }
    (output / "diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    write_html(output, report, tables, config)
    file_manifest = {
        path.name: {"sha256": digest(path), "size_bytes": path.stat().st_size}
        for path in output.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "routing-diagnostics-artifacts-v1",
                "script_sha256": digest(__file__),
                "files": file_manifest,
            },
            indent=2,
        )
        + "\n"
    )
    return report


def write_html(output, report, tables, config):
    source_rows = tables["routing-by-source"].query("route == 'medium'")
    pivot = source_rows.pivot(
        index="source_dataset", columns="method", values="fraction"
    ).reindex(columns=METHODS)
    percentages = pivot.map(lambda value: f"{value:.1%}")
    percentages.index = percentages.index.map(SOURCE_NAMES)
    percentages.columns = [METHOD_NAMES[method] for method in METHODS]
    blocks = [
        (
            "01-difficulty-source.png",
            "학습·검증 난이도 분포",
            "T1은 전체 후보 모델의 정답률로 만든 난이도 지표입니다. 정답률 0.64 이상은 easy, 0.27 이상 0.64 미만은 medium, 그 미만은 hard입니다. 사람의 난이도 판정과 동일한 정답으로 취급하지 않습니다.",
        ),
        (
            "02-difficulty-category.png",
            "카테고리별 난이도",
            "원천마다 카테고리 체계가 달라 원천/카테고리 조합을 유지했습니다. 오른쪽은 왼쪽과 같은 카테고리 순서이며, 별표(*)는 30건 미만입니다.",
        ),
        (
            "03-routing-source.png",
            "고정 budget 0.5의 라우팅 분포",
            "각 방법·원천에 저장된 λ를 그대로 적용했습니다. low/medium/high는 해당 원천의 후보 모델 순서이며 easy/medium/hard 난이도와 다른 개념입니다. 공개 벤치마크의 후보는 Claude 모델이 아닙니다.",
        ),
        (
            "04-routing-category.png",
            "같은 원천 안에서 카테고리별 라우팅",
            "같은 검증 샘플을 비교합니다. medium 비중 하나만으로 붕괴 여부를 판단할 수 없습니다. 원천별 모델 비용, 질문별 품질 예측과 λ가 함께 결정합니다.",
        ),
        (
            "05-difficulty-route-heatmaps.png",
            "난이도와 실제 선택의 관계",
            "각 행은 같은 T1 난이도에 속하는 질문입니다. R1·R2의 학습 목표는 난이도 맞히기가 아니므로 이 표를 분류 정확도나 혼동행렬로 해석하면 안 됩니다.",
        ),
        (
            "06-medium-vs-lambda.png",
            "전체 λ 그리드에서 medium 비율",
            "기존 곡선 저장 과정에서 중복 비용 점이 생략됐으므로, 원래 등록된 14개 λ 모두를 저장 점수로 복원했습니다. 별은 budget 0.5에서 고정한 원천별 정책입니다. λ 숫자의 의미는 CE·BCE·RankNet 점수 척도에 따라 다릅니다. 오른쪽은 원천 비율로 가중한 전체 검증 데이터입니다.",
        ),
        (
            "07-source-only-comparison.png",
            "원천만 아는 고정 선택과 비교",
            "Source-only fixed는 학습 데이터에서 원천별 평균 비용이 budget 0.5 이하인 후보 중 평균 품질이 가장 높은 하나를 고릅니다. Train-best single은 비용 제한 없이 학습 평균 품질로 선택합니다. 두 기준 모두 검증 결과로 후보를 다시 고르지 않았습니다. 그래프의 실현 비용은 같지 않으므로 동일 비용 우월성이나 통계적 유의성으로 해석하지 않습니다.",
        ),
    ]
    cards = "".join(
        f'<section><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p><a href="{filename}"><img src="{filename}" alt="{html.escape(title)}" loading="lazy"></a><p class="caption">PNG 원본을 열려면 그림을 누르세요.</p></section>'
        for filename, title, text in blocks
    )
    downloads = "".join(f'<a href="{name}.csv">{name}.csv</a>' for name in tables)
    locks = tables["locked-penalties"].rename(columns={"source_dataset": "source"})
    candidate_table = pd.DataFrame.from_dict(
        report["source_local_candidates"], orient="index"
    ).reindex(columns=TIERS)
    majority = tables["r0-label-diagnostic"].copy()
    for name in ("source_only_label_accuracy", "r0_raw_argmax_t1_accuracy"):
        majority[name] = majority[name].map(lambda value: f"{value:.1%}")
    cost_effect = tables["medium-cost-policy-effect"].copy()
    for name in ("lambda_zero_medium_rate", "locked_medium_rate"):
        cost_effect[name] = cost_effect[name].map(lambda value: f"{value:.1%}")
    (output / "index.html").write_text(
        f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>R0 · R1 · R2 라우팅 분포 진단</title><style>
body{{margin:0;background:#f2f5f8;color:#172b45;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1180px;margin:0 auto;padding:44px 24px 80px}}header{{margin-bottom:26px}}h1{{font-size:36px;line-height:1.25;letter-spacing:-1.3px}}h2{{font-size:23px;line-height:1.4;margin:0 0 12px}}p{{max-width:95ch;color:#45566f}}.eyebrow{{color:#087f8c;font-weight:700;letter-spacing:1.4px;font-size:12px}}.note{{border-left:4px solid #087f8c;background:#e6f4f3;padding:18px 24px}}section{{background:white;border:1px solid #dce4ed;border-radius:12px;padding:26px;margin-top:22px}}img{{width:100%;height:auto;margin-top:12px}}.caption{{font-size:12px;color:#6b7a8f}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:right;padding:9px;border-bottom:1px solid #e2e8ef}}th:first-child,td:first-child{{text-align:left}}.downloads{{display:flex;gap:8px;flex-wrap:wrap}}a{{color:#087f8c}}.downloads a{{background:#edf5f8;padding:7px 10px;border-radius:6px;font-size:12px}}code{{word-break:break-all}}details{{margin-top:16px}}@media(max-width:650px){{main{{padding:24px 12px}}h1{{font-size:28px}}section{{padding:16px}}table{{font-size:11px}}}}
</style><main><header><div class="eyebrow">VALIDATION DIAGNOSTICS · SEED 42 · FROZEN POLICY</div><h1>난이도와 라우팅 선택을<br>원천·카테고리별로 확인하기</h1><p>학습 {report["row_counts"]["train"]:,}건 / 개발검증 {report["row_counts"]["validation"]:,}건. 기존 R0·R1·R2 검증 점수를 재집계했으며, 모델 추론이나 정책 재선택은 수행하지 않았습니다.</p></header>
<div class="note"><b>난이도 라벨과 라우팅 단계는 서로 다릅니다.</b><br>T1 easy/medium/hard는 정답률 기반 난이도입니다. 선택 action low/medium/high는 원천별 3개 후보 중 하나입니다. R1의 medium 선택이 많다는 사실만으로 “질문을 전부 중간 난이도로 판정했다”고 해석할 수 없습니다.</div>
<section><h2>budget 0.5에서 medium을 선택한 비율</h2>{percentages.to_html(border=0)}<p>검증 데이터는 RouterBench 3,122건(87.2%), xRouteBench 459건(12.8%)입니다. 전체 평균만 보면 큰 원천의 양상이 과대표현될 수 있습니다.</p><details><summary>실제 후보 모델과 점수 의미</summary>{candidate_table.to_html(border=0)}<p>R0는 난이도 분류 확률에 비용 정책을 적용합니다. R1은 후보별 독립 성공 점수, R2는 후보 사이의 상대 품질 점수에 비용 정책을 적용합니다. R2 점수는 성공 확률이 아닙니다.</p></details></section>
<section><h2>medium 집중에 비용 정책이 미치는 영향</h2><p>같은 모델 점수에서 λ만 0으로 놓은 진단과, 저장된 budget 0.5 정책을 비교합니다. 예를 들어 RouterBench에서 R2의 medium 비율은 λ=0일 때 23.9%, 고정 정책 λ=1에서는 88.0%입니다. 이 차이는 비용 정책 적용으로 발생합니다. λ=0 결과는 진단용이며 정책을 다시 선택한 것이 아닙니다.</p>{cost_effect.to_html(index=False, border=0)}</section>
{cards}<section><h2>R0 난이도 진단: 원천별 학습 다수 라벨과 비교</h2><p>이 표는 λ를 적용하기 전 R0 raw argmax를 T1과 비교합니다. source-only 기준은 학습 데이터의 원천별 다수 라벨 하나입니다. R1·R2에는 같은 정확도 지표를 적용하지 않습니다.</p>{majority.to_html(index=False, border=0)}</section>
<section><h2>RouterArena 범위</h2><p>이 보고서에는 이번 R0·R1·R2의 RouterArena 추론 결과가 없습니다. 최신 결과가 있는 것처럼 다른 데이터셋을 대신 표시하지 않았습니다. 원천별 카테고리·라벨 확보 상태와 추론 가능 여부는 별도로 확인해야 합니다.</p></section>
<section><h2>검증과 재현</h2><p>원본 manifest SHA, artifact SHA·크기, 모든 sample ID의 일대일 결합, 중첩 메타데이터·품질·비용 열, train/dev 중복 그룹 분리, T1 임계값 재구성, 저장된 품질·비용 곡선의 라우팅 비율을 확인했습니다. 작은 셀과 단일 seed의 차이는 기술 통계이며 유의성 검정 결과가 아닙니다.</p><details><summary>고정 λ 보기</summary>{locks.to_html(index=False, border=0)}</details><p>수치 원본: <a href="diagnostics.json">diagnostics.json</a> · <a href="manifest.json">출력 SHA manifest</a> · <a href="../../../../ml/scripts/render-routing-diagnostics.py">재현 스크립트</a></p><p><code>PYTHONPATH=/private/tmp/llm-router-report-deps .venv/bin/python ml/scripts/render-routing-diagnostics.py</code></p><div class="downloads">{downloads}</div></section></main></html>""",
        encoding="utf-8",
    )


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("/private/tmp/llm-router-objective-results-local/comparison"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "ml/artifacts/modernbert-initial-v1/dataset",
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=root / "ml/artifacts/routing-objective-continual-v1/dataset",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "ml/config/routing_objective_continual_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "experiments/results/routing-objectives-continual-v1-A-20260924/routing-diagnostics",
    )
    args = parser.parse_args()
    report = render(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "verified_methods": report["join_and_curve_checks"],
                "source_counts": report["source_dev_counts"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
