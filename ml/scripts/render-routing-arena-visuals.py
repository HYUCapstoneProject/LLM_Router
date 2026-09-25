"""Render sealed RouterArena diagnostics without inference or policy selection.

Input is the `report` directory produced by evaluate-routing-arena-diagnostics.py.
Every category is shown (paginated); robustness is paired with full, never added
as independent observations. R1/R2 figures describe action-slot projection only.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/llm-router-arena-visuals-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

SCHEMA = "routing-arena-diagnostics-v1"
METHODS = ("ordinal", "pointwise", "pairwise")
NAMES = {
    "ordinal": "R0 · difficulty",
    "pointwise": "R1 · slot projection",
    "pairwise": "R2 · slot projection",
}
LABELS = ("easy", "medium", "hard")
ACTIONS = ("low", "medium", "high")
COLORS = ("#2688b4", "#e5a33a", "#c85569")
METHOD_COLORS = ("#63738a", "#087f8c", "#8b5fbf")
METADATA = (
    "source_sample_id",
    "evaluation_split",
    "difficulty_label",
    "category",
    "domain",
    "arena_source_dataset",
    "slice",
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_seal(directory, role, expected_sha=None):
    directory = directory.resolve()
    manifest_path = directory / "manifest.json"
    digest = sha(manifest_path)
    if (directory / "_SUCCESS").read_text().strip() != digest or (
        expected_sha is not None and digest != expected_sha
    ):
        raise ValueError("Artifact manifest seal or expected SHA mismatch")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA or manifest.get("role") != role:
        raise ValueError("Unsupported RouterArena artifact role/schema")
    for name, record in manifest["files"].items():
        path = directory / name
        if path.resolve().parent != directory or not path.is_file():
            raise ValueError("Unsafe or missing artifact filename")
        if sha(path) != record["sha256"] or path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    return manifest


def confusion(rows):
    matrix = np.zeros((3, 3), dtype=int)
    truth = rows.difficulty_label.map(
        dict(zip(LABELS, range(3), strict=True))
    ).to_numpy(int)
    np.add.at(matrix, (truth, rows.action_index.to_numpy(int)), 1)
    return matrix


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def group_metrics(rows):
    matrix = confusion(rows)
    n = len(rows)
    medium = int(matrix[1].sum())
    return {
        "n": n,
        "official_medium_n": medium,
        "selected_medium_n": int(matrix[:, 1].sum()),
        "medium_to_medium_n": int(matrix[1, 1]),
        "official_medium_fraction": safe_ratio(medium, n),
        "medium_selection_rate": safe_ratio(matrix[:, 1].sum(), n),
        "medium_label_capture_rate": safe_ratio(matrix[1, 1], medium),
        "label_slot_alignment_rate": safe_ratio(matrix.trace(), n),
        "truncated_n": int(rows.was_truncated.sum()),
    }


def validate_inputs(args):
    manifest = verify_seal(args.report_dir, "report", args.expected_report_sha256)
    summary = read_json(args.report_dir / "summary.json")
    if (
        summary.get("schema_version") != SCHEMA
        or summary.get("projection_policy") != "fixed_lambda_zero_action_slot_argmax"
        or summary.get("lambda_cost") != 0
    ):
        raise ValueError("Expected the frozen lambda-zero diagnostic protocol")
    if (
        summary.get("selection_influence") is not False
        or summary.get("cost_quality_claim_allowed") is not False
        or summary.get("production_activated") is not False
    ):
        raise ValueError(
            "Diagnostics must not claim policy selection, cost/quality or deployment"
        )
    rows = pd.read_parquet(args.report_dir / "predictions.parquet")
    required = {
        "sample_id",
        *METADATA,
        "method",
        "action_index",
        "projected_label",
        "confidence",
        "was_truncated",
        "input_token_count",
        *(f"score_{action}" for action in ACTIONS),
        *(f"action_score_{action}" for action in ACTIONS),
    }
    if required - set(rows) or rows.empty or set(rows.method) != set(METHODS):
        raise ValueError("Incomplete three-method prediction artifact")
    if (
        rows[
            [
                "sample_id",
                *METADATA,
                "method",
                "action_index",
                "was_truncated",
                "input_token_count",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Missing prediction identifiers or metadata")
    if set(rows.evaluation_split) - {"full", "robustness"} or set(
        rows.difficulty_label
    ) - set(LABELS):
        raise ValueError("Unknown split or official label")
    if (
        rows.duplicated(["method", "sample_id"]).any()
        or not rows.action_index.isin([0, 1, 2]).all()
    ):
        raise ValueError("Duplicate IDs or invalid action indices")
    if (
        not rows.was_truncated.isin([True, False]).all()
        or not rows.was_truncated.eq(rows.input_token_count.gt(512)).all()
    ):
        raise ValueError("Truncation flags disagree with the 512-token limit")
    if (
        not np.isfinite(rows.input_token_count.to_numpy(float)).all()
        or rows.input_token_count.le(0).any()
        or rows.input_token_count.mod(1).ne(0).any()
    ):
        raise ValueError("Token counts must be finite positive integers")
    reference = rows.loc[rows.method.eq("ordinal")].set_index("sample_id").sort_index()
    checks = {}
    for method in METHODS:
        part = rows.loc[rows.method.eq(method)].set_index("sample_id").sort_index()
        if not reference.index.equals(part.index):
            raise ValueError("Methods have different sample ID coverage")
        for column in METADATA:
            if not reference[column].equals(part[column]):
                raise ValueError(f"Cross-method metadata mismatch: {column}")
        raw = part[[f"score_{action}" for action in ACTIONS]].to_numpy(float)
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite logits")
        if method == "ordinal":
            scores = np.exp(raw - raw.max(axis=1, keepdims=True))
            scores /= scores.sum(axis=1, keepdims=True)
        elif method == "pointwise":
            scores = 1 / (1 + np.exp(-np.clip(raw, -30, 30)))
        else:
            scores = raw
        actions = scores.argmax(axis=1)
        if not np.array_equal(
            actions, part.action_index.to_numpy()
        ) or not np.array_equal(
            np.asarray(LABELS)[actions], part.projected_label.to_numpy()
        ):
            raise ValueError("Stored action projection differs from raw scores")
        if not np.allclose(
            scores,
            part[[f"action_score_{action}" for action in ACTIONS]].to_numpy(float),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("Transformed action scores mismatch")
        if method == "pairwise":
            if not part.confidence.isna().all():
                raise ValueError("R2 confidence must remain unavailable")
        elif not np.allclose(
            part.confidence.to_numpy(float),
            scores[np.arange(len(part)), actions],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("Selected-score confidence mismatch")
        full = part.loc[part.evaluation_split.eq("full")]
        if full.empty or full.source_sample_id.duplicated().any():
            raise ValueError("Full reference IDs must be nonempty and unique")
        stored = summary["methods"][method]["slices"]["full_clean"]
        matrix = confusion(full)
        if stored["rows"] != len(full) or not np.array_equal(
            stored["confusion_rows_true_columns_projected"], matrix
        ):
            raise ValueError("Reported confusion counts do not reproduce")
        if stored["label_support"] != dict(
            zip(LABELS, map(int, matrix.sum(axis=1)), strict=True)
        ) or stored["predicted_support"] != dict(
            zip(LABELS, map(int, matrix.sum(axis=0)), strict=True)
        ):
            raise ValueError("Reported true/projected support mismatch")
        if stored["truncated_rows"] != int(full.was_truncated.sum()):
            raise ValueError("Reported truncation count mismatch")
        for dimension in ("category", "domain", "arena_source_dataset"):
            groups = list(full.groupby(dimension, sort=True))
            if set(summary["methods"][method]["breakdown"][dimension]) != {
                str(name) for name, _ in groups
            }:
                raise ValueError("Missing category/domain/source in reported breakdown")
            for name, group in groups:
                detail = summary["methods"][method]["breakdown"][dimension][str(name)]
                if detail["rows"] != len(group) or not np.array_equal(
                    detail["confusion_rows_true_columns_projected"], confusion(group)
                ):
                    raise ValueError("Category/domain/source confusion mismatch")
        checks[method] = {
            "rows": len(part),
            "full_rows": len(full),
            "score_projection_verified": True,
            "official_label_and_breakdown_verified": True,
        }
    if args.prepared_dir is not None:
        prepared = verify_seal(
            args.prepared_dir, "prepared", manifest["prepared_manifest_sha256"]
        )
        if prepared["protocol_sha256"] != manifest["protocol_sha256"]:
            raise ValueError("Preparation protocol differs from report")
        original = (
            pd.read_parquet(args.prepared_dir / "evaluation.parquet")
            .set_index("sample_id")
            .sort_index()
        )
        if not reference.index.equals(original.index):
            raise ValueError("Report IDs differ from sealed preparation")
        for column in METADATA:
            if not reference[column].equals(original[column]):
                raise ValueError(
                    f"Report labels/metadata differ from preparation: {column}"
                )
    return rows, summary, manifest, checks


def paired_robustness(rows, summary):
    frames = []
    for method in METHODS:
        part = rows.loc[rows.method.eq(method)]
        full = part.loc[part.evaluation_split.eq("full")].set_index("source_sample_id")
        variants = part.loc[part.evaluation_split.eq("robustness")].copy()
        if variants.source_sample_id.duplicated().any() or not set(
            variants.source_sample_id
        ) <= set(full.index):
            raise ValueError("Robustness does not have unique retained full references")
        original = full.loc[variants.source_sample_id].reset_index()
        variants = variants.reset_index(drop=True)
        for column in (
            "difficulty_label",
            "category",
            "domain",
            "arena_source_dataset",
            "slice",
        ):
            if not variants[column].equals(original[column]):
                raise ValueError(
                    "Robustness official labels/strata differ from full references"
                )
        variants["full_action_index"] = original.action_index.to_numpy()
        variants["full_was_truncated"] = original.was_truncated.to_numpy()
        variants["action_change"] = variants.action_index - variants.full_action_index
        variants["transition"] = np.where(
            variants.action_change.eq(0),
            "same",
            np.where(variants.action_change.lt(0), "down", "up"),
        )
        stored = summary["methods"][method]["robustness"]
        if stored["paired_rows"] != len(variants):
            raise ValueError("Paired robustness denominator mismatch")
        for value, field in (
            ("same", "same_action_rate"),
            ("down", "downward_action_rate"),
            ("up", "upward_action_rate"),
        ):
            observed = safe_ratio(variants.transition.eq(value).sum(), len(variants))
            if (observed is None) != (stored[field] is None) or (
                observed is not None
                and not np.isclose(observed, stored[field], atol=1e-12, rtol=1e-12)
            ):
                raise ValueError("Paired robustness summary mismatch")
        frames.append(variants)
    return pd.concat(frames, ignore_index=True)


def save(fig, path, title, synthetic=False):
    fig.suptitle(
        ("SYNTHETIC QA · " if synthetic else "") + title,
        x=0.02,
        y=0.99,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="#172b45",
    )
    fig.text(
        0.02,
        0.99 - 0.36 / fig.get_figheight(),
        "Frozen λ=0 · R1/R2: action-slot alignment, not trained difficulty accuracy or Arena Score.",
        fontsize=9.5,
        color="#52637a",
        va="top",
    )
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_stacks(full, dimension, output, synthetic):
    reference = full.loc[full.method.eq("ordinal")]
    counts = reference.groupby(dimension).size().sort_index()
    names = list(counts.index)
    pages, records = [], []
    for start in range(0, len(names), 22):
        chosen = names[start : start + 22]
        height = max(4.4, 2.5 + 0.36 * len(chosen))
        fig, axes = plt.subplots(1, 4, figsize=(17, height))
        fig.subplots_adjust(
            top=1 - 1.05 / height,
            bottom=0.13,
            left=0.29,
            right=0.99,
            wspace=0.14,
        )
        for index, method in enumerate(("official", *METHODS)):
            source = (
                reference if method == "official" else full.loc[full.method.eq(method)]
            )
            column = "difficulty_label" if method == "official" else "projected_label"
            matrix = pd.crosstab(source[dimension], source[column]).reindex(
                index=chosen, columns=LABELS, fill_value=0
            )
            rates = matrix.div(matrix.sum(axis=1), axis=0)
            left = np.zeros(len(chosen))
            ax = axes[index]
            for label, color in zip(LABELS, COLORS, strict=True):
                width = rates[label].to_numpy(float)
                ax.barh(
                    range(len(chosen)),
                    width,
                    left=left,
                    color=color,
                    label=label,
                    height=0.75,
                )
                for j, value in enumerate(width):
                    if value >= 0.12:
                        ax.text(
                            left[j] + value / 2,
                            j,
                            f"{value:.0%}",
                            va="center",
                            ha="center",
                            fontsize=8,
                            color="white",
                            fontweight="bold",
                        )
                    records.append(
                        {
                            "dimension": dimension,
                            "value": chosen[j],
                            "method": method,
                            "label_or_projected_slot": label,
                            "count": int(matrix.loc[chosen[j], label]),
                            "denominator": int(counts[chosen[j]]),
                            "fraction": float(value),
                        }
                    )
                left += width
            ylabels = (
                [
                    textwrap.fill(str(name) or "(missing)", 39)
                    + f" · n={counts[name]}"
                    + (" *" if counts[name] < 30 else "")
                    for name in chosen
                ]
                if index == 0
                else [""] * len(chosen)
            )
            ax.set(
                yticks=range(len(chosen)),
                yticklabels=ylabels,
                xlim=(0, 1),
                title="Official labels" if method == "official" else NAMES[method],
            )
            ax.invert_yaxis()
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis="x", alpha=0.15)
            ax.set_axisbelow(True)
        axes[1].legend(
            loc="upper center", bbox_to_anchor=(1.1, -0.075), ncol=3, frameon=False
        )
        name = f"full-{dimension}-{start // 22 + 1:02d}.png"
        save(
            fig,
            output / name,
            f"RouterArena full · {dimension} ({start + 1}–{start + len(chosen)} of {len(names)})",
            synthetic,
        )
        pages.append(name)
    return pages, records


def plot_conditionals(full, output, synthetic):
    names = []
    for conditioned in ("official", "projected"):
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6))
        fig.subplots_adjust(top=0.75, bottom=0.17, left=0.11, right=0.99, wspace=0.42)
        for ax, method in zip(axes, METHODS, strict=True):
            matrix = confusion(full.loc[full.method.eq(method)])
            if conditioned == "projected":
                matrix = matrix.T
            support = matrix.sum(axis=1)
            rate = np.divide(
                matrix,
                support[:, None],
                out=np.full((3, 3), np.nan),
                where=support[:, None] > 0,
            )
            ax.imshow(
                np.ma.masked_invalid(rate), cmap="Blues", vmin=0, vmax=1, aspect="auto"
            )
            ax.set(
                title=NAMES[method],
                xticks=range(3),
                xticklabels=LABELS,
                yticks=range(3),
                yticklabels=[
                    f"{label}\nn={n}" for label, n in zip(LABELS, support, strict=True)
                ],
                xlabel="Projected slot"
                if conditioned == "official"
                else "Official difficulty",
                ylabel="Official difficulty"
                if conditioned == "official"
                else "Projected slot",
            )
            for y in range(3):
                for x in range(3):
                    value = rate[y, x]
                    text = (
                        "N/A\n(n=0)"
                        if np.isnan(value)
                        else f"{value:.1%}\n({matrix[y, x]})"
                    )
                    ax.text(
                        x,
                        y,
                        text,
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white"
                        if np.isfinite(value) and value > 0.55
                        else "#172b45",
                    )
        name = f"conditional-on-{conditioned}.png"
        save(
            fig,
            output / name,
            f"P({'projected slot | official label' if conditioned == 'official' else 'official label | projected slot'}) · full",
            synthetic,
        )
        names.append(name)
    return names


def metrics_plot(metrics, output, synthetic):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.1))
    fig.subplots_adjust(top=0.77, bottom=0.24, left=0.07, right=0.99, wspace=0.28)
    fields = (
        "medium_selection_rate",
        "medium_label_capture_rate",
        "label_slot_alignment_rate",
    )
    titles = (
        "Medium selections / all rows",
        "Medium→medium / official medium",
        "Matching label / all rows",
    )
    for ax, field, title in zip(axes, fields, titles, strict=True):
        for i, (group, color) in enumerate(
            zip(
                ("full", "not_truncated", "truncated"),
                ("#71829a", "#168a94", "#da963e"),
                strict=True,
            )
        ):
            data = (
                metrics.loc[metrics.group.eq(group)]
                .set_index("method")
                .reindex(METHODS)
            )
            values = data[field].to_numpy(float)
            positions = np.arange(3) + (i - 1) * 0.25
            ax.bar(
                positions,
                np.nan_to_num(values),
                width=0.23,
                color=color,
                label=group.replace("_", " "),
            )
            for x, value, count in zip(positions, values, data.n, strict=True):
                ax.text(
                    x,
                    (value if np.isfinite(value) else 0) + 0.018,
                    "N/A" if not np.isfinite(value) else f"{value:.0%}",
                    ha="center",
                    fontsize=8,
                    rotation=90,
                    va="bottom",
                )
        ax.set(
            title=title,
            xticks=range(3),
            xticklabels=["R0", "R1 projection", "R2 projection"],
            ylim=(0, 1.15),
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False
    )
    name = "medium-and-truncation.png"
    save(
        fig,
        output / name,
        "Medium coverage and label-slot alignment · truncation strata",
        synthetic,
    )
    return name


def plot_robustness(pairs, output, synthetic):
    records = []
    for method in METHODS:
        for group, part in (
            ("all_pairs", pairs.loc[pairs.method.eq(method)]),
            (
                "both_not_truncated",
                pairs.loc[
                    pairs.method.eq(method)
                    & ~pairs.was_truncated
                    & ~pairs.full_was_truncated
                ],
            ),
            (
                "either_truncated",
                pairs.loc[
                    pairs.method.eq(method)
                    & (pairs.was_truncated | pairs.full_was_truncated)
                ],
            ),
        ):
            records.append(
                {
                    "method": method,
                    "group": group,
                    "paired_n": len(part),
                    **{
                        change + "_rate": safe_ratio(
                            part.transition.eq(change).sum(), len(part)
                        )
                        for change in ("down", "same", "up")
                    },
                }
            )
    table = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.subplots_adjust(top=0.76, bottom=0.22, left=0.14, right=0.99, wspace=0.75)
    for ax, group in zip(
        axes, ("all_pairs", "both_not_truncated", "either_truncated"), strict=True
    ):
        data = table.loc[table.group.eq(group)].set_index("method").reindex(METHODS)
        left = np.zeros(3)
        for change, color in zip(
            ("down", "same", "up"), ("#5794c2", "#1a8d83", "#c77970"), strict=True
        ):
            values = data[change + "_rate"].to_numpy(float)
            ax.barh(
                range(3),
                np.nan_to_num(values),
                left=left,
                color=color,
                height=0.6,
                label=change,
            )
            for i, value in enumerate(values):
                if np.isfinite(value) and value >= 0.1:
                    ax.text(
                        left[i] + value / 2,
                        i,
                        f"{value:.0%}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=9,
                    )
            left += np.nan_to_num(values)
        for i, count in enumerate(data.paired_n):
            if count == 0:
                ax.text(
                    0.5, i, "N/A · no pairs", ha="center", va="center", color="#64748b"
                )
        ax.set(
            yticks=range(3),
            yticklabels=[
                f"{NAMES[m]}\nn={n}"
                for m, n in zip(METHODS, data.paired_n, strict=True)
            ],
            xlim=(0, 1),
            title=group.replace("_", " "),
        )
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False
    )
    name = "paired-robustness-retention.png"
    save(fig, output / name, "Full → robustness · paired action retention", synthetic)
    return name, table


def render(args):
    rows, summary, manifest, checks = validate_inputs(args)
    pairs = paired_robustness(rows, summary)
    output = args.output_dir.resolve()
    if output.exists():
        raise ValueError("Output directory must not exist; preserve prior results")
    output.mkdir(parents=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#334155",
            "text.color": "#172b45",
            "axes.edgecolor": "#bfccd8",
        }
    )
    full = rows.loc[rows.evaluation_split.eq("full")].copy()
    reference = full.loc[full.method.eq("ordinal")]
    images, tables = [], {}
    for dimension in ("category", "domain", "arena_source_dataset"):
        pages, records = plot_stacks(full, dimension, output, args.synthetic)
        images.extend(pages)
        tables["full-" + dimension + "-distributions"] = pd.DataFrame(records)
    images.extend(plot_conditionals(full, output, args.synthetic))
    metric_rows = []
    for method in METHODS:
        method_rows = full.loc[full.method.eq(method)]
        for group, part in (
            ("full", method_rows),
            ("not_truncated", method_rows.loc[~method_rows.was_truncated]),
            ("truncated", method_rows.loc[method_rows.was_truncated]),
        ):
            metric_rows.append(
                {"method": method, "group": group, **group_metrics(part)}
            )
    tables["full-medium-and-truncation"] = pd.DataFrame(metric_rows)
    images.append(
        metrics_plot(tables["full-medium-and-truncation"], output, args.synthetic)
    )
    name, tables["paired-robustness-retention"] = plot_robustness(
        pairs, output, args.synthetic
    )
    images.append(name)
    tables["paired-robustness-rows"] = pairs[
        [
            "method",
            "sample_id",
            "source_sample_id",
            "category",
            "domain",
            "difficulty_label",
            "action_index",
            "full_action_index",
            "transition",
            "was_truncated",
            "full_was_truncated",
        ]
    ]
    category_pairs = []
    for (method, category), part in pairs.groupby(["method", "category"]):
        category_pairs.append(
            {
                "method": method,
                "category": category,
                "paired_n": len(part),
                "same_action_rate": float(part.transition.eq("same").mean()),
            }
        )
    tables["paired-robustness-by-category"] = pd.DataFrame(
        category_pairs, columns=["method", "category", "paired_n", "same_action_rate"]
    )
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    report = {
        "schema_version": "routing-arena-visuals-v1",
        "synthetic_only": args.synthetic,
        "source_report_manifest_sha256": sha(args.report_dir / "manifest.json"),
        "protocol_sha256": manifest["protocol_sha256"],
        "script_sha256": sha(__file__),
        "frozen_projection_policy": summary["projection_policy"],
        "new_inference": False,
        "new_policy_selection": False,
        "full_independent_rows": len(reference),
        "paired_variants": int(len(pairs) / 3),
        "counts_not_additive": True,
        "checks": checks,
        "prepared_metadata_verified": args.prepared_dir is not None,
        "images": images,
        "tables": {
            name: json.loads(table.to_json(orient="records"))
            for name, table in tables.items()
            if name != "paired-robustness-rows"
        },
        "limitations": [
            "R1/R2 action-slot projection is not a trained difficulty classifier or an Arena Score.",
            "Official difficulty labels and candidate action slots are different targets.",
            "Validation T1 solve-rate labels and RouterArena official Difficulty have different definitions; accuracy differences are not same-task degradation estimates.",
            "Historical RouterArena evaluation was observed before this training; this is not a pristine blind test.",
            "R1/R2 argmax projections do not measure candidate success rates or active production routing.",
            "Truncated and nontruncated subsets may differ in category/label composition; no causal truncation claim.",
            "Robustness variants are paired with full; counts are not independent or additive.",
            "Empty denominators are N/A; no observations are invented.",
            "This is single-seed descriptive analysis, not a significance test or deployment approval.",
        ],
    }
    (output / "visuals.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    warning = (
        "<b>합성 QA 데이터입니다. 실제 실험 결과가 아닙니다.</b><br>"
        if args.synthetic
        else ""
    )
    metrics = tables["full-medium-and-truncation"].copy()
    for field in (
        "official_medium_fraction",
        "medium_selection_rate",
        "medium_label_capture_rate",
        "label_slot_alignment_rate",
    ):
        metrics[field] = metrics[field].map(
            lambda x: "N/A" if pd.isna(x) else f"{x:.1%}"
        )
    pictures = "".join(
        f'<section><a href="{name}"><img src="{name}" alt="{html.escape(name)}" loading="lazy"></a><p class="caption">{html.escape(name)} · PNG를 누르면 원본을 엽니다.</p></section>'
        for name in images
    )
    links = "".join(f'<a href="{name}.csv">{name}.csv</a>' for name in tables)
    (output / "index.html").write_text(
        f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RouterArena 라벨·라우팅 진단</title><style>
body{{margin:0;background:#f2f5f8;color:#172b45;font:16px/1.65 system-ui,sans-serif}}main{{max-width:1240px;margin:auto;padding:36px 24px 70px}}h1{{font-size:34px;line-height:1.3}}h2{{font-size:23px}}p{{color:#45566f}}section{{background:white;border:1px solid #dce4ed;border-radius:12px;padding:24px;margin-top:20px}}img{{width:100%;height:auto}}.note{{border-left:4px solid #087f8c;background:#e6f4f3;padding:18px 24px}}.caption{{font-size:12px;color:#65768a}}.table{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:12px}}th,td{{padding:8px;border-bottom:1px solid #e2e8ef;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}a{{color:#087f8c}}.links{{display:flex;flex-wrap:wrap;gap:10px;font-size:13px}}code{{word-break:break-all}}
</style><main><h1>RouterArena · 실제 라벨과<br>세 모델의 선택 분포</h1><p>정제된 full {len(reference):,}건, 대응하는 robustness 변형 {len(pairs) // 3:,}쌍. λ=0과 512토큰 설정을 고정한 보고서의 저장 예측만 시각화했습니다.</p><div class="note">{warning}<b>R1·R2의 그래프는 action 슬롯의 라벨 투영입니다.</b><br>R0는 easy/medium/hard를 학습한 난이도 분류기입니다. R1·R2는 low/medium/high 선택을 같은 순서의 라벨 이름에 투영한 진단이며, 난이도 분류 성능·실제 서비스 품질·공식 Arena Score를 뜻하지 않습니다.</div>
<section><h2>읽는 방법</h2><p>카테고리·도메인·출처 그래프의 첫 패널은 공식 난이도 라벨이고, 다음 세 패널은 각 모델의 선택 분포입니다. 모든 카테고리를 페이지로 나눠 표시합니다. 별표(*)는 30건 미만입니다. 조건부 표의 행은 100%이며, 표본이 없는 행은 N/A로 표시합니다.</p><p>개발검증의 T1 정답률 기반 난이도와 RouterArena의 공식 Difficulty는 정의가 다릅니다. 정확도 차이를 같은 과제에서의 성능 하락이라고 단정하지 않습니다. 최초 학습 전에 과거 RouterArena 평가를 확인한 이력이 있으므로 완전히 보지 않은 blind test도 아닙니다. R1·R2의 argmax 투영은 세 후보 모델의 실측 성공률이나 현재 운영 라우팅 경로가 아닙니다.</p><p>Medium selection은 전체 중 medium 슬롯 선택 비율입니다. Medium-label capture는 공식 medium 질문 중 medium 슬롯을 선택한 비율이며, R0에서만 난이도 medium recall로 해석합니다. Truncated는 원래 입력이 512토큰을 초과한 경우입니다. 두 길이 집단의 카테고리 구성이 다를 수 있어 차이를 토큰 절단의 인과 효과로 해석하지 않습니다.</p><p>Robustness는 같은 원본 질문에 대응하는 변형입니다. full과 robustness 건수를 합산하지 않습니다. same은 선택 유지, down/up은 슬롯 순서 하향/상향이며, 품질 개선·악화를 직접 뜻하지 않습니다.</p></section>
{pictures}<section><h2>분모를 포함한 수치</h2><div class="table">{metrics.to_html(index=False, border=0)}</div></section><section><h2>재현과 원본 수치</h2><p>입력 report seal·파일 SHA, 3개 모델의 sample ID·공식 라벨·카테고리 일치, logits→projection 재구성, 원본 confusion 및 세부 분류 집계, token/truncation과 paired robustness를 확인했습니다. 추가 모델 추론·λ 선택·배포는 수행하지 않았습니다.</p><p>단일 seed의 기술 통계입니다. 원시 report manifest SHA: <code>{report["source_report_manifest_sha256"]}</code></p><p><a href="visuals.json">visuals.json</a> · <a href="manifest.json">출력 manifest</a></p><div class="links">{links}</div></section></main></html>""",
        encoding="utf-8",
    )
    files = {
        path.name: {"sha256": sha(path), "size_bytes": path.stat().st_size}
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "routing-arena-visuals-artifacts-v1",
                "synthetic_only": args.synthetic,
                "source_report_manifest_sha256": report[
                    "source_report_manifest_sha256"
                ],
                "files": files,
            },
            indent=2,
        )
        + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--expected-report-sha256")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Prominent synthetic QA watermark; never use for real measurements",
    )
    args = parser.parse_args()
    report = render(args)
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "synthetic_only": args.synthetic,
                "full_rows": report["full_independent_rows"],
                "paired_variants": report["paired_variants"],
                "figures": len(report["images"]),
            }
        )
    )


if __name__ == "__main__":
    main()
