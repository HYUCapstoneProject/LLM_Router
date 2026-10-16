"""Actual problem-solving rates on the sealed test under the recorded (v1) and audited (v2) grading.

First-try success = the routed tier's recorded success. Final resolution = success at any tier
reached by escalating after each failure. v2 applies the 2026-10-01 outcome-audit corrections to
the same rows (RouterBench exclusions dropped). RouterArena has no per-model outcomes, so no rate.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special

TIERS = ("low", "medium", "high")
LABELS = ("easy", "medium", "hard")
R1_LOCKED = {"routerbench": 0.1, "xroutebench": 0.03}
INK, MUTED, GRID = "#0b0b0b", "#6f6e69", "#e4e3df"
LIGHT, DARK = "#a9c9ef", "#2a78d6"


def xroute_success_v2(test, raw_test, audit_dir, src_root):
    """Rebuild each test row's per-arm cell key exactly as the dataset builder did, then apply corrections."""
    sys.path.insert(0, str(src_root))
    from llm_router_labeling.shared.structured_text import structured_duplicate_key
    from llm_router_labeling.routing_data.outcome_audit import xroute_cell_key
    raw = pd.read_parquet(raw_test)
    ident = raw.task_id.where(raw.task_id.notna(), raw.embedding_id).astype(str)
    key = raw.task_name.astype(str) + "|" + ident + "|" + raw["query"].map(structured_duplicate_key)
    raw["sample_id"] = key.map(lambda k: "xroutebench:" + hashlib.sha256(str(k).encode("utf-8")).hexdigest()[:20])
    raw["cell_key"] = xroute_cell_key(raw)
    fixes = pd.read_parquet(Path(audit_dir) / "xroutebench-corrections.parquet").set_index("cell_key").verified_accepted
    raw["fixed"] = raw.cell_key.map(fixes)
    success = test[[f"success_{t}" for t in TIERS]].to_numpy(float).copy()
    lookup = raw.set_index(["sample_id", "model_name"]).fixed
    hits = 0
    for i, row in enumerate(test.itertuples()):
        for j, t in enumerate(TIERS):
            value = lookup.get((row.sample_id, getattr(row, f"model_{t}")))
            if isinstance(value, pd.Series):
                value = value.iloc[0]
            if value is not None and pd.notna(value):
                success[i, j] = max(success[i, j], float(value))  # xRouteBench fixes never revoke a success
                hits += 1
    return success, np.ones(len(test), bool), hits


def routerbench_success_v2(test, audit_dir):
    fixes = pd.read_parquet(Path(audit_dir) / "routerbench-corrections.parquet").set_index(["sample_id", "model"]).verified_accepted
    excluded = set(pd.read_parquet(Path(audit_dir) / "routerbench-exclusions.parquet").sample_id)
    rid = test.sample_id.str.replace("routerbench:", "", regex=False)
    success = test[[f"success_{t}" for t in TIERS]].to_numpy(float).copy()
    hits = 0
    for i, (r, row) in enumerate(zip(rid, test.itertuples())):
        for j, t in enumerate(TIERS):
            key = (r, getattr(row, f"model_{t}"))
            if key in fixes.index:
                success[i, j] = float(fixes[key])
                hits += 1
    return success, ~rid.isin(excluded).to_numpy(), hits


def rates(success, tier, keep):
    s, tier = success[keep], np.asarray(tier)[keep]
    solved = np.zeros(len(s), bool)
    for t in range(3):
        solved |= (tier <= t) & (s[:, t] == 1)
    return float(s[np.arange(len(s)), tier].mean()), float(solved.mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("test", "scores-root", "f4-run", "f5-run", "audit-dir", "xroute-raw-test", "src-root", "output-dir"):
        p.add_argument("--" + flag, required=True)
    a = p.parse_args()
    out = Path(a.output_dir)
    test = pd.read_parquet(a.test)
    finals = {}
    for name, run, arm, seeds in (("F0 (R0 1.0)", a.f5_run, "f0_frozen", [0]), ("F4", a.f4_run, "f4_bidirectional", [42, 43, 44]),
                                  ("F5", a.f5_run, "f5_matched_prior", [42, 43, 44])):
        finals[name] = [pd.read_parquet(Path(run) / "scores" / arm / f"seed-{s}" / "final/test/predictions.parquet")
                        .set_index("sample_id").prediction.map(LABELS.index) for s in seeds]
    rows, notes = [], {}
    for source in ("routerbench", "xroutebench"):
        frame = test[test.source_dataset.eq(source)].sort_values("sample_id").reset_index(drop=True)
        ids = frame.sample_id
        v1 = frame[[f"success_{t}" for t in TIERS]].to_numpy(float)
        if source == "routerbench":
            v2, keep2, hits = routerbench_success_v2(frame, a.audit_dir)
        else:
            v2, keep2, hits = xroute_success_v2(frame, a.xroute_raw_test, a.audit_dir, a.src_root)
        notes[source] = {"rows": len(frame), "v2_rows": int(keep2.sum()), "corrected_cells_matched": hits,
                         "success_changed_cells": int((v1[keep2] != v2[keep2]).sum())}
        r1 = special.expit(pd.read_parquet(Path(a.scores_root) / f"r1/a-scores/test_{source}-8192.parquet")
                           .set_index("sample_id").loc[ids, ["score_low", "score_medium", "score_high"]].to_numpy(float))
        dc = frame[[f"decision_cost_{t}" for t in TIERS]].iloc[0].to_numpy(float)
        best_v1 = np.where(v1.max(1) == 1, v1.argmax(1), 0)
        best_v2 = np.where(v2.max(1) == 1, v2.argmax(1), 0)
        methods = {"start low (always-low)": [np.zeros(len(frame), int)],
                   "start medium (always-medium)": [np.ones(len(frame), int)],
                   "high only (always-high)": [np.full(len(frame), 2)],
                   **{k: [v.loc[ids].to_numpy() for v in vs] for k, vs in finals.items()},
                   "R1 locked policy": [np.argmax(r1 - R1_LOCKED[source] * dc, 1)]}
        for name, preds in methods.items():
            for grading, success, keep in (("v1", v1, np.ones(len(frame), bool)), ("v2", v2, keep2)):
                first, final = zip(*[rates(success, p, keep) for p in preds])
                rows.append({"source": source, "method": name, "grading": grading,
                             "first_try_success": float(np.mean(first)), "final_resolution": float(np.mean(final))})
        for grading, success, keep, best in (("v1", v1, np.ones(len(frame), bool), best_v1), ("v2", v2, keep2, best_v2)):
            first, final = rates(success, best, keep)
            rows.append({"source": source, "method": "oracle (cheapest solving tier)", "grading": grading,
                         "first_try_success": first, "final_resolution": final})
    table = pd.DataFrame(rows)
    table.to_csv(out / "resolution-rates.csv", index=False)
    pd.Series(notes).to_json(out / "resolution-rates-notes.json", indent=2)

    order = ["start low (always-low)", "start medium (always-medium)", "high only (always-high)", "F0 (R0 1.0)",
             "F5", "F4", "R1 locked policy", "oracle (cheapest solving tier)"]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1, 0.6]})
    titles = {"routerbench": "RouterBench test (3,473; v2 drops 28 excluded)", "xroutebench": "xRouteBench test (3,729)"}
    for ax, source in zip(axes, ("routerbench", "xroutebench")):
        part = table[table.source.eq(source)]
        for y, name in enumerate(order[::-1]):
            v2 = part[(part.method == name) & (part.grading == "v2")].iloc[0]
            v1 = part[(part.method == name) & (part.grading == "v1")].iloc[0]
            ax.barh(y + 0.18, v2.final_resolution, height=0.34, color=DARK, label="final resolution (escalate on failure)" if y == 0 else None)
            ax.barh(y - 0.18, v2.first_try_success, height=0.34, color=LIGHT, label="first-try success" if y == 0 else None)
            ax.text(v2.final_resolution + 0.01, y + 0.18, f"{v2.final_resolution:.1%}", va="center", fontsize=8, color=INK)
            ax.text(v2.first_try_success + 0.01, y - 0.18, f"{v2.first_try_success:.1%}", va="center", fontsize=8, color=INK)
            for value, offset in ((v1.final_resolution, 0.18), (v1.first_try_success, -0.18)):
                ax.plot([value, value], [y + offset - 0.17, y + offset + 0.17], color=INK, lw=1.5,
                        label="v1 recorded grading" if (y == 0 and offset > 0) else None)
        ax.set_yticks(range(len(order)), order[::-1])
        ax.set(title=titles[source], xlim=(0, 1.08), xlabel="Share of questions solved (bars: audited v2 grading)")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    axes[1].set_yticklabels([])
    axes[0].legend(loc="lower left", fontsize=8, frameon=False, bbox_to_anchor=(0, 1.06), ncol=3)
    ax = axes[2]
    ax.axis("off")
    ax.set_title("RouterArena (8,379)")
    ax.text(0.0, 0.6, "No per-model responses or correctness\nrecords exist: only official difficulty\n"
            "labels and reference answers.\n\nA solve rate needs new LLM calls\n(e.g. the service tiers), so it is\n"
            "not shown here.", fontsize=8.5, color=INK, va="center", wrap=True)
    fig.suptitle("Actual solve rates on the sealed test: escalation lifts every policy to ~85-94%; routers mainly change the first try",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "resolution-rates.png", dpi=170)
    print(table.pivot_table(index=["source", "method"], columns="grading", values=["first_try_success", "final_resolution"]).round(3).to_string())
    print(notes)


if __name__ == "__main__":
    main()
