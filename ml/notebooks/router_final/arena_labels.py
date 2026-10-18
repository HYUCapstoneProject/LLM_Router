"""EV-10 RouterArena label agreement: how well each router's 3-level call matches RouterArena's official Difficulty.

RouterArena has a single difficulty label per question (easy/medium/hard, Bloom-level, judged by DeepSeek-V3.1) and
no per-model results, so only classification agreement can be measured (no cost or resolution).

Common rule for every router: its scores are turned into per-tier success probabilities with one adapter fitted on
the pooled dev of the four sources (per tier isotonic, the 07b adapter without the source split, since RouterArena
has no source of its own), and the tau = .8 rule picks the cheapest tier with p >= .8 (else the most likely). That
start tier low/medium/high is read as easy/medium/hard ("tau .8 decision"). Because that rule is a cost decision and
leans to hard, a second reading ranks questions by difficulty (1 - mean calibrated success over tiers) and cuts the
ranking at the dev needed-tier shares (cut points fitted on dev, never on RouterArena labels): "dev-share cut".
Routers that classify directly (R0-group, RT-08) are also reported with their own argmax. Settings are the ones 07b
locked.

Rows: RouterArena questions without a Context (the routers saw only the question text; 6,797), and all 8,400 for
reference. Metrics: accuracy, macro-F1, per-label recall, Cohen's kappa, Spearman with the ordinal label, confusion,
and accuracy by Domain (9) and Category (28). Baselines: always easy (majority), label-frequency guess.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_track as C  # noqa: E402
import lock as L  # noqa: E402

LABELS = ("easy", "medium", "hard")


def pooled_inputs(run_dir: Path, repo: Path) -> tuple:
    data = L.load_outcomes(run_dir)
    excluded, _ = L.evaluation_exclusions(repo)
    group_cols = [f"p_{t}" for t in L.TIERS]
    dev = data["dev"][~data["dev"].sample_id.isin(excluded)]
    dev = dev[dev[group_cols].notna().all(axis=1)].reset_index(drop=True)
    b = data["blended"]
    table = L.class_table(b.difficulty_label, b[[f"success_{t}" for t in L.TIERS]].to_numpy(float), LABELS)
    return dev, table


def predict(kind: str, scores: pd.DataFrame, dev: pd.DataFrame, table: np.ndarray, arena_ids) -> tuple:
    cols = [f"score_{t}" for t in L.TIERS]
    s = scores.drop_duplicates("sample_id").set_index("sample_id")
    Xd = L.features(kind, s.loc[dev.sample_id, cols].to_numpy(float), table)
    Xa = L.features(kind, s.loc[arena_ids, cols].to_numpy(float), table)
    Y = dev[[f"p_{t}" for t in L.TIERS]].to_numpy(float)
    P = L.fit_predict(Xd, Y, Xa)
    Pd = L.fit_predict(Xd, Y, Xd)
    tau = L.decide("tau", P, 0.0, tau=L.TAU)
    # dev-share cut: difficulty = 1 - mean success; cut points reproduce dev's needed-tier shares
    share = dev.needed_tier.value_counts(normalize=True).reindex(list(L.TIERS)).fillna(0).to_numpy()
    dd, da = 1 - Pd.mean(axis=1), 1 - P.mean(axis=1)
    cuts = np.quantile(dd, [share[0], share[0] + share[1]])
    return tau, np.searchsorted(cuts, da, side="right")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    out = {"accuracy": float((y == p).mean())}
    f1 = []
    for k, lab in enumerate(LABELS):
        tp = ((p == k) & (y == k)).sum()
        rec = tp / max((y == k).sum(), 1)
        prec = tp / max((p == k).sum(), 1)
        out[f"recall_{lab}"] = float(rec)
        f1.append(0.0 if rec + prec == 0 else 2 * rec * prec / (rec + prec))
    out["macro_f1"] = float(np.mean(f1))
    po = out["accuracy"]
    pe = sum(((y == k).mean() * (p == k).mean()) for k in range(3))
    out["cohen_kappa"] = float((po - pe) / (1 - pe)) if pe < 1 else float("nan")
    ry, rp = pd.Series(y).rank().to_numpy(), pd.Series(p).rank().to_numpy()
    out["spearman"] = float(np.corrcoef(ry, rp)[0, 1]) if rp.std() > 0 else float("nan")
    out["predicted_share"] = "/".join(f"{(p == k).mean():.0%}" for k in range(3))
    return out


def run(run_dir: Path, repo: Path, output_dir: Path) -> None:
    rec = json.loads((run_dir / "lock" / "selection-lock.json").read_text())
    arena = pd.read_parquet(run_dir / "inputs" / "routerarena-full.parquet",
                            columns=["Global Index", "Difficulty", "Domain", "Category", "Context"])
    arena["sample_id"] = "routerarena:" + arena["Global Index"].astype(str)
    arena["y"] = arena.Difficulty.map({lab: k for k, lab in enumerate(LABELS)})
    arena["with_context"] = arena.Context.fillna("").astype(str).str.strip().ne("")
    dev, table = pooled_inputs(run_dir, repo)
    preds = {}
    models = {m: (v["kind"], run_dir / v["scores_file"]) for m, v in rec["methods"].items()}
    models["rt08"] = ("class_probs", run_dir / "scores-modernbert" / "rt08" / "scores.parquet")
    for name, (kind, path) in sorted(models.items()):
        scores = pd.read_parquet(path)
        a = scores[scores.sample_id.isin(arena.sample_id)]
        if a[[f"score_{t}" for t in L.TIERS]].isna().any(axis=1).any():
            print(f"[arena] {name}: no RouterArena scores, skipped", flush=True)
            continue
        tau, cut = predict(kind, scores, dev, table, arena.sample_id)
        preds[f"{name} | tau .8 decision"], preds[f"{name} | dev-share cut"] = tau, cut
        if kind == "class_probs":  # the classifier's own call
            s = scores.drop_duplicates("sample_id").set_index("sample_id")
            preds[f"{name} (argmax)"] = s.loc[arena.sample_id, [f"score_{t}" for t in L.TIERS]].to_numpy(float).argmax(axis=1)
        print(f"[arena] {name} done", flush=True)
    rng = np.random.default_rng(42)
    prior = arena.y.value_counts(normalize=True).sort_index().to_numpy()
    preds["baseline: always easy"] = np.zeros(len(arena), int)
    preds["baseline: label-frequency guess"] = rng.choice(3, size=len(arena), p=prior)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, domain_rows, cat_rows, conf = [], [], [], {}
    for subset, mask in (("no_context", ~arena.with_context), ("all", np.ones(len(arena), bool))):
        y = arena.y.to_numpy()[mask]
        for name, p in preds.items():
            rows.append({"subset": subset, "router": name, "rows": int(mask.sum()), **metrics(y, p[mask])})
            if subset == "no_context":
                conf[name] = pd.crosstab(pd.Series(y, name="label").map(dict(enumerate(LABELS))),
                                         pd.Series(p[mask], name="pred").map(dict(enumerate(LABELS))))
                part = arena[mask].assign(hit=(p[mask] == y))
                for col, store in (("Domain", domain_rows), ("Category", cat_rows)):
                    for g, gp in part.groupby(col):
                        store.append({"router": name, col.lower(): g, "rows": len(gp), "accuracy": gp.hit.mean(),
                                      "label_mix": "/".join(f"{(gp.y == k).mean():.0%}" for k in range(3))})
    overall = pd.DataFrame(rows)
    overall.to_csv(output_dir / "routerarena-agreement.csv", index=False)
    pd.DataFrame(domain_rows).to_csv(output_dir / "routerarena-by-domain.csv", index=False)
    pd.DataFrame(cat_rows).to_csv(output_dir / "routerarena-by-category.csv", index=False)
    with open(output_dir / "routerarena-confusion.md", "w") as f:
        for name, c in conf.items():
            c = c.reindex(index=list(LABELS), columns=list(LABELS)).fillna(0).astype(int)
            lines = ["| 라벨 \\ 예측 | " + " | ".join(LABELS) + " |", "|---|---:|---:|---:|"]
            lines += [f"| {lab} | " + " | ".join(f"{v:,}" for v in c.loc[lab]) + " |" for lab in LABELS]
            f.write(f"### {name}\n\n" + "\n".join(lines) + "\n\n")
    meta = {"rows": {"all": len(arena), "no_context": int((~arena.with_context).sum())},
            "label_mix_no_context": arena[~arena.with_context].Difficulty.value_counts().to_dict(),
            "adapter": "per tier isotonic fitted on the pooled dev of the four sources (evaluation exclusions dropped)",
            "rule": "tau = .8 start tier read as easy/medium/hard; class models also by argmax",
            "lock_sha256": C.sha256_file(run_dir / "lock" / "selection-lock.json"),
            "routerarena_sha256": C.sha256_file(run_dir / "inputs" / "routerarena-full.parquet"),
            "skipped": sorted(set(models) - {k for k in preds})}
    (output_dir / "routerarena-agreement.json").write_text(json.dumps(meta, indent=1, default=str))
    print(overall[overall.subset.eq("no_context")].sort_values("accuracy", ascending=False)
          [["router", "accuracy", "macro_f1", "cohen_kappa", "spearman", "predicted_share"]].round(3).to_string(index=False))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(argv)
    run(a.run_dir, a.repo, a.output_dir)


if __name__ == "__main__":
    main()
