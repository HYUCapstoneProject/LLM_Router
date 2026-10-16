"""EV-10 / RT-08 data preparation (00_prepare): model-group tiers, labels, service costs and splits.

One CPU step that turns the frozen inputs into the tables every later notebook reads:
  * A split (RouterBench + xRouteBench) with v2-audited per-model outcomes.
  * LLMRouterBench and SPROUT re-split into train 50 / dev 15 / test 35 by split group.
  * Per source, near-frontier models grouped into three price tiers (train statistics only).
  * Per question: group success probability for each tier, the three-level "needed tier" label
    (cheapest tier with probability >= tau), and the fixed service cost of each tier (1:3:5).
  * RT-08 inputs: the old-definition label (overall solve rate, T1 .27/.64) recomputed on v2.
Test outcomes are written but their content is never summarized here beyond row counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TIERS = ("low", "medium", "high")
SPLIT_ROLES = ("train", "dev", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------- tier grouping
def kmeans_bands(values: np.ndarray, k: int = 3, iters: int = 200) -> np.ndarray:
    """Deterministic 1-D k-means; returns band index ordered by center (0 = cheapest)."""
    centers = np.quantile(values, np.linspace(0, 1, k))
    for _ in range(iters):
        assign = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        new = np.array([values[assign == j].mean() if (assign == j).any() else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    rank = np.argsort(np.argsort(centers))
    return rank[assign]


def group_models(stats: pd.DataFrame, delta: float) -> pd.DataFrame:
    """stats: model, quality, cost (train only). Keep near-frontier models, split into three price bands."""
    s = stats.sort_values(["cost", "quality"], ascending=[True, False]).reset_index(drop=True)
    s["gap_to_frontier"] = s["quality"].cummax() - s["quality"]
    s["kept"] = s["gap_to_frontier"] <= delta + 1e-12
    kept = s[s.kept]
    if len(kept) < 3:
        raise ValueError("need at least three near-frontier models")
    s["tier"] = None
    s.loc[s.kept, "tier"] = [TIERS[b] for b in kmeans_bands(np.log(kept["cost"].to_numpy()))]
    means = s[s.kept].groupby("tier")["quality"].mean().reindex(TIERS)
    if not (means.is_monotonic_increasing and means.notna().all()):
        raise ValueError(f"tier quality is not ordered: {means.to_dict()}")
    return s


# ---------------------------------------------------------------- RouterArena overlap keys
def alnum(text) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def question_body(text) -> str:
    """Question stem before the first lettered option line, normalized."""
    return alnum(re.split(r"\n\s*(?:\(?[A-J][.)]\s)", str(text), maxsplit=1)[0])


# ---------------------------------------------------------------- labels
def needed_tier(probs: np.ndarray, tau: float) -> np.ndarray:
    """Cheapest tier whose success probability >= tau; else the most likely tier (ties -> cheaper)."""
    sure = probs >= tau - 1e-12
    first_sure = np.where(sure.any(1), sure.argmax(1), -1)
    best = probs.argmax(1)  # argmax picks the first (cheaper) tier on ties
    return np.where(first_sure >= 0, first_sure, best)


def old_definition_label(rate: np.ndarray, hard: float, easy: float) -> np.ndarray:
    return np.where(rate >= easy, "easy", np.where(rate < hard, "hard", "medium"))


# ---------------------------------------------------------------- splits
def assign_split(group_ids: pd.Series, fractions: dict, seed: int, forced_test: set) -> pd.Series:
    """Deterministic hash split by group; forced groups go to test."""
    cuts = np.cumsum([fractions[r] for r in SPLIT_ROLES])
    def role(g):
        if g in forced_test:
            return "test"
        u = int(hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()[:12], 16) / float(16 ** 12)
        return SPLIT_ROLES[int(np.searchsorted(cuts, u, side="right"))]
    return group_ids.astype(str).map(role)


# ---------------------------------------------------------------- A split outcomes
def accepted_matrices(config_path: Path, root: Path, src_root: Path) -> dict:
    """Per-model accepted (True/False/NA) matrices exactly as the A-split builder computes them.

    Reuses the builder's own loaders so the acceptance rule (exact tasks == 1.0, continuous >= .70)
    and, when the config has an outcome_audit section, the v2 corrections and exclusions match.
    Returns {source: DataFrame indexed by our sample_id}; RouterBench exclusions are absent rows.
    """
    sys.path.insert(0, str(src_root))
    from llm_router_labeling.routing_data import preprocessing as builder
    cfg = json.loads(Path(config_path).read_text())
    audit = builder.load_outcome_audit(cfg, root)
    structured = cfg.get("input_contract") == "structure-preserving-v2"
    x = cfg["xroutebench"]
    data_dir = root / str(x.get("data_dir", f"ml/data/raw/xroutebench/{x['revision']}"))
    files = {name: data_dir / str(spec["filename"]) for name, spec in x["sources"].items()} if "sources" in x else \
        {"train": data_dir / x["train_file"], "test": data_dir / x["test_file"], "candidates": data_dir / x["candidates_file"]}
    candidates = pd.read_parquet(files["candidates"])["model_name"].astype(str).tolist()
    frames = []
    for split in ("train", "test"):
        raw = builder._validate_xroute_rows(pd.read_parquet(files[split]), split=split, candidates=candidates,
                                            exact_metrics=set(x["exact_metrics"]), continuous_metrics=set(x["continuous_metrics"]),
                                            structured=structured, outcome_audit=audit)
        pivoted, accepted, _ = builder._pivot_xroute(raw, source_split=split, candidates=candidates,
                                                     minimum_valid=int(x.get("minimum_valid_models", 12)), structured=structured)
        frames.append(accepted)
    xr = pd.concat(frames)
    xr = xr[~xr.index.duplicated()]
    _, rb = builder._load_routerbench(cfg, root, audit)
    return {"routerbench": rb.astype("Float64").astype(float), "xroutebench": xr.astype("Float64").astype(float)}


# ---------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--repo", required=True, help="checkout holding ml/src and committed inputs")
    p.add_argument("--data-root", default=None, help="root for git-ignored data (defaults to --repo)")
    p.add_argument("--output-dir", required=True)
    a = p.parse_args(argv)
    repo, out = Path(a.repo), Path(a.output_dir)
    cfg = json.loads(Path(a.config).read_text())
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path(a.data_root) if a.data_root else repo
    inputs = {k: (repo / v if (repo / v).exists() else data_root / v) for k, v in cfg["inputs"].items()}
    manifest = {"config": cfg, "inputs": {k: sha256_file(v) for k, v in inputs.items() if v.is_file()}}
    tau, delta, thr = cfg["tau"], cfg["frontier_delta"], cfg["success_threshold"]
    svc = np.array([cfg["service_cost"][t] for t in TIERS], float)
    frames, groups = [], {}

    # A split: train, D_dev, sealed test (RB + xRB) with v2 per-model outcomes.
    a_roles = {"train": inputs["a_train"], "dev": inputs["a_dev"],
               "test": [inputs["a_test_routerbench"], inputs["a_test_xroutebench"]]}
    a = []
    for role, path in a_roles.items():
        part = pd.concat([pd.read_parquet(x) for x in (path if isinstance(path, list) else [path])], ignore_index=True)
        part["role"] = role
        a.append(part)
    a = pd.concat(a, ignore_index=True)
    stats_a = pd.read_parquet(inputs["a_candidate_statistics"])
    v2 = accepted_matrices(inputs["v2_data_config"], data_root, repo / "ml/src")
    v1 = accepted_matrices(inputs["v1_data_config"], data_root, repo / "ml/src")
    for source, part in a.groupby("source_dataset"):
        idx = part.index
        succ = v2[source].reindex(part.sample_id).set_axis(idx)
        excl = ~part.sample_id.isin(v2[source].index).to_numpy()
        rate_v1 = v1[source].reindex(part.sample_id).mean(axis=1).to_numpy()
        st = stats_a[stats_a.source_dataset.eq(source)].rename(columns={"model_name": "model", "mean_quality": "quality", "mean_cost": "cost"})
        g = group_models(st[["model", "quality", "cost"]], delta)
        groups[source] = g.to_dict("records")
        frame = part[["sample_id", "source_dataset", "role", "model_input_text"]].copy()
        frame["excluded_v2"] = excl
        for t in TIERS:
            members = [m for m in g.loc[g.tier.eq(t), "model"] if m in succ.columns]
            frame[f"p_{t}"] = succ.loc[idx, members].mean(axis=1, skipna=True).to_numpy()
        frame["old_rate_v1_check"] = rate_v1
        frame["old_rate_v2"] = succ.loc[idx].mean(axis=1, skipna=True).to_numpy()
        frame["old_label_v2"] = old_definition_label(frame["old_rate_v2"].to_numpy(), cfg["old_t1"]["hard"], cfg["old_t1"]["easy"])
        frame["old_label_v1"] = part["difficulty_label"].to_numpy()
        for t in TIERS:  # representative arms (RT-08 "대표 모델 기준")
            arm = part[f"model_{t}"]
            frame[f"arm_success_v2_{t}"] = [succ.at[i, m] if m in succ.columns else np.nan for i, m in zip(idx, arm)]
        frames.append(frame)

    # External sources: re-split by split group, Arena overlap forced to test.
    sys.path.insert(0, str(repo / "ml/src"))
    arena = pd.read_parquet(inputs["routerarena"])
    rule = cfg["routerarena_overlap"]
    arena_body = set(arena["Question"].map(lambda q: question_body(q)[: rule["body_chars"]]))
    arena_text = set(arena["Question"].map(lambda q: alnum(q)[: rule["text_chars"]]))
    sprout_cost = pd.read_csv(inputs["sprout_model_cost"]) if "sprout_model_cost" in inputs else None
    for source, spec in cfg["external"].items():
        ext = pd.read_parquet(inputs[spec["input"]])
        ext = ext[ext.exclusion_reason.isna()]
        models = sorted(c[len("success_"):] for c in ext.columns if c.startswith("success_"))
        text = ext["model_input_text"].astype(str)
        overlap = text.map(lambda q: question_body(q)[: rule["body_chars"]]).isin(arena_body) | \
            text.map(lambda q: alnum(q)[: rule["text_chars"]]).isin(arena_text)
        forced = set(ext.loc[overlap | ext["slice"].isin(spec.get("test_only_slices", [])), "split_group_id"].astype(str))
        ext = ext.assign(role=assign_split(ext["split_group_id"], cfg["external_split"], cfg["split_seed"], forced),
                         arena_overlap=overlap.to_numpy())
        train = ext[ext.role.eq("train") & ext["slice"].isin(spec["train_slices"])]
        if source == "sprout":
            sc = sprout_cost[sprout_cost.slice.eq("secondary_general_chat")]
            col = {m.replace("-", "_").replace(".", "_"): m for m in sc.model}
            mean_cost = {m: float(sc.loc[sc.model.eq(col.get(m, m)), "mean_cost_usd"].iloc[0]) for m in models}
        else:
            mean_cost = {m: float(train[f"cost_{m}"].mean()) for m in models}
        st = pd.DataFrame({"model": models, "quality": [train[f"success_{m}"].mean() for m in models],
                           "cost": [mean_cost[m] for m in models]})
        g = group_models(st, delta)
        groups[source] = g.to_dict("records")
        frame = ext[["sample_id", "role", "model_input_text", "slice", "arena_overlap"]].copy()
        frame.insert(1, "source_dataset", source)
        frame["excluded_v2"] = False
        for t in TIERS:
            members = list(g.loc[g.tier.eq(t), "model"])
            frame[f"p_{t}"] = ext[[f"success_{m}" for m in members]].mean(axis=1, skipna=True).to_numpy()
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    probs = data[[f"p_{t}" for t in TIERS]].to_numpy(float)
    data["needed_tier"] = [TIERS[i] for i in needed_tier(np.nan_to_num(probs, nan=0.0), tau)]
    for t, c in zip(TIERS, svc):
        data[f"service_cost_{t}"] = c
    data["source_weight"] = 1.0 / data.groupby(["source_dataset", "role"])["sample_id"].transform("count")
    data["source_weight"] *= data.groupby("role")["sample_id"].transform("count") / data["source_dataset"].nunique()

    for role in SPLIT_ROLES:
        part = data[data.role.eq(role)].reset_index(drop=True)
        path = out / f"{role}.parquet"
        part.to_parquet(path, index=False)
        manifest.setdefault("outputs", {})[path.name] = {"sha256": sha256_file(path), "rows": len(part)}
    (out / "groups.json").write_text(json.dumps(groups, ensure_ascii=False, indent=2, default=float))
    summary = {"rows": data.groupby(["source_dataset", "role"]).size().unstack(fill_value=0).to_dict("index"),
               "arena_overlap_rows": int(data.get("arena_overlap", pd.Series(dtype=bool)).fillna(False).sum()),
               "needed_tier_mix_train": data[data.role.eq("train")].groupby("source_dataset")["needed_tier"]
               .value_counts(normalize=True).round(3).unstack(fill_value=0).to_dict("index"),
               "groups": {s: {t: [r["model"] for r in rec if r["tier"] == t] for t in TIERS} for s, rec in groups.items()}}
    manifest["summary"] = summary
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    return summary


if __name__ == "__main__":
    main()
