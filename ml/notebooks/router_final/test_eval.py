"""EV-10 08_test: open the sealed test once and evaluate every router locked by 07b (plan 2.5, 3, 4.4).

Nothing is chosen here. Settings, adapters (calibrated test probabilities), rho, cost targets and lambdas all come
from <run>/lock/selection-lock.json. This step reads the test outcomes (00_prepare test.parquet, group pass
probabilities under v2 grading) and, for the A split, rebuilds the v1 group probabilities from the raw data so the
A test can also be reported under v1 grading (plan 2.5, H6). Questions in the evaluation exclusion lists
(non-English, document-grounded QA) are dropped, as in 07b.

Track 1 (cost-aware): for each router, rule (escalation-aware, one-shot), kappa and cost level, the router's test
value (resolved - kappa x failed attempts) minus the best fixed start tier or mixture at the router's test cost
(the "gain"). Paired group bootstrap (same draws for every router of a source). kappa* = smallest kappa with gain
> .01 and a 95% interval above 0. Bases: main (Claude 1:3:5), group_price and length (sensitivities, plan 2.3),
v1 (A test, point only).
Track 2 (tau rule): start at the cheapest tier with calibrated p >= tau, escalate on failure; needed-tier
accuracy, first-attempt success, resolution, calls and cost over the tau curve.

Writes <output>/ (normally <run>/test, which must not exist: the test is opened once). `--rehearse-on-dev` runs
the same code on dev outcomes into another folder to check the pipeline without reading the test.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import categories as K  # noqa: E402
import common_track as C  # noqa: E402
import lock as L  # noqa: E402

TIERS, SOURCES, A_SOURCES = L.TIERS, L.SOURCES, L.A_SOURCES
EXTERNAL = ("llmrouterbench", "sprout")
MARGIN = 0.01  # plan 2.5: gain > .01
BOOT = {"iterations": 10000, "seed": 42, "confidence": 0.95,
        "unit": "duplicate_group_id (A split) / split_group_id (LLMRouterBench, SPROUT)",
        "pairing": "same group draws for every router, rule, kappa and level of a source; sources resampled independently"}
CHUNK = 500
REFERENCE = ("oracle", "r0_operational")  # reported, never ranked
STRATA = ("solvable", "beyond_partial", "beyond_zero")  # plan 2.5 (10-18): some group >= tau / all < tau / all 0
BLOOM = {"easy": 0, "medium": 1, "hard": 2}


# ---------------------------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------------------------

def envelope_batch(fc: np.ndarray, fv: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorised lock.envelope for three fixed points. fc, fv: (..., 3) costs and values; x: (...) cost."""
    order = np.lexsort((fv, fc), axis=-1)
    c, v = np.take_along_axis(fc, order, -1), np.take_along_axis(fv, order, -1)
    c0, c1, c2 = c[..., 0], c[..., 1], c[..., 2]
    v0, v1, v2 = v[..., 0], v[..., 1], v[..., 2]
    middle = (v1 - v0) * (c2 - c0) > (v2 - v0) * (c1 - c0)  # strictly above the chord: on the upper hull

    def seg(best, ca, va, cb, vb, use):
        t = (x - ca) / np.where(cb > ca, cb - ca, 1.0)
        best = np.where(use & (x >= cb), np.maximum(best, vb), best)
        return np.where(use & (x > ca) & (x < cb), np.maximum(best, va + (vb - va) * t), best)

    best = v0.copy()
    best = seg(best, c0, v0, c1, v1, middle)
    best = seg(best, c1, v1, c2, v2, middle)
    best = seg(best, c0, v0, c2, v2, ~middle)
    return np.where(x <= c0, v0, best)


class Draws:
    """Group bootstrap of column means. Every call with the same (seed, source) sees identical draws."""

    def __init__(self, groups, iterations: int, seed: int, stream: int):
        self.codes, uniq = pd.factorize(pd.Series(groups).astype(str))
        self.groups, self.iterations, self.seed = len(uniq), iterations, [seed, stream]

    def counts(self):
        rng = np.random.default_rng(np.random.SeedSequence(self.seed))
        p = np.full(self.groups, 1.0 / self.groups)
        for start in range(0, self.iterations, CHUNK):
            b = min(CHUNK, self.iterations - start)
            yield start, rng.multinomial(self.groups, p, size=b).astype(np.float32)[:, self.codes]

    def means(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float32)
        out = np.empty((self.iterations, X.shape[1]), np.float32)
        for start, W in self.counts():
            out[start:start + len(W)] = (W @ X) / W.sum(axis=1, keepdims=True)
        return out


def interval(draws: np.ndarray, confidence: float = BOOT["confidence"]) -> tuple:
    a = (1 - confidence) / 2 * 100
    lo, hi = np.percentile(draws, [a, 100 - a], axis=0)
    return lo, hi


def kappa_star(rows: pd.DataFrame) -> float:
    """Smallest kappa whose gain exceeds MARGIN with the interval above 0 (NaN when none)."""
    ok = rows[(rows.gain > MARGIN) & (rows.gain_lo > 0)]
    return float(ok.kappa.min()) if len(ok) else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def lambda_lookup(table: dict) -> dict:
    return {(rule, round(r["kappa"], 2), r["level"]): r["lambda"] for rule, rows in table.items() for r in rows}


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------

def load_lock(run_dir: Path, repo: Path) -> dict:
    lock_dir = run_dir / "lock"
    record = json.loads((lock_dir / "selection-lock.json").read_text())
    if record.get("missing_scores"):
        raise SystemExit(f"the lock was a dry run (missing scores: {record['missing_scores']})")
    if record["test_outcomes_read"]:
        raise SystemExit("the lock says it read test outcomes")
    files = {m: (v["calibrated_file"], v["calibrated_sha256"]) for m, v in record["methods"].items()}
    for basis in ("group", "representative"):
        for name, v in record["rt08"][basis]["models"].items():
            files[f"{name}-{basis}"] = (v["calibrated_file"], v["calibrated_sha256"])
    for name, (rel, digest) in files.items():
        if C.sha256_file(lock_dir / rel) != digest:
            raise SystemExit(f"calibrated file changed since 07b: {rel}")
    excluded = set()
    for name, en in record["exclusions"].items():
        if C.sha256_file(repo / en["file"]) != en["sha256"]:
            raise SystemExit(f"the evaluation exclusion list changed since 07b: {en['file']}")
        excluded |= set(pd.read_csv(repo / en["file"]).sample_id)
    return {"record": record, "files": {k: lock_dir / v[0] for k, v in files.items()},
            "sha256": C.sha256_file(lock_dir / "selection-lock.json"), "excluded": excluded}


def resolve_inputs(repo: Path, data_root: Path) -> tuple:
    cfg = json.loads((Path(__file__).resolve().parent / "config.json").read_text())
    return cfg, {k: (repo / v if (repo / v).exists() else data_root / v) for k, v in cfg["inputs"].items()}


def used_files(inputs: dict, data_root: Path, role: str) -> dict:
    """Every outcome or group file this step reads besides the run folder (their SHA-256 goes to the manifest)."""
    files = {k: inputs[k] for k in (("a_dev",) if role == "dev" else ("a_test_routerbench", "a_test_xroutebench"))
             + EXTERNAL + ("v1_data_config",)}
    v1 = json.loads(Path(inputs["v1_data_config"]).read_text())
    for key in ("labeled_path", "metadata_path", "model_tiers_path"):
        files[f"v1_routerbench_{key}"] = data_root / v1["routerbench"][key]
    x = v1["xroutebench"]
    for key in ("train_file", "test_file", "candidates_file"):
        files[f"v1_xroutebench_{key}"] = data_root / x["data_dir"] / x[key]
    missing = [str(v) for v in files.values() if not Path(v).is_file()]
    if missing:
        raise SystemExit("missing input files (upload them under the data root): " + ", ".join(missing))
    return files


def group_ids(inputs: dict, role: str) -> pd.Series:
    """Bootstrap unit per question: the A split's duplicate group, the external re-split group."""
    parts = []
    a_files = [inputs["a_dev"]] if role == "dev" else [inputs["a_test_routerbench"], inputs["a_test_xroutebench"]]
    for path in a_files:
        parts.append(pd.read_parquet(path, columns=["sample_id", "duplicate_group_id"]).rename(
            columns={"duplicate_group_id": "group_id"}))
    for source in EXTERNAL:
        parts.append(pd.read_parquet(inputs[source], columns=["sample_id", "split_group_id"]).rename(
            columns={"split_group_id": "group_id"}))
    g = pd.concat(parts, ignore_index=True).drop_duplicates("sample_id")
    return g.set_index("sample_id").group_id.astype(str)


def v1_group_probabilities(inputs: dict, repo: Path, data_root: Path, groups: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """A split group pass probabilities under v1 grading (00_prepare's own matrix builder and groups)."""
    import prepare
    v1 = prepare.accepted_matrices(inputs["v1_data_config"], data_root, repo / "ml/src")
    out = []
    for source in A_SOURCES:
        rows = frame[frame.source_dataset.eq(source)]
        succ = v1[source].reindex(rows.sample_id)
        g = pd.DataFrame(groups[source])
        cols = {}
        for t in TIERS:
            members = [m for m in g.loc[g.tier.eq(t), "model"] if m in succ.columns]
            cols[f"p_{t}"] = succ[members].mean(axis=1, skipna=True).to_numpy()
        out.append(pd.DataFrame({"sample_id": rows.sample_id.to_numpy(), **cols}))
    return pd.concat(out, ignore_index=True)


def load_truth(run_dir: Path, repo: Path, data_root: Path, lock: dict, rehearse: bool) -> dict:
    role = "dev" if rehearse else "test"
    prepare_dir = run_dir / "prepare"
    manifest = json.loads((prepare_dir / "manifest.json").read_text())
    path = prepare_dir / f"{role}.parquet"
    if C.sha256_file(path) != manifest["outputs"][path.name]["sha256"]:
        raise SystemExit(f"00_prepare output changed: {path.name}")
    cols = ["sample_id", "source_dataset", "excluded_v2", "needed_tier", "arena_overlap", "slice"] + \
        [f"p_{t}" for t in TIERS] + [f"arm_success_v2_{t}" for t in TIERS]
    frame = pd.read_parquet(path, columns=cols)
    counts = {"rows": frame.groupby("source_dataset").size().to_dict()}
    frame = frame[~frame.excluded_v2.fillna(False).astype(bool)]
    counts["after_v2_exclusions"] = frame.groupby("source_dataset").size().to_dict()
    frame = frame[~frame.sample_id.isin(lock["excluded"])]
    counts["after_evaluation_exclusions"] = frame.groupby("source_dataset").size().to_dict()
    frame = frame[frame[[f"p_{t}" for t in TIERS]].notna().all(axis=1)].reset_index(drop=True)
    counts["evaluated"] = frame.groupby("source_dataset").size().to_dict()
    cfg, inputs = resolve_inputs(repo, data_root)
    files = used_files(inputs, data_root, role)
    groups = json.loads((prepare_dir / "groups.json").read_text())
    gid = group_ids(inputs, role)
    if not frame.sample_id.isin(gid.index).all():
        raise SystemExit("questions without a bootstrap group")
    frame["group_id"] = frame.source_dataset + ":" + gid.loc[frame.sample_id].to_numpy()
    v1 = v1_group_probabilities(inputs, repo, data_root, groups, frame).set_index("sample_id")
    a = frame.source_dataset.isin(A_SOURCES)
    for t in TIERS:
        frame[f"p_v1_{t}"] = np.nan
        frame.loc[a, f"p_v1_{t}"] = v1.loc[frame.loc[a, "sample_id"], f"p_{t}"].to_numpy()
    counts["a_rows_with_v1"] = frame[a & frame[[f"p_v1_{t}" for t in TIERS]].notna().all(axis=1)] \
        .groupby("source_dataset").size().to_dict()
    frame["needed"] = frame.needed_tier.map({t: k for k, t in enumerate(TIERS)}).astype(int)
    best = frame[[f"p_{t}" for t in TIERS]].max(axis=1)
    frame["stratum"] = np.where(best >= L.TAU, "solvable", np.where(best > 0, "beyond_partial", "beyond_zero"))
    counts["strata"] = {s: g.stratum.value_counts().reindex(list(STRATA)).fillna(0).astype(int).to_dict()
                        for s, g in frame.groupby("source_dataset")}
    counts["needed_tier_share"] = {s: g.needed_tier.value_counts(normalize=True).reindex(list(TIERS)).fillna(0).round(4).to_dict()
                                   for s, g in frame.groupby("source_dataset")}
    cats = K.category_table(inputs, roles=(role,))
    frame = frame.join(cats[["category", "task_type"]], on="sample_id")
    if frame.category.isna().any():
        raise SystemExit("questions without a category")
    return {"frame": frame, "counts": counts, "role": role, "inputs": inputs,
            "test_sha256": manifest["outputs"][path.name]["sha256"],
            "input_sha256": {k: C.sha256_file(Path(v)) for k, v in files.items()}
            | {"groups.json": C.sha256_file(prepare_dir / "groups.json")}}


def calibrated(path: Path, frame: pd.DataFrame, role: str) -> pd.DataFrame:
    c = pd.read_parquet(path)
    c = c[c.role.eq(role)].drop_duplicates("sample_id").set_index("sample_id")
    return c


def phat_for(c: pd.DataFrame, ids) -> np.ndarray:
    if not pd.Index(ids).isin(c.index).all():
        raise SystemExit("calibrated probabilities missing for evaluated questions")
    P = c.loc[ids, [f"p_hat_{t}" for t in TIERS]].to_numpy(float)
    if not np.isfinite(P).all():
        raise SystemExit("non-finite calibrated probabilities on evaluated questions")
    return P


# ---------------------------------------------------------------------------------------------
# Track 1
# ---------------------------------------------------------------------------------------------

def decision_sets(Phat: np.ndarray, lam: dict, rho: float, targets: list, cost: np.ndarray) -> list:
    chain = L.predicted_chain(Phat, rho, cost)
    sets = []
    for kappa in L.KAPPAS:
        for level, target in zip(L.LEVELS, targets):
            lmb = lam[("escalation", round(float(kappa), 2), level)]
            sets.append({"rule": "escalation", "kappa": float(kappa), "level": level, "lambda": lmb, "target": target,
                         "starts": L.decide("escalation", Phat, lmb, float(kappa), chain, cost=cost)})
    for level, target in zip(L.LEVELS, targets):
        lmb = lam[("one_shot", 0.0, level)]
        sets.append({"rule": "one_shot", "kappa": None, "level": level, "lambda": lmb, "target": target,
                     "starts": L.decide("one_shot", Phat, lmb, cost=cost)})
    return sets


def oracle_lambdas(P: np.ndarray, targets: list, cost: np.ndarray) -> dict:
    """Reference only: the true probabilities with lambdas tuned on the evaluated rows themselves."""
    return lambda_lookup(L.lambda_table(P, P, 0.0, targets, cost))


def evaluate_track1(P: np.ndarray, sets: list, cost: np.ndarray, draws: Draws | None) -> tuple:
    """Records for every (set, kappa) and the gain draws of escalation sets keyed (kappa, level)."""
    n = len(P)
    cols = []
    for st in [np.full(n, s) for s in range(3)] + [s["starts"] for s in sets]:
        o = L.outcomes(P, st, cost)
        cols += [o["resolved"], o["fails"], o["cost"], o["first"]]
    X = np.column_stack(cols)
    point = X.mean(axis=0)
    boot = draws.means(X) if draws is not None else None
    fp = point[:12].reshape(3, 4)
    fb = boot[:, :12].reshape(-1, 3, 4) if boot is not None else None
    records, kept = [], {}
    for i, s in enumerate(sets):
        base = 12 + 4 * i
        r, f, c, first = point[base:base + 4]
        kappas = [s["kappa"]] if s["rule"] == "escalation" else list(map(float, L.KAPPAS))
        for kappa in kappas:
            value = r - kappa * f
            env = float(envelope_batch(fp[None, :, 2], fp[None, :, 0] - kappa * fp[None, :, 1], np.array([c]))[0])
            rec = {"rule": s["rule"], "kappa": kappa, "level": s["level"], "lambda": s["lambda"], "target_cost": s["target"],
                   "cost": c, "resolved": r, "fails": f, "first": first, "calls": r + f, "value": value,
                   "envelope": env, "gain": value - env, "gain_lo": np.nan, "gain_hi": np.nan}
            if boot is not None:
                rb, fb_, cb = boot[:, base], boot[:, base + 1], boot[:, base + 2]
                g = (rb - kappa * fb_) - envelope_batch(fb[:, :, 2], fb[:, :, 0] - kappa * fb[:, :, 1], cb)
                rec["gain_lo"], rec["gain_hi"] = (float(v) for v in interval(g))
                if s["rule"] == "escalation":
                    kept[(round(kappa, 2), s["level"])] = g.astype(np.float32)
            records.append(rec)
    fixed = [{"start": TIERS[s], "resolved": fp[s, 0], "fails": fp[s, 1], "cost": fp[s, 2], "first": fp[s, 3]}
             for s in range(3)]
    return records, kept, fixed


def summarize_draws(kept: dict) -> dict:
    """Mean gain over the kappa grid and levels (the 07b selection metric) and the level-.5 gains per kappa."""
    keys = sorted(kept)
    mg = np.mean([kept[k] for k in keys], axis=0)
    half = {k[0]: kept[k] for k in keys if k[1] == 0.5}
    return {"mean_gain": mg, "half": half}


# ---------------------------------------------------------------------------------------------
# Track 2
# ---------------------------------------------------------------------------------------------

def evaluate_track2(P: np.ndarray, needed: np.ndarray, starts: np.ndarray, cost: np.ndarray) -> np.ndarray:
    o = L.outcomes(P, starts, cost)
    return np.column_stack([(starts == needed).astype(float), o["first"], o["resolved"], o["resolved"] + o["fails"],
                            o["cost"]])


TRACK2 = ("needed_tier_accuracy", "first", "resolved", "calls", "cost")


def solvable_accuracy(M: np.ndarray, solvable: np.ndarray) -> float:
    """Needed-tier accuracy over the solvable stratum only: a question no group solves has no needed tier."""
    return float(M[solvable, 0].mean()) if solvable.any() else float("nan")


# ---------------------------------------------------------------------------------------------

def run(run_dir: Path, repo: Path, data_root: Path, output_dir: Path, rehearse: bool, iterations: int) -> dict:
    if output_dir.exists():
        raise SystemExit(f"{output_dir} exists: the test is opened once (move it away only to redo a failed run)")
    t0 = time.time()
    lock = load_lock(run_dir, repo)
    rec = lock["record"]
    ev = rec["evaluation"]
    truth = load_truth(run_dir, repo, data_root, lock, rehearse)
    frame, role = truth["frame"], truth["role"]
    print(f"[08] {role} rows evaluated: {truth['counts']['evaluated']}", flush=True)
    methods = sorted(rec["methods"])
    cal = {m: calibrated(lock["files"][m], frame, role) for m in methods}
    cal["r0_operational"] = calibrated(lock["files"]["r0_operational-group"], frame, role)
    group_price, length_cfg = ev["sensitivity"]["group_price"], ev["sensitivity"]["length"]
    length, length_info = L.length_table(run_dir)
    if length_info["sha256"] != length_cfg["lengths_sha256"]:
        raise SystemExit("length-inputs changed since 07b")
    output_dir.mkdir(parents=True)

    track1, fixed_rows, track2, overlap, starts_cols, summaries = [], [], [], [], {}, {}
    for si, source in enumerate(SOURCES):
        rows = frame[frame.source_dataset.eq(source)].reset_index(drop=True)
        ids = rows.sample_id.to_numpy()
        P = rows[[f"p_{t}" for t in TIERS]].to_numpy(float)
        draws = Draws(rows.group_id, iterations, BOOT["seed"], si)
        names = methods + ["oracle"] + (["r0_operational"] if source in A_SOURCES else [])
        bases = {"main": (L.SERVICE_COST, ev["cost_levels"]["targets"][source], P, draws),
                 "group_price": (np.array(group_price["tier_cost"][source]), group_price["targets"][source], P, draws),
                 "length": (L.length_costs(length, ids, length_cfg["normalizer_mean_dev_length"][source]),
                            length_cfg["targets"][source], P, draws)}
        if source in A_SOURCES:
            Pv1 = rows[[f"p_v1_{t}" for t in TIERS]].to_numpy(float)
            ok = np.isfinite(Pv1).all(axis=1)
            bases["v1"] = (L.SERVICE_COST, ev["cost_levels"]["targets"][source], Pv1, None)
        for name in names:
            if name == "oracle":
                Phat = P
            else:
                Phat = phat_for(cal[name], ids)
            for basis, (cost, targets, truth_P, d) in bases.items():
                if name == "oracle":  # decisions on every row; v1 rows without outcomes are dropped below
                    lam, rho = oracle_lambdas(truth_P if basis != "v1" else truth_P[ok], targets, cost), 0.0
                    Phat_b = np.nan_to_num(truth_P)
                elif name == "r0_operational":
                    if basis in ("group_price", "length"):
                        continue
                    lam, rho, Phat_b = lambda_lookup(rec["rt08"]["group"]["models"]["r0_operational"]["lambda"][source]), \
                        rec["rt08"]["group"]["rho"][source], Phat
                else:
                    table = {"group_price": rec["methods"][name]["sensitivity_group_price"]["lambda"],
                             "length": rec["methods"][name]["sensitivity_length"]["lambda"]}.get(basis, rec["methods"][name]["lambda"])
                    lam, rho, Phat_b = lambda_lookup(table[source]), ev["rho"][source], Phat
                sets = decision_sets(Phat_b, lam, rho, targets, cost)
                use = slice(None) if basis != "v1" else ok
                for s in sets:
                    s["starts_all"] = s["starts"]
                    s["starts"] = s["starts"][use]
                records, kept, fixed = evaluate_track1(truth_P[use], sets, cost, d if basis != "v1" else None)
                for r in records:
                    track1.append({"basis": basis, "source": source, "method": name, **r})
                if name == methods[0]:
                    fixed_rows += [{"basis": basis, "source": source, **f} for f in fixed]
                if kept:
                    summaries[(basis, source, name)] = summarize_draws(kept)
                if basis == "main":
                    for s in sets:
                        tag = f"k{s['kappa']:.2f}" if s["kappa"] is not None else "oneshot"
                        starts_cols.setdefault(name, {})[(source, s["rule"], tag, s["level"])] = s["starts_all"]
            # track 2 on the main basis
            needed = rows.needed.to_numpy()
            in_arena = rows.arena_overlap.eq(True).to_numpy()
            solvable = rows.stratum.eq("solvable").to_numpy()
            for tau in L.TAU_CURVE:
                st = L.decide("tau", Phat, 0.0, tau=float(tau))
                M = evaluate_track2(P, needed, st, L.SERVICE_COST)
                rec2 = {"source": source, "method": name, "tau": float(tau),
                        **{k: float(v) for k, v in zip(TRACK2, M.mean(axis=0))},
                        "needed_tier_accuracy_solvable": solvable_accuracy(M, solvable)}
                if abs(tau - L.TAU) < 1e-9:
                    starts_cols.setdefault(name, {})[(source, "tau", "t0.80", None)] = st
                    if in_arena.any():
                        overlap.append({"source": source, "method": name, "rows": int(in_arena.sum()),
                                        **{k: float(v) for k, v in zip(TRACK2, M[in_arena].mean(axis=0))}})
                    if name not in REFERENCE:
                        B = draws.means(np.column_stack([M[:, 0] * solvable, M[:, 1], solvable]))
                        B = np.column_stack([B[:, 0] / B[:, 2], B[:, 1]])  # solvable-only accuracy, first success
                        for j, k in enumerate(("needed_tier_accuracy_solvable", "first")):
                            rec2[f"{k}_lo"], rec2[f"{k}_hi"] = (float(v) for v in interval(B[:, j]))
                        summaries[("track2", source, name)] = B
                track2.append(rec2)
            print(f"[08] {source} {name} done ({time.time() - t0:.0f}s)", flush=True)
        # track 2 baselines: fixed starts and the needed tier itself
        needed = rows.needed.to_numpy()
        for label, st in [(f"fixed_{t}", np.full(len(rows), k)) for k, t in enumerate(TIERS)] + [("needed_tier", needed)]:
            M = evaluate_track2(P, needed, st, L.SERVICE_COST)
            track2.append({"source": source, "method": label, "tau": np.nan,
                           **{k: float(v) for k, v in zip(TRACK2, M.mean(axis=0))},
                           "needed_tier_accuracy_solvable": solvable_accuracy(M, solvable)})
            if in_arena.any():
                overlap.append({"source": source, "method": label, "rows": int(in_arena.sum()),
                                **{k: float(v) for k, v in zip(TRACK2, M[in_arena].mean(axis=0))}})

    t1 = pd.DataFrame(track1)
    t1.to_parquet(output_dir / "track1.parquet", index=False)
    pd.DataFrame(fixed_rows).to_parquet(output_dir / "fixed-starts.parquet", index=False)
    t2 = pd.DataFrame(track2)
    t2.to_parquet(output_dir / "track2.parquet", index=False)
    pd.DataFrame(overlap, columns=["source", "method", "rows", *TRACK2]).to_parquet(
        output_dir / "track2-arena-overlap.parquet", index=False)
    ks = (t1.groupby(["basis", "source", "method", "rule", "level"]).apply(kappa_star, include_groups=False)
          .rename("kappa_star").reset_index())
    ks.to_parquet(output_dir / "kappa-star.parquet", index=False)

    mean_gain = mean_gain_table(summaries, t1)
    mean_gain.to_parquet(output_dir / "mean-gain.parquet", index=False)
    rt08 = evaluate_rt08(rec, lock, frame, role, iterations)
    rt08["table"].to_parquet(output_dir / "rt08.parquet", index=False)
    diagnostics = diagnose(run_dir, rec, frame, role, iterations, t2)
    hypotheses = judge(summaries, t1, t2, ks, mean_gain, rt08, diagnostics, methods)
    write_starts(starts_cols, frame, output_dir / "test-starts.parquet")
    by_cat = by_category(frame, starts_cols)
    by_cat.to_parquet(output_dir / "by-category.parquet", index=False)
    by_str = by_category(frame, starts_cols, minimum=1, groupings=(("stratum", "stratum"),))
    by_str.to_parquet(output_dir / "by-stratum.parquet", index=False)

    outputs = {p.name: C.sha256_file(p) for p in sorted(output_dir.glob("*.parquet"))}
    (output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=1, default=float))
    (output_dir / "hypotheses.json").write_text(json.dumps(hypotheses, ensure_ascii=False, indent=1, default=float))
    manifest = {"experiment": "EV-10 08_test", "rehearsal_on_dev": rehearse, "test_outcomes_read": not rehearse,
                "role_evaluated": role, "lock_sha256": lock["sha256"], "outcomes_sha256": truth["test_sha256"],
                "input_sha256": truth["input_sha256"],
                "counts": truth["counts"], "bootstrap": BOOT | {"iterations": iterations}, "margin": MARGIN,
                "methods": methods, "references": list(REFERENCE), "outputs": outputs,
                "seconds": round(time.time() - t0, 1)}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=float))
    pooled = by_cat[by_cat.source.eq("all") & ~by_cat.method.astype(str).str.startswith("fixed_")].pivot_table(
        index="method", columns="group", values="mean_gain")
    pooled = pooled[[t for t in K.TYPES if t in pooled.columns]]
    table = ["| 방법 | " + " | ".join(pooled.columns) + " |", "|---|" + "---:|" * len(pooled.columns)]
    table += [f"| {m} | " + " | ".join("–" if np.isnan(v) else f"{v:+.3f}" for v in r) + " |" for m, r in pooled.iterrows()]
    ps = by_str[by_str.source.eq("all") & ~by_str.method.astype(str).str.startswith("fixed_")].pivot_table(
        index="method", columns="group", values="mean_gain")
    ps = ps[[g for g in STRATA if g in ps.columns]]
    stable = ["| 방법 | " + " | ".join(ps.columns) + " |", "|---|" + "---:|" * len(ps.columns)]
    stable += [f"| {m} | " + " | ".join("–" if np.isnan(v) else f"{v:+.3f}" for v in r) + " |" for m, r in ps.iterrows()]
    (output_dir / "test-summary.md").write_text(summary_markdown(
        manifest | {"by_category_table": table, "by_stratum_table": stable}, mean_gain, ks, hypotheses, t2))
    return manifest


def mean_gain_table(summaries: dict, t1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for basis in ("main", "group_price", "length"):
        by_method = {}
        for (b, source, name), s in summaries.items():
            if b != basis:
                continue
            point = t1[(t1.basis.eq(basis)) & t1.source.eq(source) & t1.method.eq(name) & t1.rule.eq("escalation")].gain.mean()
            lo, hi = interval(s["mean_gain"])
            rows.append({"basis": basis, "source": source, "method": name, "mean_gain": point, "lo": lo, "hi": hi})
            by_method.setdefault(name, {})[source] = (point, s["mean_gain"])
        for name, per in by_method.items():
            if len(per) == len(SOURCES):
                point = float(np.mean([p for p, _ in per.values()]))
                lo, hi = interval(np.mean([d for _, d in per.values()], axis=0))
                rows.append({"basis": basis, "source": "macro", "method": name, "mean_gain": point, "lo": lo, "hi": hi})
    v1 = t1[t1.basis.eq("v1") & t1.rule.eq("escalation")].groupby(["source", "method"]).gain.mean().reset_index()
    rows += [{"basis": "v1", "source": r.source, "method": r.method, "mean_gain": r.gain, "lo": np.nan, "hi": np.nan}
             for r in v1.itertuples()]
    out = pd.DataFrame(rows)
    ranked = ~out.method.isin(REFERENCE)
    out["rank"] = np.nan
    out.loc[ranked, "rank"] = out[ranked].groupby(["basis", "source"]).mean_gain.rank(ascending=False, method="min")
    return out


def evaluate_rt08(rec: dict, lock: dict, frame: pd.DataFrame, role: str, iterations: int) -> dict:
    """RT-08 (plan 3): RT-08 (v2 retrained) against the operational R0 (v1) on the A split, both bases."""
    rows_out, draws_out = [], {}
    for basis, truth_cols in (("group", [f"p_{t}" for t in TIERS]), ("representative", [f"arm_success_v2_{t}" for t in TIERS])):
        info = rec["rt08"][basis]
        for si, source in enumerate(A_SOURCES):
            rows = frame[frame.source_dataset.eq(source)]
            rows = rows[rows[truth_cols].notna().all(axis=1)].reset_index(drop=True)
            P = rows[truth_cols].to_numpy(float)
            draws = Draws(rows.group_id, iterations, BOOT["seed"], 100 + 10 * si + (basis == "representative"))
            for name in ("rt08", "r0_operational"):
                c = calibrated(lock["files"][f"{name}-{basis}"], frame, role)
                Phat = phat_for(c, rows.sample_id)
                lam = lambda_lookup(info["models"][name]["lambda"][source])
                sets = decision_sets(Phat, lam, info["rho"][source], info["cost_targets"][source], L.SERVICE_COST)
                records, kept, _ = evaluate_track1(P, sets, L.SERVICE_COST, draws)
                rows_out += [{"basis": basis, "source": source, "model": name, **r} for r in records]
                draws_out[(basis, source, name)] = summarize_draws(kept)
    table = pd.DataFrame(rows_out)
    verdicts = []
    for basis in ("group", "representative"):
        for source in A_SOURCES:
            a, b = draws_out[(basis, source, "rt08")], draws_out[(basis, source, "r0_operational")]
            per_kappa = []
            for kappa in sorted(a["half"]):
                d = a["half"][kappa] - b["half"][kappa]
                pa = table[(table.basis.eq(basis)) & table.source.eq(source) & table.rule.eq("escalation")
                           & table.level.eq(0.5) & np.isclose(table.kappa, kappa)]
                diff = float(pa[pa.model.eq("rt08")].gain.iloc[0] - pa[pa.model.eq("r0_operational")].gain.iloc[0])
                lo, hi = interval(d)
                per_kappa.append({"kappa": kappa, "diff": diff, "lo": float(lo), "hi": float(hi)})
            better = all(r["diff"] > MARGIN and r["lo"] > 0 for r in per_kappa)
            any_diff = any(abs(r["diff"]) > MARGIN and (r["lo"] > 0 or r["hi"] < 0) for r in per_kappa)
            verdicts.append({"basis": basis, "source": source, "per_kappa_level_0.5": per_kappa,
                             "verdict": "v1_errors_harmed" if better else ("no_material_effect" if not any_diff else "mixed")})
    return {"table": table, "verdicts": verdicts}


def diagnose(run_dir: Path, rec: dict, frame: pd.DataFrame, role: str, iterations: int, t2: pd.DataFrame) -> dict:
    """H3 (IRT difficulty vs our labels and RouterArena's Bloom labels) and the RouterArena overlap rows."""
    out = {}
    irt_rel = rec["methods"]["irt_router"]["scores_file"]
    irt = pd.read_parquet(run_dir / irt_rel, columns=["sample_id", "role", "irt_difficulty"]).drop_duplicates("sample_id") \
        .set_index("sample_id").irt_difficulty
    h3 = {}
    for si, source in enumerate(SOURCES):
        rows = frame[frame.source_dataset.eq(source)].reset_index(drop=True)
        x, y = irt.loc[rows.sample_id].to_numpy(float), rows.needed.to_numpy(float)
        draws = Draws(rows.group_id, iterations, BOOT["seed"], 200 + si)
        boot = []
        for _, W in draws.counts():
            for w in W.astype(int):
                idx = np.repeat(np.arange(len(rows)), w)
                boot.append(spearman(x[idx], y[idx]))
        lo, hi = interval(np.array(boot))
        h3[source] = {"spearman": spearman(x, y), "lo": float(lo), "hi": float(hi), "n": int(len(rows))}
    arena = pd.read_parquet(run_dir / "inputs" / "routerarena-full.parquet", columns=["Global Index", "Difficulty", "Context"])
    arena["sample_id"] = "routerarena:" + arena["Global Index"].astype(str)
    # routers saw the Question only; a question that comes with its own Context (passage, premise, code) is incomplete
    with_context = arena.Context.fillna("").astype(str).str.strip().ne("")
    out["routerarena_rows"] = {"all": int(len(arena)), "with_context_dropped": int(with_context.sum()),
                               "used": int((~with_context).sum())}
    arena = arena[~with_context]
    arena = arena[arena.sample_id.isin(irt.index)]
    arena = arena[np.isfinite(irt.loc[arena.sample_id].to_numpy(float))].reset_index(drop=True)
    x, y = irt.loc[arena.sample_id].to_numpy(float), arena.Difficulty.map(BLOOM).to_numpy(float)
    rng = np.random.default_rng(np.random.SeedSequence([BOOT["seed"], 300]))
    boot = [spearman(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(min(iterations, 2000)))]
    lo, hi = interval(np.array(boot))
    h3["routerarena_bloom"] = {"spearman": spearman(x, y), "lo": float(lo), "hi": float(hi), "n": int(len(x)),
                               "bootstrap": "questions, 2000 draws (no groups in RouterArena)"}
    out["h3_irt_difficulty"] = h3
    # R0-group's predicted class against the Bloom label (plan 2.4 RouterArena diagnostic)
    r0 = pd.read_parquet(run_dir / rec["methods"]["r0_group"]["scores_file"]).drop_duplicates("sample_id").set_index("sample_id")
    s = r0.loc[arena.sample_id, [f"score_{t}" for t in TIERS]].to_numpy(float)
    prob = L.softmax(s)
    out["routerarena_r0_group"] = {
        "spearman_expected_class_vs_bloom": spearman(prob @ np.arange(3), y),
        "agreement_argmax_vs_bloom": float((prob.argmax(axis=1) == y).mean()),
        "confusion": pd.crosstab(arena.Difficulty.to_numpy(), np.array(TIERS)[prob.argmax(axis=1)]).to_dict()}
    # RouterArena-overlap questions inside the external test (plan 2.4): track 2 at tau = .8 and fixed starts
    ov = frame[frame.arena_overlap.eq(True)]
    out["routerarena_overlap"] = {"rows": ov.groupby("source_dataset").size().to_dict(), "note":
                                  "rows of the LLMRouterBench/SPROUT test that also appear in RouterArena; "
                                  "see track2-arena-overlap.parquet (tau = .8 and fixed starts)"}
    return out


def judge(summaries: dict, t1, t2, ks, mean_gain, rt08, diagnostics, methods) -> dict:
    """Pre-registered checks of plan 4.4 (H1-H7) and plan 3 (RT-08). Descriptive numbers plus the stated rule."""
    out = {}
    main = mean_gain[mean_gain.basis.eq("main")]
    # H1: differences to the top router (macro and per source), kNN rank
    h1 = {}
    for source in list(SOURCES) + ["macro"]:
        m = main[main.source.eq(source) & ~main.method.isin(REFERENCE)].sort_values("mean_gain", ascending=False)
        top = m.method.iloc[0]

        def draw(name):
            if source == "macro":
                return np.mean([summaries[("main", s, name)]["mean_gain"] for s in SOURCES], axis=0)
            return summaries[("main", source, name)]["mean_gain"]
        rows = []
        for name in m.method:
            d = draw(top) - draw(name)
            lo, hi = interval(d)
            rows.append({"method": name, "mean_gain": float(m.set_index("method").mean_gain[name]),
                         "diff_to_top": float(m.mean_gain.iloc[0] - m.set_index("method").mean_gain[name]),
                         "lo": float(lo), "hi": float(hi), "includes_zero": bool(lo <= 0 <= hi)})
        h1[source] = {"top": top, "knn_rank": int(m.method.tolist().index("knn") + 1), "routers": rows,
                      "share_not_distinguishable_from_top": float(np.mean([r["includes_zero"] for r in rows[1:]]))}
    out["H1"] = {"rule": "difference to the top router's mean gain: does its 95% interval include 0; kNN rank", "result": h1}
    # H2: track 2, R0-group vs R1-group within +-.02
    h2 = {}
    for source in SOURCES:
        a, b = summaries[("track2", source, "r0_group")], summaries[("track2", source, "r1_group")]
        pa = t2[t2.source.eq(source) & np.isclose(t2.tau, 0.8)].set_index("method")
        res = {}
        for j, k in enumerate(("needed_tier_accuracy_solvable", "first")):
            d = float(pa.loc["r0_group", k] - pa.loc["r1_group", k])
            lo, hi = interval(a[:, j] - b[:, j])
            res[k] = {"diff_r0_minus_r1": d, "lo": float(lo), "hi": float(hi), "within_0.02": bool(abs(d) <= 0.02)}
        h2[source] = res
    out["H2"] = {"rule": "tau = .8: R0-group minus R1-group needed-tier accuracy (solvable stratum; a question no group "
                         "solves has no needed tier, 10-18) and first-attempt success (all rows) within +-.02",
                 "result": h2}
    out["H3"] = {"rule": "Spearman of IRT-Router difficulty with our needed tier (positive) and with RouterArena's Bloom "
                         "label (weak)", "result": diagnostics["h3_irt_difficulty"]}
    # H4: retrained vs released RouteLLM at level .5 over every kappa
    out["H4"] = {"rule": "cost level .5: retrained minus released gain > .01 with the interval above 0 at every kappa",
                 "result": pair_over_kappa(summaries, t1, "routellm_retrained", "routellm_released")}
    # H5: mixed vs A-only training
    h5 = {}
    for mixed, a_only in (("r1_group", "r1_group_a"), ("knn", "knn_a")):
        per = {}
        for source in SOURCES:
            d = summaries[("main", source, mixed)]["mean_gain"] - summaries[("main", source, a_only)]["mean_gain"]
            point = float(main[main.source.eq(source)].set_index("method").mean_gain.loc[mixed]
                          - main[main.source.eq(source)].set_index("method").mean_gain.loc[a_only])
            lo, hi = interval(d)
            per[source] = {"diff": point, "lo": float(lo), "hi": float(hi)}
        ext_ok = all(per[s]["diff"] > MARGIN for s in EXTERNAL)
        a_ok = all(per[s]["diff"] >= -MARGIN for s in A_SOURCES)
        h5[f"{mixed}_vs_{a_only}"] = {"per_source": per, "external_better": ext_ok, "a_not_worse": a_ok,
                                      "supported": ext_ok and a_ok}
    out["H5"] = {"rule": "mean gain: mixed minus A-only > .01 on LLMRouterBench and SPROUT, >= -.01 on the A split",
                 "result": h5}
    # H6: ranking stability under v1 vs v2 on the A test
    from scipy.stats import kendalltau
    h6 = {}
    for source in A_SOURCES:
        v2 = mean_gain[mean_gain.basis.eq("main") & mean_gain.source.eq(source) & ~mean_gain.method.isin(REFERENCE)]
        v1 = mean_gain[mean_gain.basis.eq("v1") & mean_gain.source.eq(source) & ~mean_gain.method.isin(REFERENCE)]
        both = v2.set_index("method").mean_gain.to_frame("v2").join(v1.set_index("method").mean_gain.rename("v1"), how="inner")
        tau = float(kendalltau(both.v2, both.v1).statistic)
        h6[source] = {"kendall_tau": tau, "routers": int(len(both)), "supported": bool(tau >= 0.7)}
    out["H6"] = {"rule": "Kendall tau of the A-test router ranking (mean gain) under v1 and v2 grading >= .7", "result": h6}
    k = ks[ks.basis.eq("main") & ks.rule.eq("escalation") & ~ks.method.isin(REFERENCE)]
    out["H7"] = {"rule": "kappa* per router (escalation-aware rule) at cost levels .25/.5/.75; most routers should not win "
                         "at small kappa under 1:3:5",
                 "result": {f"{r.source}/{r.method}/{r.level}": r.kappa_star for r in k.itertuples()},
                 "share_without_kappa_star": float(k.kappa_star.isna().mean())}
    out["RT-08"] = {"rule": "cost level .5: v2-retrained minus operational R0 gain > .01 with interval above 0 at every kappa "
                            "-> v1 errors harmed; no kappa with |diff| > .01 and interval excluding 0 -> no material effect",
                    "result": rt08["verdicts"]}
    return out


def pair_over_kappa(summaries: dict, t1: pd.DataFrame, a: str, b: str) -> dict:
    res = {}
    for source in SOURCES:
        da, db = summaries[("main", source, a)]["half"], summaries[("main", source, b)]["half"]
        sel = t1[t1.basis.eq("main") & t1.source.eq(source) & t1.rule.eq("escalation") & t1.level.eq(0.5)]
        per = []
        for kappa in sorted(da):
            row = sel[np.isclose(sel.kappa, kappa)].set_index("method").gain
            lo, hi = interval(da[kappa] - db[kappa])
            per.append({"kappa": kappa, "diff": float(row[a] - row[b]), "lo": float(lo), "hi": float(hi)})
        res[source] = {"per_kappa": per, "supported": all(r["diff"] > MARGIN and r["lo"] > 0 for r in per)}
    return res


def by_category(frame: pd.DataFrame, starts: dict, minimum: int = 30,
                groupings=(("category", "category"), ("task_type", "task_type"))) -> pd.DataFrame:
    """Descriptive results per source category and shared task type (main basis, point estimates, not judged).

    Same locked decisions as the whole-source results; the fixed-start envelope is rebuilt inside each category.
    mean_gain: escalation-aware rule over the kappa grid and levels. tau .8: track-2 metrics and start shares.
    Groups with fewer than `minimum` questions keep their counts but no gains."""
    rows = []
    for source in SOURCES:
        part = frame[frame.source_dataset.eq(source)].reset_index(drop=True)
        P = part[[f"p_{t}" for t in TIERS]].to_numpy(float)
        needed = part.needed.to_numpy()
        for level_name, column in groupings:
            for group, idx in part.groupby(column).indices.items():
                n = len(idx)
                base = {"grouping": level_name, "source": source, "group": group, "rows": n,
                        "task_type": part.task_type.iloc[idx[0]]}
                if n < minimum:
                    rows.append(base | {"method": None})
                    continue
                Pg = P[idx]
                fixed = [L.outcomes(Pg, np.full(n, k)) for k in range(3)]
                fc = np.array([f["cost"].mean() for f in fixed])
                for name, by in starts.items():
                    esc = [(k, st) for k, st in by.items() if k[0] == source and k[1] == "escalation"]
                    if not esc:
                        continue
                    gains = []
                    for (src, rule, tag, level), st in esc:
                        kappa = float(tag[1:])
                        o = L.outcomes(Pg, st[idx])
                        fv = np.array([f["resolved"].mean() - kappa * f["fails"].mean() for f in fixed])
                        env = envelope_batch(fc[None], fv[None], np.array([o["cost"].mean()]))[0]
                        gains.append(o["resolved"].mean() - kappa * o["fails"].mean() - env)
                    tau = by[(source, "tau", "t0.80", None)][idx]
                    M = evaluate_track2(Pg, needed[idx], tau, L.SERVICE_COST).mean(axis=0)
                    share = np.bincount(tau, minlength=3) / n
                    solv = part.stratum.iloc[idx].eq("solvable").to_numpy()
                    rows.append(base | {"method": name, "mean_gain": float(np.mean(gains)),
                                        **{f"tau_{k}": float(v) for k, v in zip(TRACK2, M)},
                                        "tau_needed_tier_accuracy_solvable": solvable_accuracy(
                                            evaluate_track2(Pg, needed[idx], tau, L.SERVICE_COST), solv),
                                        **{f"tau_start_{t}": float(v) for t, v in zip(TIERS, share)}})
                for k, t in enumerate(TIERS):  # fixed-start reference rows
                    M = evaluate_track2(Pg, needed[idx], np.full(n, k), L.SERVICE_COST).mean(axis=0)
                    rows.append(base | {"method": f"fixed_{t}", **{f"tau_{key}": float(v) for key, v in zip(TRACK2, M)}})
    out = pd.DataFrame(rows)
    # pooled over sources (task type, stratum): row-weighted mean of the per-source values
    pool_by = [g for g, _ in groupings if g != "category"]
    if not pool_by:
        return out
    typed = out[out.grouping.isin(pool_by) & out.method.notna()]
    num = [c for c in out.columns if c.startswith(("mean_gain", "tau_"))]
    pooled = (typed.assign(**{c: typed[c] * typed.rows for c in num})
              .groupby(["grouping", "group", "method"])[num + ["rows"]].sum())
    pooled[num] = pooled[num].div(pooled.rows, axis=0)
    pooled = pooled.reset_index().assign(source="all", task_type=lambda d: d.group.where(d.grouping.eq("task_type")))
    return pd.concat([out, pooled], ignore_index=True)


def write_starts(cols: dict, frame: pd.DataFrame, path: Path) -> None:
    """Every router's start tier per evaluated question (main basis): escalation and one-shot per kappa/level, tau .8."""
    parts = []
    for source in SOURCES:
        rows = frame[frame.source_dataset.eq(source)]
        table = {"sample_id": rows.sample_id.to_numpy(), "source_dataset": source}
        for name, by in cols.items():
            for (src, rule, tag, level), st in by.items():
                if src == source:
                    table[f"{name}|{rule}|{tag}|{level}"] = st.astype(np.int8)
        parts.append(pd.DataFrame(table))
    out = pd.concat(parts, ignore_index=True)
    cols = [c for c in out.columns if "|" in c]
    out[cols] = out[cols].astype(float).fillna(-1).astype(np.int8)  # -1: router not evaluated on that source (operational R0)
    out.to_parquet(path, index=False)


def summary_markdown(manifest: dict, mean_gain: pd.DataFrame, ks: pd.DataFrame, hyp: dict, t2: pd.DataFrame) -> str:
    head = "08 test 요약 (dev 리허설 — test가 아니다)" if manifest["rehearsal_on_dev"] else "08 test 요약"
    share = manifest["counts"]["needed_tier_share"]
    lines = [f"# {head}", "", f"평가 행: {manifest['counts']['evaluated']}", "",
             "필요 단계 비율(low / medium / high): " + "; ".join(
                 f"{s} " + " / ".join(f"{share[s][t]:.1%}" for t in TIERS) for s in SOURCES if s in share), "",
             "## 평균 이득 (갈래 ①, 상향 고려 규칙, κ 격자·비용 수준 평균, 95% 구간)", "",
             "| 방법 | " + " | ".join(list(SOURCES) + ["macro"]) + " | 집단 가격 macro | 길이 반영 macro |",
             "|---|" + "---:|" * (len(SOURCES) + 3)]
    main = mean_gain[mean_gain.basis.eq("main")].set_index(["method", "source"])
    gp = mean_gain[mean_gain.basis.eq("group_price")].set_index(["method", "source"])
    ln = mean_gain[mean_gain.basis.eq("length")].set_index(["method", "source"])
    macro = mean_gain[mean_gain.basis.eq("main") & mean_gain.source.eq("macro")].sort_values("mean_gain", ascending=False)
    order = list(macro.method) + [m for m in main.index.get_level_values(0).unique() if m not in set(macro.method)]
    for m in order:
        cells = []
        for s in list(SOURCES) + ["macro"]:
            if (m, s) in main.index:
                r = main.loc[(m, s)]
                cells.append(f"{r.mean_gain:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]")
            else:
                cells.append("–")
        g = f"{gp.loc[(m, 'macro')].mean_gain:+.3f}" if (m, "macro") in gp.index else "–"
        n = f"{ln.loc[(m, 'macro')].mean_gain:+.3f}" if (m, "macro") in ln.index else "–"
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {g} | {n} |")
    strata = manifest["counts"]["strata"]
    lines += ["", "## 풀리는 정도별 층 (10-18 사전 고정, 기술 통계)", "",
              "solvable = 어느 집단이든 풀 확률 ≥ .8, beyond_partial = 모두 .8 미만이지만 0은 아님, beyond_zero = 모두 0.", "",
              "| 원천 | " + " | ".join(STRATA) + " |", "|---|" + "---:|" * len(STRATA)]
    lines += [f"| {s} | " + " | ".join(f"{strata[s][g]:,}" for g in STRATA) + " |" for s in SOURCES if s in strata]
    lines += ["", "층별 평균 이득(층 안에서 고정 시작 기준선을 다시 만든다; 원천을 합칠 때 문항 수 가중):", ""]
    lines += manifest.get("by_stratum_table", [])
    lines += ["", "## 과제 유형별 평균 이득 (기술 통계, 판정 아님; 원천을 합칠 때 문항 수 가중)", ""]
    bc = manifest.get("by_category_table")
    if bc is not None:
        lines += bc
    lines += ["", "## 가설", ""]
    for key in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "RT-08"):
        lines.append(f"- **{key}**: {hyp[key]['rule']}")
    lines += ["", "자세한 값: `hypotheses.json`, `track1.parquet`, `track2.parquet`, `kappa-star.parquet`, `rt08.parquet`."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True, help="checkout with ml/src, committed inputs and the exclusion list")
    p.add_argument("--data-root", type=Path, default=None, help="git-ignored raw data (defaults to --repo)")
    p.add_argument("--output-dir", type=Path, required=True, help="normally <run>/test")
    p.add_argument("--rehearse-on-dev", action="store_true", help="evaluate dev outcomes instead (pipeline check)")
    p.add_argument("--iterations", type=int, default=BOOT["iterations"])
    a = p.parse_args(argv)
    if not a.rehearse_on_dev and a.iterations != BOOT["iterations"]:
        raise SystemExit("the test run uses the pre-set bootstrap size")
    out = run(a.run_dir, a.repo, a.data_root or a.repo, a.output_dir, a.rehearse_on_dev, a.iterations)
    print(json.dumps({k: out[k] for k in ("role_evaluated", "counts", "seconds")}, indent=1, default=float))


if __name__ == "__main__":
    main()
