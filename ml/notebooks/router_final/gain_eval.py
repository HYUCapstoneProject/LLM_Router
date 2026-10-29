"""EV-10 11_gain: re-evaluate the locked routers with the first-attempt gain G instead of the service utility U.

    G = A - H(C)

A is the router's expected first-attempt success (the group pass probability of the tier it starts at), C its mean
first-attempt cost (service prices .2/.6/1.0) and H(C) the best first-attempt success that fixed start tiers and
their mixtures reach at a cost of at most C (07b's envelope with kappa = 0, first attempt only). No exchange rate
lambda and no failure weight kappa enter G. Escalation is not part of G.

Nothing is retrained. Every score the 04-07a steps saved is reused:

  lock  (dev only)  every setting of every method is calibrated exactly as in 07b (same adapters, same 5-fold
                    cross-fitting); one setting per method is selected by the dev G of the tau rule (tau = .8),
                    averaged over the four sources. For the one-shot rule argmax(p_hat - lambda x price), lambda is
                    picked per source and cost level so that the dev first-attempt cost is closest to 25/50/75% of
                    the fixed first-attempt cost range (.4/.6/.8 with 1:3:5). Writes <run>/gain-lock.
  test              opens the test once more with the gain lock. Paired group bootstrap as in 08 (same draws).
                    Writes <run>/gain-test (must not exist). `--rehearse-on-dev` writes elsewhere without test.

The operational R0 and RT-08 (A split, references, never ranked) reuse the 07b calibration, which does not depend
on the metric. R1-BGE is calibrated and locked like every method but left out of the test outputs (OMITTED).
Seeds 17 and 73 (09) reuse the 09 calibration (tau rule only).
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
import common_track as C  # noqa: E402
import lock as L  # noqa: E402
import test_eval as T  # noqa: E402

TIERS, SOURCES, A_SOURCES = L.TIERS, L.SOURCES, L.A_SOURCES
EXTERNAL = T.EXTERNAL
TAU = L.TAU
LEVELS = L.LEVELS  # fractions of each source's fixed first-attempt cost range
REFERENCES = ("needed_tier", "r0_operational", "rt08")  # reported, never ranked
OMITTED = ("r1_bge",)  # user decision 2026-10-21: kept in the lock, left out of every result table and figure
SEEDS = (17, 73)
PAIR = ("r1_group", "r1_group_a")  # H5: mixed data, A split only


# ---------------------------------------------------------------------------------------------
# Metric (pure numpy; unit-tested)
# ---------------------------------------------------------------------------------------------

def tier_cost(cost: np.ndarray, n: int) -> np.ndarray:
    """Per question and tier prices from (3,) or (n, 3)."""
    return np.broadcast_to(np.asarray(cost, float), (n, 3))


def gain_point(P: np.ndarray, starts: np.ndarray, cost: np.ndarray = L.SERVICE_COST) -> dict:
    """A, C, H, G and start shares for one decision vector."""
    n = len(P)
    c = tier_cost(cost, n)
    rows = np.arange(n)
    A, Cr = float(P[rows, starts].mean()), float(c[rows, starts].mean())
    fa, fc = P.mean(axis=0), c.mean(axis=0)
    H = float(T.envelope_batch(fc[None], fa[None], np.array([Cr]))[0])
    share = np.bincount(starts, minlength=3) / n
    return {"A": A, "C": Cr, "H": H, "G": A - H, **{f"start_{t}": float(s) for t, s in zip(TIERS, share)}}


def first_cost_targets(P: np.ndarray, cost: np.ndarray = L.SERVICE_COST) -> list:
    fc = tier_cost(cost, len(P)).mean(axis=0)
    lo, hi = fc.min(), fc.max()
    return [float(lo + f * (hi - lo)) for f in LEVELS]


def lambda_first(Y: np.ndarray, Phat: np.ndarray, targets: list, cost: np.ndarray = L.SERVICE_COST) -> list:
    """Per cost level: the one-shot lambda whose dev first-attempt cost comes closest to the target."""
    curve = [(float(lam), gain_point(Y, L.decide("one_shot", Phat, lam, cost=cost), cost)) for lam in L.LAMBDAS]
    rows = []
    for level, target in zip(LEVELS, targets):
        lam, res = min(curve, key=lambda item: (abs(item[1]["C"] - target), item[0]))
        rows.append({"level": level, "target_cost": target, "lambda": lam, **{f"dev_{k}": v for k, v in res.items()}})
    return rows


def boot_gain(boot: np.ndarray, a: int, c: int, fixed_a: list, fixed_c: list) -> np.ndarray:
    """G per bootstrap draw from column means: router first success/cost and the fixed tiers' ones."""
    return boot[:, a] - T.envelope_batch(boot[:, fixed_c], boot[:, fixed_a], boot[:, c])


# ---------------------------------------------------------------------------------------------
# lock (dev only)
# ---------------------------------------------------------------------------------------------

def lock(run_dir: Path, repo: Path, output_dir: Path, allow_missing: bool) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit("output directory must be new and empty")
    old_dir = run_dir / "lock"
    old = json.loads((old_dir / "selection-lock.json").read_text())
    data = L.load_outcomes(run_dir)
    excluded, exclusions = L.evaluation_exclusions(repo)
    for name, en in exclusions.items():
        if en["sha256"] != old["exclusions"][name]["sha256"]:
            raise SystemExit(f"exclusion list differs from the 07b lock: {en['file']}")
    dev = data["dev"][~data["dev"].sample_id.isin(excluded)].reset_index(drop=True)
    seen = set()

    def evaluated_only(scores: pd.DataFrame) -> pd.DataFrame:
        seen.update(excluded.intersection(scores.sample_id))
        return scores[~scores.sample_id.isin(excluded)].reset_index(drop=True)

    files, missing = L.score_files(run_dir, repo, allow_missing)
    group_cols = [f"p_{t}" for t in TIERS]
    complete_dev = dev[dev[group_cols].notna().all(axis=1)]
    targets = {s: first_cost_targets(complete_dev.loc[complete_dev.source_dataset.eq(s), group_cols].to_numpy(float))
               for s in SOURCES}
    blended = data["blended"]
    label_tables = {"r0_group": {s: L.class_table(blended.loc[blended.source_dataset.eq(s), "difficulty_label"],
                                                  blended.loc[blended.source_dataset.eq(s), [f"success_{t}" for t in TIERS]].to_numpy(float),
                                                  ("easy", "medium", "hard")) for s in SOURCES}}

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibrated").mkdir()
    by_method: dict = {}
    for (method, setting), path in files.items():
        if method != "rt08":
            by_method.setdefault(method, []).append((setting, path))
    selection_rows, methods = [], {}
    for method, settings in sorted(by_method.items()):
        evaluated = []
        for setting, path in settings:
            scores = evaluated_only(pd.read_parquet(path))
            oof, frozen = L.calibrate_method(method, scores, dev, group_cols, label_tables.get(method, {}), SOURCES)
            per_source = {}
            for source, (d, X, Y, Phat) in oof.items():
                per_source[source] = gain_point(Y, L.decide("tau", Phat, 0.0, tau=TAU))
                selection_rows.append({"method": method, "setting": setting, "source": source, "rule": "tau",
                                       "tau": TAU, **{f"dev_{k}": v for k, v in per_source[source].items()}})
            score = float(np.mean([r["G"] for r in per_source.values()]))
            evaluated.append((score, setting, path, frozen, oof))
            print(f"[gain-lock] {method} {setting}: dev G (tau .8) {score:+.4f}", flush=True)
        score, setting, path, frozen, oof = max(evaluated, key=lambda e: e[0])
        lam = {s: lambda_first(Y, Phat, targets[s]) for s, (d, X, Y, Phat) in oof.items()}
        out_path = output_dir / "calibrated" / f"{method}.parquet"
        frozen.to_parquet(out_path, index=False)
        before = old["methods"].get(method, {}).get("selected_setting")
        same = None
        if before == setting:  # same setting: the 07b calibration must be reproduced exactly
            ref = pd.read_parquet(old_dir / old["methods"][method]["calibrated_file"])
            cols = [f"p_hat_{t}" for t in TIERS]
            merged = frozen.merge(ref, on=["sample_id", "role"], suffixes=("", "_old"))
            same = float(np.nanmax(np.abs(merged[cols].to_numpy() - merged[[c + "_old" for c in cols]].to_numpy())))
            if len(merged) != len(frozen) or same > 1e-9:
                raise SystemExit(f"{method}: calibration differs from 07b for the same setting (max diff {same})")
        methods[method] = {
            "selected_setting": setting, "selected_setting_07b": before, "changed_from_07b": before != setting,
            "kind": L.KIND[method], "dev_mean_G": score, "settings_tried": len(evaluated),
            "scores_file": str(path.relative_to(run_dir)), "scores_sha256": C.sha256_file(path),
            "calibrated_file": f"calibrated/{method}.parquet", "calibrated_sha256": C.sha256_file(out_path),
            "max_diff_vs_07b_calibration": same, "lambda": lam}
    if by_method and seen != excluded:
        raise SystemExit(f"{len(excluded - seen)} excluded ids are not in the scores")

    references = {}
    for name in ("r0_operational", "rt08"):  # 07b group-basis calibration, A split
        info = old["rt08"]["group"]["models"][name]
        if C.sha256_file(old_dir / info["calibrated_file"]) != info["calibrated_sha256"]:
            raise SystemExit(f"07b calibration changed: {info['calibrated_file']}")
        references[name] = {"calibrated_file": f"lock/{info['calibrated_file']}", "calibrated_sha256": info["calibrated_sha256"]}
    seeds = {}
    seeds_lock = run_dir / "seeds" / "seeds-lock.json"
    if seeds_lock.exists():
        rec = json.loads(seeds_lock.read_text())
        for key, v in rec["models"].items():
            path = run_dir / "seeds" / "calibrated" / f"{key}.parquet"
            if C.sha256_file(path) != v["calibrated_sha256"]:
                raise SystemExit(f"09 calibration changed: {path}")
            seeds[key] = {"calibrated_file": f"seeds/calibrated/{key}.parquet", "calibrated_sha256": v["calibrated_sha256"]}

    table = pd.DataFrame(selection_rows)
    table.to_parquet(output_dir / "selection-table.parquet", index=False)
    record = {
        "experiment": "EV-10 11_gain lock", "test_outcomes_read": False, "missing_scores": missing,
        "base_lock_sha256": C.sha256_file(old_dir / "selection-lock.json"),
        "inputs": data["hashes"] | {"dev_rows": complete_dev.groupby("source_dataset").size().to_dict()},
        "exclusions": exclusions,
        "evaluation": {
            "metric": "G = A - H(C); A = expected first-attempt success (group pass probability of the start tier), "
                      "C = mean first-attempt price, H = best fixed start tier or mixture with first-attempt cost <= C",
            "service_cost": L.SERVICE_COST.tolist(), "tau": TAU, "tau_curve": list(L.TAU_CURVE),
            "cost_levels": {"fractions": list(LEVELS), "targets": targets,
                            "definition": "fraction of each source's range between the cheapest and the dearest fixed "
                                          "start tier, first attempt only (dev)"},
            "selection": "07b adapters (5-fold cross-fitted on dev); setting score = dev G of the tau rule (tau = .8), "
                         "mean over the four sources",
            "lambda": "one-shot rule argmax(p_hat - lambda x price); per source and level the lambda whose dev "
                      "first-attempt cost is closest to the target (07b grid)",
            "lambda_grid": [float(L.LAMBDAS[0]), float(L.LAMBDAS[1]), float(L.LAMBDAS[-1]), len(L.LAMBDAS)],
        },
        "methods": methods, "references": references, "seeds": seeds,
        "selection_table_sha256": C.sha256_file(output_dir / "selection-table.parquet"),
    }
    (output_dir / "gain-lock.json").write_text(json.dumps(record, ensure_ascii=False, indent=1, default=float))
    (output_dir / "gain-lock-summary.md").write_text(lock_markdown(record, table))
    return record


def lock_markdown(record: dict, table: pd.DataFrame) -> str:
    lines = ["# 11 gain lock 요약", "", "dev에서 정한 내용이다. test 결과는 읽지 않았다. 지표는 첫 시도 이득 G = A − H(C)다.", ""]
    if record["missing_scores"]:
        lines += [f"**빠진 점수:** {', '.join(record['missing_scores'])}", ""]
    ev = record["evaluation"]
    lines += ["## 비용 목표 (한 번에 고르기 규칙, 첫 시도 비용)", "", "| 원천 | dev 행 | 25% | 50% | 75% |", "|---|---:|---:|---:|---:|"]
    for s in SOURCES:
        lines.append(f"| {s} | {record['inputs']['dev_rows'].get(s, 0)} | "
                     + " | ".join(f"{c:.3f}" for c in ev["cost_levels"]["targets"][s]) + " |")
    sel = {m: v["selected_setting"] for m, v in record["methods"].items()}
    chosen = table[table.apply(lambda r: sel.get(r["method"]) == r["setting"], axis=1)]
    g = chosen.pivot_table(index="method", columns="source", values="dev_G")
    lines += ["", "## 방법별 선택 설정과 dev G (τ = .8)", "",
              "설정 선택 기준은 τ = .8 규칙의 dev G(원천 4개 평균)다. 07b(U 기준)와 다르면 표시한다.", "",
              "| 방법 | 선택 설정 | 07b 설정 | 시도 | " + " | ".join(SOURCES) + " | 평균 |",
              "|---|---|---|---:|" + "---:|" * (len(SOURCES) + 1)]
    for m, v in sorted(record["methods"].items(), key=lambda kv: -kv[1]["dev_mean_G"]):
        mark = " **(바뀜)**" if v["changed_from_07b"] else ""
        cells = " | ".join(f"{g.loc[m, s]:+.3f}" if s in g.columns else "–" for s in SOURCES)
        lines.append(f"| {m} | `{v['selected_setting']}`{mark} | `{v['selected_setting_07b']}` | {v['settings_tried']} | "
                     f"{cells} | {v['dev_mean_G']:+.3f} |")
    lines += ["", "## 설정별 dev G 전체", "", "| 방법 | 설정 | 평균 dev G |", "|---|---|---:|"]
    means = table.groupby(["method", "setting"]).dev_G.mean().reset_index()
    for r in means.sort_values(["method", "dev_G"], ascending=[True, False]).itertuples():
        mark = " ✓" if sel.get(r.method) == r.setting else ""
        lines.append(f"| {r.method} | `{r.setting}`{mark} | {r.dev_G:+.4f} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------------------------

def load_gain_lock(run_dir: Path, repo: Path, lock_dir: Path | None = None) -> dict:
    lock_dir = lock_dir if lock_dir is not None else run_dir / "gain-lock"
    record = json.loads((lock_dir / "gain-lock.json").read_text())
    if record.get("missing_scores"):
        raise SystemExit(f"the gain lock was a dry run (missing scores: {record['missing_scores']})")
    if record["test_outcomes_read"]:
        raise SystemExit("the gain lock says it read test outcomes")
    files = {m: (lock_dir / v["calibrated_file"], v["calibrated_sha256"]) for m, v in record["methods"].items()}
    files |= {m: (run_dir / v["calibrated_file"], v["calibrated_sha256"]) for m, v in record["references"].items()}
    files |= {m: (run_dir / v["calibrated_file"], v["calibrated_sha256"]) for m, v in record["seeds"].items()}
    for name, (path, digest) in files.items():
        if C.sha256_file(path) != digest:
            raise SystemExit(f"calibrated file changed since the gain lock: {path}")
    excluded = set()
    for name, en in record["exclusions"].items():
        if C.sha256_file(repo / en["file"]) != en["sha256"]:
            raise SystemExit(f"the evaluation exclusion list changed: {en['file']}")
        excluded |= set(pd.read_csv(repo / en["file"]).sample_id)
    return {"record": record, "files": {k: v[0] for k, v in files.items()},
            "sha256": C.sha256_file(lock_dir / "gain-lock.json"), "excluded": excluded}


def decision_sets(name: str, Phat: np.ndarray, needed: np.ndarray, lam: dict | None) -> list:
    """(rule, param, starts, bootstrapped) for one router on one source."""
    if name == "needed_tier":
        return [("tau", TAU, needed, True)]
    sets = [("tau", float(tau), L.decide("tau", Phat, 0.0, tau=float(tau)), abs(tau - TAU) < 1e-9) for tau in L.TAU_CURVE]
    if lam is not None:
        sets += [("one_shot", r["level"], L.decide("one_shot", Phat, r["lambda"]), True) for r in lam]
    return sets


def evaluate(run_dir: Path, repo: Path, data_root: Path, output_dir: Path, rehearse: bool, iterations: int,
             gain_lock_dir: Path | None = None) -> dict:
    if output_dir.exists():
        raise SystemExit(f"{output_dir} exists: the test is opened once (move it away only to redo a failed run)")
    t0 = time.time()
    glock = load_gain_lock(run_dir, repo, gain_lock_dir)
    rec = glock["record"]
    old = json.loads((run_dir / "lock" / "selection-lock.json").read_text())
    if C.sha256_file(run_dir / "lock" / "selection-lock.json") != rec["base_lock_sha256"]:
        raise SystemExit("the 07b lock changed since the gain lock")
    truth = T.load_truth(run_dir, repo, data_root, glock, rehearse)
    frame, role = truth["frame"], truth["role"]
    print(f"[gain] {role} rows evaluated: {truth['counts']['evaluated']}", flush=True)
    methods = sorted(m for m in rec["methods"] if m not in OMITTED)
    cal = {m: T.calibrated(glock["files"][m], frame, role) for m in methods + list(rec["references"])}
    prices, _ = L.group_prices(run_dir)
    length, length_info = L.length_table(run_dir)
    length_cfg = old["evaluation"]["sensitivity"]["length"]
    if length_info["sha256"] != length_cfg["lengths_sha256"]:
        raise SystemExit("length-inputs changed since 07b")
    output_dir.mkdir(parents=True)

    points, draws_kept, strata_rows, fixed_rows = [], {}, [], []
    group_cols = [f"p_{t}" for t in TIERS]
    for si, source in enumerate(SOURCES):
        rows = frame[frame.source_dataset.eq(source)].reset_index(drop=True)
        ids, n = rows.sample_id.to_numpy(), len(rows)
        P = rows[group_cols].to_numpy(float)
        needed = rows.needed.to_numpy()
        draws = T.Draws(rows.group_id, iterations, T.BOOT["seed"], si)
        bases = {"main": L.SERVICE_COST, "group_price": prices[source],
                 "length": L.length_costs(length, ids, length_cfg["normalizer_mean_dev_length"][source])}
        names = methods + ["needed_tier"] + (["r0_operational", "rt08"] if source in A_SOURCES else [])
        sets = {}
        for name in names:
            Phat = None if name == "needed_tier" else T.phat_for(cal[name], ids)
            lam = rec["methods"][name]["lambda"][source] if name in rec["methods"] else None
            sets[name] = decision_sets(name, Phat, needed, lam)
        seed_sets = {}
        for key in rec["seeds"]:
            c = T.calibrated(glock["files"][key], frame, role)
            seed_sets[key] = L.decide("tau", T.phat_for(c, ids), 0.0, tau=TAU)
        for basis, cost in bases.items():
            cmat = tier_cost(cost, n)
            cols, index = [P[:, t] for t in range(3)] + [cmat[:, t] for t in range(3)], {}
            fixed_a, fixed_c = [0, 1, 2], [3, 4, 5]
            fa, fc = P.mean(axis=0), cmat.mean(axis=0)
            fixed_rows += [{"basis": basis, "source": source, "start": t, "A": float(fa[k]), "C": float(fc[k])}
                           for k, t in enumerate(TIERS)]
            todo = [(name, rule, param, st, boot) for name, ss in sets.items() for rule, param, st, boot in ss
                    if basis == "main" or (rule == "tau" and abs(param - TAU) < 1e-9)]
            todo += [(key, "tau", TAU, st, True) for key, st in seed_sets.items() if basis == "main"]
            for name, rule, param, st, boot in todo:
                rec_point = {"basis": basis, "source": source, "method": name, "rule": rule, "param": param,
                             **gain_point(P, st, cost), "G_lo": np.nan, "G_hi": np.nan}
                points.append(rec_point)
                if boot:
                    index[(name, rule, param)] = (len(cols), len(cols) + 1)
                    cols += [P[np.arange(n), st], cmat[np.arange(n), st]]
            B = draws.means(np.column_stack(cols))
            for key, (a, c) in index.items():
                g = boot_gain(B, a, c, fixed_a, fixed_c).astype(np.float32)
                draws_kept[(basis, source) + key] = g
            for p in points:
                key = (p["method"], p["rule"], p["param"])
                if p["basis"] == basis and p["source"] == source and key in index:
                    p["G_lo"], p["G_hi"] = (float(v) for v in T.interval(draws_kept[(basis, source) + key]))
            if basis == "main":  # descriptive: G inside each solvability stratum (envelope rebuilt inside)
                for stratum, idx in rows.groupby("stratum").indices.items():
                    for name, rule, param, st, boot in todo:
                        if boot:
                            strata_rows.append({"source": source, "stratum": stratum, "rows": len(idx), "method": name,
                                                "rule": rule, "param": param, **gain_point(P[idx], st[idx], cost)})
        print(f"[gain] {source} done ({time.time() - t0:.0f}s)", flush=True)

    pts = pd.DataFrame(points)
    macro = macro_table(pts, draws_kept)
    pairs = pair_table(draws_kept, pts, methods)
    strata = pd.DataFrame(strata_rows)
    seeds = seeds_table(draws_kept, pts) if rec["seeds"] else pd.DataFrame()
    pts.to_parquet(output_dir / "gain.parquet", index=False)
    macro.to_parquet(output_dir / "gain-macro.parquet", index=False)
    pairs.to_parquet(output_dir / "pairs.parquet", index=False)
    strata.to_parquet(output_dir / "by-stratum.parquet", index=False)
    pd.DataFrame(fixed_rows).to_parquet(output_dir / "fixed-starts.parquet", index=False)
    if len(seeds):
        seeds.to_parquet(output_dir / "seeds.parquet", index=False)
    manifest = {"experiment": "EV-10 11_gain test", "rehearsal_on_dev": rehearse, "test_outcomes_read": not rehearse,
                "role_evaluated": role, "gain_lock_sha256": glock["sha256"], "outcomes_sha256": truth["test_sha256"],
                "input_sha256": truth["input_sha256"], "counts": truth["counts"],
                "bootstrap": T.BOOT | {"iterations": iterations}, "methods": methods, "references": list(REFERENCES),
                "omitted": [m for m in OMITTED if m in rec["methods"]],
                "outputs": {p.name: C.sha256_file(p) for p in sorted(output_dir.glob("*.parquet"))},
                "seconds": round(time.time() - t0, 1)}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=float))
    (output_dir / "gain-summary.md").write_text(test_markdown(manifest, rec, macro, pairs, strata, seeds))
    return manifest


SCOPES = {"macro": SOURCES, "a_split": A_SOURCES, "external": EXTERNAL}


def macro_table(pts: pd.DataFrame, kept: dict) -> pd.DataFrame:
    """Per scope: unweighted mean over its sources (point), same-index mean of the source draws (interval)."""
    rows = []
    for (basis, name, rule, param), g in pts.groupby(["basis", "method", "rule", "param"], sort=False):
        for scope, sources in SCOPES.items():
            part = g[g.source.isin(sources)]
            if part.source.nunique() != len(sources):
                continue
            row = {"basis": basis, "scope": scope, "method": name, "rule": rule, "param": param,
                   **{k: float(part[k].mean()) for k in ("A", "C", "H", "G", "start_low", "start_medium", "start_high")}}
            keys = [(basis, s, name, rule, param) for s in sources]
            if all(k in kept for k in keys):
                row["G_lo"], row["G_hi"] = (float(v) for v in T.interval(np.mean([kept[k] for k in keys], axis=0)))
            rows.append(row)
    out = pd.DataFrame(rows)
    ranked = out.method.isin(set(out.method) - set(REFERENCES)) & ~out.method.str.contains(r"_s(?:17|73)$")
    out["rank"] = np.nan
    out.loc[ranked, "rank"] = out[ranked].groupby(["basis", "scope", "rule", "param"]).G.rank(ascending=False, method="min")
    return out


def pair_table(kept: dict, pts: pd.DataFrame, methods: list) -> pd.DataFrame:
    """Paired differences of G (row minus column method), main basis, macro over the four sources."""
    rows = []
    for rule, param in (("tau", TAU),) + tuple(("one_shot", lv) for lv in LEVELS):
        have = [m for m in methods if all(("main", s, m, rule, param) in kept for s in SOURCES)]
        mean = {m: np.mean([kept[("main", s, m, rule, param)] for s in SOURCES], axis=0) for m in have}
        point = {m: float(pts[(pts.basis.eq("main")) & pts.method.eq(m) & pts.rule.eq(rule)
                              & np.isclose(pts.param, param)].G.mean()) for m in have}
        for i, a in enumerate(have):
            for b in have[i + 1:]:
                lo, hi = T.interval(mean[a] - mean[b])
                rows.append({"rule": rule, "param": param, "a": a, "b": b, "dG": point[a] - point[b],
                             "lo": float(lo), "hi": float(hi), "separated": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def seeds_table(kept: dict, pts: pd.DataFrame) -> pd.DataFrame:
    """H5 descriptive under G (tau .8): mixed minus A-only, per scope, each seed and the three-seed mean."""
    rows = []
    variants = {42: {b: b for b in PAIR}} | {s: {b: f"{b}_s{s}" for b in PAIR} for s in SEEDS}
    for scope, sources in SCOPES.items():
        per_seed = {}
        for seed, names in variants.items():
            keys = {b: [("main", s, names[b], "tau", TAU) for s in sources] for b in PAIR}
            if not all(k in kept for ks in keys.values() for k in ks):
                continue
            d = np.mean([kept[k] for k in keys[PAIR[0]]], axis=0) - np.mean([kept[k] for k in keys[PAIR[1]]], axis=0)
            pt = {b: float(pts[(pts.basis.eq("main")) & pts.method.eq(names[b]) & pts.rule.eq("tau")
                               & np.isclose(pts.param, TAU) & pts.source.isin(sources)].G.mean()) for b in PAIR}
            per_seed[seed] = (pt[PAIR[0]] - pt[PAIR[1]], d)
            lo, hi = T.interval(d)
            rows.append({"scope": scope, "seed": str(seed), "dG": per_seed[seed][0], "lo": float(lo), "hi": float(hi)})
        if len(per_seed) == len(variants):
            lo, hi = T.interval(np.mean([d for _, d in per_seed.values()], axis=0))
            rows.append({"scope": scope, "seed": "mean", "dG": float(np.mean([p for p, _ in per_seed.values()])),
                         "lo": float(lo), "hi": float(hi)})
    return pd.DataFrame(rows)


def test_markdown(manifest: dict, rec: dict, macro: pd.DataFrame, pairs: pd.DataFrame, strata: pd.DataFrame,
                  seeds: pd.DataFrame) -> str:
    head = "dev 리허설(해석하지 않음)" if manifest["rehearsal_on_dev"] else "test"
    lines = [f"# 11 gain 결과 — {head}", "",
             "G = A − H(C). A는 첫 시도 성공(집단 풀 확률), C는 첫 시도 비용(1:3:5), H는 같은 비용 이하의 고정 시작 단계(혼합 포함)가 "
             "낼 수 있는 가장 좋은 첫 시도 성공이다. 구간은 질문 묶음 단위 paired bootstrap 95%. 참고 행(needed_tier, 운영 R0, RT-08)은 "
             "순위에 넣지 않는다.", ""]
    main = macro[macro.basis.eq("main") & macro.scope.eq("macro")]

    def table(rule: str, param: float, title: str) -> list:
        part = main[main.rule.eq(rule) & np.isclose(main.param, param)].sort_values("G", ascending=False)
        out = [f"## {title}", "", "| 방법 | G | 95% 구간 | A | C | 시작 low/medium/high |", "|---|---:|---|---:|---:|---|"]
        for r in part.itertuples():
            ci = f"[{r.G_lo:+.3f}, {r.G_hi:+.3f}]" if not np.isnan(getattr(r, "G_lo", np.nan)) else "–"
            tag = " (참고)" if r.method in REFERENCES else ""
            out.append(f"| {r.method}{tag} | {r.G:+.3f} | {ci} | {r.A:.3f} | {r.C:.3f} | "
                       f"{r.start_low:.2f} / {r.start_medium:.2f} / {r.start_high:.2f} |")
        return out + [""]

    lines += table("tau", TAU, "τ = .8 규칙 (주 결과, 원천 4개 평균)")
    for lv in LEVELS:
        lines += table("one_shot", lv, f"한 번에 고르기 규칙, 비용 수준 {int(lv * 100)}%")
    sens = macro[macro.scope.eq("macro") & macro.rule.eq("tau") & np.isclose(macro.param, TAU)].pivot_table(
        index="method", columns="basis", values="G")
    lines += ["## 비용 민감도 (τ = .8, 원천 4개 평균 G)", "", "| 방법 | 1:3:5 | 집단 가격 | 길이 반영 |", "|---|---:|---:|---:|"]
    for m, r in sens.sort_values("main", ascending=False).iterrows():
        lines.append(f"| {m} | {r.get('main', np.nan):+.3f} | {r.get('group_price', np.nan):+.3f} | {r.get('length', np.nan):+.3f} |")
    sep = pairs[pairs.rule.eq("tau") & pairs.separated]
    lines += ["", f"## 라우터 간 차이 (τ = .8)", "", f"{len(pairs[pairs.rule.eq('tau')])}쌍 중 95% 구간이 0을 포함하지 않는 쌍: {len(sep)}개 "
              "(`pairs.parquet`).", ""]
    if len(strata):
        st = strata[strata.rule.eq("tau") & np.isclose(strata.param, TAU)].groupby(["method", "stratum"]).G.mean().unstack()
        st = st[[c for c in T.STRATA if c in st.columns]]
        lines += ["## 풀리는 정도별 층 (τ = .8, 원천 평균, 층 안에서 기준선을 다시 만듦, 기술 통계)", "",
                  "| 방법 | " + " | ".join(st.columns) + " |", "|---|" + "---:|" * len(st.columns)]
        for m, r in st.sort_values(st.columns[0], ascending=False).iterrows():
            lines.append(f"| {m} | " + " | ".join(f"{v:+.3f}" for v in r) + " |")
        lines.append("")
    if len(seeds):
        lines += ["## H5 (기술 통계, τ = .8): R1-집단 − R1-집단-A", "", "| 범위 | seed | ΔG | 95% 구간 |", "|---|---|---:|---|"]
        for r in seeds.itertuples():
            lines.append(f"| {r.scope} | {r.seed} | {r.dG:+.3f} | [{r.lo:+.3f}, {r.hi:+.3f}] |")
        lines.append("")
    changed = [m for m, v in rec["methods"].items() if v["changed_from_07b"]]
    lines += ["## 설정", "", f"07b(U 기준)와 다른 설정을 고른 방법: {', '.join(changed) if changed else '없음'}.", ""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("lock")
    a.add_argument("--run-dir", type=Path, required=True)
    a.add_argument("--repo", type=Path, required=True)
    a.add_argument("--output-dir", type=Path, required=True, help="normally <run>/gain-lock")
    a.add_argument("--allow-missing", action="store_true", help="dry run with the scores available so far")
    b = sub.add_parser("test")
    b.add_argument("--run-dir", type=Path, required=True)
    b.add_argument("--repo", type=Path, required=True)
    b.add_argument("--data-root", type=Path, required=True)
    b.add_argument("--output-dir", type=Path, required=True, help="normally <run>/gain-test")
    b.add_argument("--rehearse-on-dev", action="store_true")
    b.add_argument("--iterations", type=int, default=T.BOOT["iterations"])
    args = p.parse_args(argv)
    if args.cmd == "lock":
        out = lock(args.run_dir, args.repo, args.output_dir, args.allow_missing)
        print(json.dumps({m: {k: v[k] for k in ("selected_setting", "selected_setting_07b", "dev_mean_G")}
                          for m, v in out["methods"].items()}, indent=1, default=float))
    else:
        out = evaluate(args.run_dir, args.repo, args.data_root, args.output_dir, args.rehearse_on_dev, args.iterations)
        print(json.dumps({k: out[k] for k in ("role_evaluated", "counts", "seconds")}, indent=1, default=float))


if __name__ == "__main__":
    main()
