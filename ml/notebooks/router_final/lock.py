"""EV-10 07b_lock: calibrate every router on dev, pick one setting per method, and freeze the decision rules.

Reads dev outcomes (00_prepare dev.parquet) and the dev/test/RouterArena scores written by 03-06 and 07a.
Test outcomes are not read. Evaluation uses English questions without a provided document only: the questions in
EXCLUSIONS (sample-id lists, no text) are dropped from dev and from every calibrated output, so 08_test never sees
them. Writes <output>/:

  selection-lock.json      what 08_test may use: selected setting per method, adapters' inputs, rho, cost
                           targets, lambda tables, decision rules, SHA-256 of every file it depends on
  calibrated/<name>.parquet per-tier success probabilities for the dev and test rows of the four sources
  selection-table.parquet  dev score of every setting (the selection evidence)

Evaluation model (plan 2.1, 2.5): calling tier t succeeds with the group success probability p_t (a random
model of the group); after a failure the next tier is called. For a start tier s the expected first-attempt
success is p_s, resolution 1 - prod_{t>=s}(1 - p_t), cost sum_t c_t prod_{s<=u<t}(1 - p_u) and failed attempts
sum_{t>=s} prod_{s<=u<=t}(1 - p_u), with service costs c = (.2, .6, 1.0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_track as C  # noqa: E402

TIERS = C.TIERS
SOURCES = C.SOURCES
A_SOURCES = ("routerbench", "xroutebench")
SERVICE_COST = np.array([0.2, 0.6, 1.0])
KAPPAS = np.round(np.arange(0, 1.0001, 0.05), 2)  # plan 2.5: 0 to 1 in .05 steps
LEVELS = (0.25, 0.5, 0.75)  # cost targets as fractions of each source's fixed-start cost range (see cost_targets)
TAU = 0.8
TAU_CURVE = tuple(np.round(np.arange(0.5, 0.9501, 0.05), 2))
LAMBDAS = np.concatenate([[0.0], np.geomspace(1e-3, 100.0, 240)])
FOLDS = 5
RULES = ("escalation", "one_shot")

# How a method's scores become three tier features before isotonic calibration.
KIND = {"knn": "per_tier", "knn_a": "per_tier", "mlp": "per_tier", "svm": "per_tier", "mf": "per_tier",
        "graphrouter": "per_tier", "irt_router": "per_tier", "avengers_pro": "per_tier", "r1_bge": "per_tier",
        "routerdc": "per_tier", "r1_group": "per_tier", "r1_group_a": "per_tier",
        "r2_group": "per_tier", "r3_group": "per_tier",
        "hybrid_llm": "single", "routellm_retrained": "single", "routellm_released": "single",
        "r0_group": "class_probs", "rt08": "class_probs", "r0_operational": "class_probs"}
OURS_03 = ("r0_group", "r1_group", "r1_group_a")
OPERATIONAL_R0 = "experiments/results/context-length-qualification-20260925/r0/scores"
NON_ENGLISH = "experiments/results/router-final-inputs/non-english-eval-exclusions.csv"  # see the .md beside it
DOCUMENT_GROUNDED = "experiments/results/router-final-inputs/document-grounded-eval-exclusions.csv"  # same .md folder
EXCLUSIONS = {"non_english": NON_ENGLISH, "document_grounded": DOCUMENT_GROUNDED}  # evaluation only, never training
OUTPUT_WEIGHT = 5.0  # length cost: Claude output price / input price (Haiku 1/5, Sonnet 3/15, Opus 5/25 USD per 1M)


# ---------------------------------------------------------------------------------------------
# Evaluation core (pure numpy; unit-tested)
# ---------------------------------------------------------------------------------------------

def outcomes(P: np.ndarray, start: np.ndarray, cost: np.ndarray = SERVICE_COST) -> dict:
    """Per-question expectations when starting at `start` and escalating on failure (independent draws).

    `cost` is one price per tier (3,) or one per question and tier (n, 3) (length-cost sensitivity)."""
    n = len(P)
    reach, spent, fails = np.ones(n), np.zeros(n), np.zeros(n)
    for t in range(3):
        on = t >= start
        spent += np.where(on, reach * cost[..., t], 0.0)
        fails += np.where(on, reach * (1 - P[:, t]), 0.0)
        reach = np.where(on, reach * (1 - P[:, t]), reach)
    return {"first": P[np.arange(n), start], "resolved": 1 - reach, "cost": spent, "fails": fails}


def summary(P: np.ndarray, start: np.ndarray, cost: np.ndarray = SERVICE_COST) -> dict:
    return {k: float(v.mean()) for k, v in outcomes(P, start, cost).items()}


def fixed_points(P: np.ndarray, cost: np.ndarray = SERVICE_COST) -> list:
    return [summary(P, np.full(len(P), s), cost) for s in range(3)]


def envelope(points: list, kappa: float, x: float) -> float:
    """Best value resolved - kappa*fails over fixed starts and their mixtures with expected cost <= x.

    Below the cheapest fixed start the cheapest one is used (a policy cannot spend less there).
    """
    pts = sorted((p["cost"], p["resolved"] - kappa * p["fails"]) for p in points)
    hull = []
    for c, v in pts:  # upper concave hull, left to right
        while len(hull) >= 2 and (hull[-1][1] - hull[-2][1]) * (c - hull[-2][0]) <= (v - hull[-2][1]) * (hull[-1][0] - hull[-2][0]):
            hull.pop()
        hull.append((c, v))
    if x <= hull[0][0]:
        return hull[0][1]
    best = hull[0][1]
    for (c0, v0), (c1, v1) in zip(hull, hull[1:]):
        if x >= c1:
            best = max(best, v1)
        elif x > c0:
            best = max(best, v0 + (v1 - v0) * (x - c0) / (c1 - c0))
    return best


def estimate_rho(P: np.ndarray) -> float:
    """Dependence of tier failures across questions (EV-05): P(fa & fb) = rho*min + (1-rho)*prod."""
    f = 1 - P
    pairs = ((0, 1), (1, 2), (0, 2))
    num = sum((f[:, a] * f[:, b]).mean() - f[:, a].mean() * f[:, b].mean() for a, b in pairs)
    den = sum(min(f[:, a].mean(), f[:, b].mean()) - f[:, a].mean() * f[:, b].mean() for a, b in pairs)
    return float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 0.0


def predicted_chain(Phat: np.ndarray, rho: float, cost: np.ndarray = SERVICE_COST):
    """Predicted resolution, cost and failed attempts for each start tier (EV-05 escalation-aware rule)."""
    n = len(Phat)
    R, Cst, Fl = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
    for s in range(3):
        f = 1 - Phat[:, s:]
        joint = [f[:, 0]]
        for k in range(1, f.shape[1]):
            prev = joint[-1]
            joint.append(rho * np.minimum(prev, f[:, k]) + (1 - rho) * prev * f[:, k])
        R[:, s] = 1 - joint[-1]
        Cst[:, s] = cost[..., s] + sum(joint[k - 1] * cost[..., s + k] for k in range(1, f.shape[1]))
        Fl[:, s] = sum(joint)
    return R, Cst, Fl


def decide(rule: str, Phat: np.ndarray, lam: float, kappa: float = 0.0, chain=None, tau: float = TAU,
           cost: np.ndarray = SERVICE_COST) -> np.ndarray:
    if rule == "one_shot":
        return np.argmax(Phat - lam * cost, axis=1)
    if rule == "escalation":
        R, Cst, Fl = chain
        return np.argmax(R - lam * Cst - kappa * Fl, axis=1)
    if rule == "tau":  # track 2: cheapest tier with p >= tau, else the most likely (ties to the cheaper)
        sure = Phat >= tau
        return np.where(sure.any(axis=1), sure.argmax(axis=1), Phat.argmax(axis=1))
    raise ValueError(rule)


def cost_targets(P: np.ndarray, cost: np.ndarray = SERVICE_COST) -> list:
    costs = [p["cost"] for p in fixed_points(P, cost)]
    lo, hi = min(costs), max(costs)
    return [lo + f * (hi - lo) for f in LEVELS]


def lambda_table(P: np.ndarray, Phat: np.ndarray, rho: float, targets: list, cost: np.ndarray = SERVICE_COST) -> dict:
    """For each rule, kappa and cost target: the lambda whose dev decisions come closest to the target cost,
    with the resulting dev cost, value and gain over the fixed-start envelope (all priced with `cost`)."""
    points = fixed_points(P, cost)
    chain = predicted_chain(Phat, rho, cost)
    out = {}
    for rule in RULES:
        kappas = KAPPAS if rule == "escalation" else [0.0]
        rows = []
        for kappa in kappas:
            curve = [(lam, summary(P, decide(rule, Phat, lam, kappa, chain, cost=cost), cost)) for lam in LAMBDAS]
            for level, target in zip(LEVELS, targets):
                lam, res = min(curve, key=lambda item: (abs(item[1]["cost"] - target), item[0]))
                rows.append({"kappa": float(kappa), "level": level, "target_cost": target, "lambda": float(lam),
                             **{f"dev_{k}": v for k, v in res.items()}})
        out[rule] = rows
    for row in out["escalation"]:
        row["dev_value"] = row["dev_resolved"] - row["kappa"] * row["dev_fails"]
        row["dev_envelope"] = envelope(points, row["kappa"], row["dev_cost"])
        row["dev_gain"] = row["dev_value"] - row["dev_envelope"]
    return out


# ---------------------------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------------------------

def softmax(S: np.ndarray) -> np.ndarray:
    z = np.exp(S - S.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def features(kind: str, S: np.ndarray, table: np.ndarray | None = None) -> np.ndarray:
    if kind == "per_tier":
        return S
    if kind == "single":
        return np.repeat(S[:, :1], 3, axis=1)
    if kind == "class_probs":  # EV-05: class probabilities x P(tier succeeds | class) from train
        return softmax(S) @ table
    raise ValueError(kind)


def class_table(labels: pd.Series, Y: np.ndarray, classes: tuple) -> np.ndarray:
    rows = []
    for c in classes:
        m = (labels == c).to_numpy()
        rows.append(Y[m].mean(axis=0) if m.any() else Y.mean(axis=0))
    return np.vstack(rows)


def fit_predict(Xfit: np.ndarray, Yfit: np.ndarray, Xapply: np.ndarray) -> np.ndarray:
    from sklearn.isotonic import IsotonicRegression
    out = np.zeros((len(Xapply), 3))
    for j in range(3):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing="auto", out_of_bounds="clip")
        out[:, j] = iso.fit(Xfit[:, j], Yfit[:, j]).predict(Xapply[:, j])
    return out


def fold_of(sample_ids) -> np.ndarray:
    return np.array([int(hashlib.sha1(s.encode()).hexdigest()[:8], 16) % FOLDS for s in sample_ids])


def cross_fit(X: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    out = np.zeros_like(Y, dtype=float)
    for k in range(FOLDS):
        m = folds == k
        out[m] = fit_predict(X[~m], Y[~m], X[m])
    return out


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------

def load_outcomes(run_dir: Path) -> dict:
    """Dev outcomes and train tables. Test outcomes are not read."""
    prepare_dir, data_dir = run_dir / "prepare", run_dir / "ours" / "data"
    manifest = json.loads((prepare_dir / "manifest.json").read_text())
    for name in ("train.parquet", "dev.parquet"):
        if C.sha256_file(prepare_dir / name) != manifest["outputs"][name]["sha256"]:
            raise SystemExit(f"00_prepare output changed: {name}")
    data_manifest = json.loads((data_dir / "manifest.json").read_text())
    if C.sha256_file(data_dir / "blended" / "train.parquet") != data_manifest["variants"]["blended"]["files"]["train.parquet"]:
        raise SystemExit("03_ours training data changed")
    p_cols = [f"p_{t}" for t in TIERS]
    arm_cols = [f"arm_success_v2_{t}" for t in TIERS]
    dev = pd.read_parquet(prepare_dir / "dev.parquet", columns=["sample_id", "source_dataset", "excluded_v2", "needed_tier"]
                          + p_cols + arm_cols)
    dev = dev[~dev.excluded_v2.fillna(False).astype(bool)]
    train = pd.read_parquet(prepare_dir / "train.parquet", columns=["sample_id", "source_dataset", "excluded_v2",
                                                                    "old_label_v1", "old_label_v2"] + p_cols + arm_cols)
    train = train[~train.excluded_v2.fillna(False).astype(bool)]
    blended = pd.read_parquet(data_dir / "blended" / "train.parquet",
                              columns=["source_dataset", "difficulty_label"] + [f"success_{t}" for t in TIERS])
    return {"dev": dev.reset_index(drop=True), "train": train.reset_index(drop=True), "blended": blended,
            "hashes": {"prepare_manifest": C.sha256_file(prepare_dir / "manifest.json"),
                       "ours_data_manifest": C.sha256_file(data_dir / "manifest.json")}}


def group_prices(run_dir: Path) -> tuple:
    """Sensitivity cost (plan 2.3): per source, the mean data price of each tier's group models, high = 1.

    Calling a tier calls a random model of its group (plan 2.1), so the group mean is that call's expected price.
    Model prices are 00_prepare's train means (SPROUT: the data authors' price table)."""
    prepare_dir = run_dir / "prepare"
    groups = json.loads((prepare_dir / "groups.json").read_text())
    sealed = json.loads((prepare_dir / "manifest.json").read_text())["summary"]["groups"]
    prices = {}
    for source in SOURCES:
        kept = pd.DataFrame(groups[source])
        kept = kept[kept.kept.astype(bool)]
        if {t: sorted(kept.loc[kept.tier.eq(t), "model"]) for t in TIERS} != {t: sorted(sealed[source][t]) for t in TIERS}:
            raise SystemExit(f"groups.json does not match the sealed 00_prepare groups: {source}")
        mean = kept.groupby("tier").cost.mean().reindex(list(TIERS)).to_numpy(float)
        prices[source] = mean / mean[-1]
    return prices, C.sha256_file(prepare_dir / "groups.json")


def length_table(run_dir: Path) -> tuple:
    """Plan 2.3 sensitivity 1: token-weighted length L = input + 5 x median output tokens per question."""
    folder = run_dir / "length-inputs"
    manifest = json.loads((folder / "manifest.json").read_text())
    path = folder / "lengths.parquet"
    if C.sha256_file(path) != manifest["outputs"]["lengths.parquet"]:
        raise SystemExit("length-inputs changed since it was built")
    t = pd.read_parquet(path)
    filled = {}
    for source in SOURCES:  # no valid answer recorded: the source's dev median output length
        rows = t.source_dataset.eq(source)
        gap = rows & t.output_tokens_median.isna()
        filled[source] = int(gap.sum())
        t.loc[gap, "output_tokens_median"] = t.loc[rows & t.role.eq("dev"), "output_tokens_median"].median()
    t["length"] = t.input_tokens + OUTPUT_WEIGHT * t.output_tokens_median
    return t.drop_duplicates("sample_id").set_index("sample_id").length, {"sha256": C.sha256_file(path), "filled": filled}


def length_costs(length: pd.Series, ids, normalizer: float) -> np.ndarray:
    """Per question and tier: tier price x length / (the source's mean dev length)."""
    L = length.reindex(ids).to_numpy(float)
    if not np.isfinite(L).all():
        raise SystemExit("questions without a length")
    return (L / normalizer)[:, None] * SERVICE_COST[None, :]


def lambda_record(tables: dict) -> dict:
    return {s: {rule: [{k: r[k] for k in ("kappa", "level", "lambda", "dev_cost")} for r in t[rule]] for rule in RULES}
            for s, t in tables.items()}


def mean_gain(tables: dict) -> float:
    return float(np.mean([np.mean([r["dev_gain"] for r in t["escalation"]]) for t in tables.values()]))


def exclusion_list(repo: Path, rel: str) -> tuple:
    path = repo / rel
    table = pd.read_csv(path)
    if not table.sample_id.is_unique:
        raise SystemExit(f"duplicate sample ids in {rel}")
    return set(table.sample_id), {"file": rel, "sha256": C.sha256_file(path), "rows": int(len(table)),
                                  "by_source_role": {f"{s}/{r}": int(n) for (s, r), n
                                                     in table.groupby(["source_dataset", "role"]).size().items()}}


def evaluation_exclusions(repo: Path) -> tuple:
    """Evaluation questions dropped by the data contract: non-English (the service and its BGE are English-only)
    and document-grounded QA (RAG-type: a passage is given and the answer must come from it)."""
    ids, records = set(), {}
    for name, rel in EXCLUSIONS.items():
        part, records[name] = exclusion_list(repo, rel)
        ids |= part
    return ids, records


def check(path: Path, digest: str) -> None:
    if C.sha256_file(path) != digest:
        raise SystemExit(f"scores changed since their run recorded them: {path}")


def score_files(run_dir: Path, repo: Path, allow_missing: bool) -> tuple:
    """Every (method, setting) -> scores.parquet path, from the manifests of 04, 05, 06 and 07a."""
    files, missing = {}, []
    common = run_dir / "common" / "manifest.json"
    if common.exists():
        manifest = json.loads(common.read_text())
        for method, settings in manifest["settings"].items():
            for s in settings:
                key = f"{method}/{s}"
                path = run_dir / "common" / method / s / "scores.parquet"
                if key in manifest["outputs"]:
                    check(path, manifest["outputs"][key])
                    files[(method, s)] = path
                else:
                    missing.append(key)
    else:
        missing.append("common/manifest.json")
    for method in ("routerdc", "routellm_retrained", "routellm_released"):
        mf = run_dir / "finetune" / method / "manifest.json"
        if not mf.exists():
            missing.append(f"finetune/{method}")
            continue
        manifest = json.loads(mf.read_text())
        for s in manifest["settings"]:
            path = run_dir / "finetune" / method / s / "scores.parquet"
            if s in manifest["outputs"]:
                check(path, manifest["outputs"][s])
                files[(method, s)] = path
            else:
                missing.append(f"{method}/{s}")
    for name in OURS_03 + ("rt08",):
        path = run_dir / "scores-modernbert" / name / "scores.parquet"
        if (path.parent / "_SUCCESS").exists():
            check(path, json.loads((path.parent / "metrics.json").read_text())["scores_sha256"])
            files[(name, "default")] = path
        else:
            missing.append(name)
    if missing and not allow_missing:
        raise SystemExit("missing scores (run 04, 05a, 05b, 06 and 07a first): " + ", ".join(missing))
    return files, missing


def operational_r0(repo: Path) -> pd.DataFrame:
    folder = repo / OPERATIONAL_R0
    parts = []
    for name, role in (("validation-512", "dev"), ("test_routerbench-512", "test"), ("test_xroutebench-512", "test")):
        t = pd.read_parquet(folder / f"{name}.parquet", columns=["sample_id"] + [f"score_{t}" for t in TIERS])
        parts.append(t.assign(role=role))
    out = pd.concat(parts, ignore_index=True).drop_duplicates("sample_id")
    out["source_dataset"] = out.sample_id.str.split(":").str[0]
    return out


# ---------------------------------------------------------------------------------------------

def calibrate_method(name: str, scores: pd.DataFrame, dev: pd.DataFrame, basis_cols: list, tables: dict,
                     sources: tuple) -> tuple:
    """OOF (selection) and full-dev (frozen) calibrated probabilities per source."""
    kind = KIND[name]
    score_cols = [f"score_{t}" for t in TIERS]
    oof, frozen = {}, []
    for source in sources:
        d = dev[dev.source_dataset.eq(source)]
        d = d[d[basis_cols].notna().all(axis=1)]
        s = scores.set_index("sample_id")
        if not d.sample_id.isin(s.index).all():
            raise SystemExit(f"{name}: dev rows without scores in {source}")
        X = features(kind, s.loc[d.sample_id, score_cols].to_numpy(float), tables.get(source))
        Y = d[basis_cols].to_numpy(float)
        oof[source] = (d, X, Y, cross_fit(X, Y, fold_of(d.sample_id)))
        rows = scores[scores.source_dataset.eq(source)]
        Xall = features(kind, rows[score_cols].to_numpy(float), tables.get(source))
        ok = np.isfinite(Xall).all(axis=1)
        P = np.full((len(rows), 3), np.nan)
        P[ok] = fit_predict(X, Y, Xall[ok])
        frozen.append(rows[["sample_id", "source_dataset", "role"]].assign(
            **{f"p_hat_{t}": P[:, k] for k, t in enumerate(TIERS)}))
    return oof, pd.concat(frozen, ignore_index=True)


def lock(run_dir: Path, repo: Path, output_dir: Path, allow_missing: bool) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit("output directory must be new and empty")
    data = load_outcomes(run_dir)
    excluded, exclusions = evaluation_exclusions(repo)
    dev = data["dev"]
    for name, rel in EXCLUSIONS.items():
        exclusions[name]["dev_rows_removed"] = int(dev.sample_id.isin(exclusion_list(repo, rel)[0]).sum())
    dev = dev[~dev.sample_id.isin(excluded)].reset_index(drop=True)
    seen = set()

    def evaluated_only(scores: pd.DataFrame) -> pd.DataFrame:
        seen.update(excluded.intersection(scores.sample_id))
        return scores[~scores.sample_id.isin(excluded)].reset_index(drop=True)

    files, missing = score_files(run_dir, repo, allow_missing)
    group_cols, arm_cols = [f"p_{t}" for t in TIERS], [f"arm_success_v2_{t}" for t in TIERS]
    complete_dev = dev[dev[group_cols].notna().all(axis=1)]

    prices, groups_sha = group_prices(run_dir)
    length, length_info = length_table(run_dir)
    rho, targets, price_targets, length_targets, normalizer = {}, {}, {}, {}, {}
    for source in SOURCES:
        rows = complete_dev[complete_dev.source_dataset.eq(source)]
        P = rows[group_cols].to_numpy(float)
        rho[source], targets[source] = estimate_rho(P), cost_targets(P)
        price_targets[source] = cost_targets(P, prices[source])
        normalizer[source] = float(length.reindex(rows.sample_id).mean())
        length_targets[source] = cost_targets(P, length_costs(length, rows.sample_id, normalizer[source]))
    blended = data["blended"]
    label_tables = {"r0_group": {s: class_table(blended.loc[blended.source_dataset.eq(s), "difficulty_label"],
                                                blended.loc[blended.source_dataset.eq(s), [f"success_{t}" for t in TIERS]].to_numpy(float),
                                                ("easy", "medium", "hard")) for s in SOURCES}}

    output_dir.mkdir(parents=True)
    (output_dir / "calibrated").mkdir()
    selection_rows, methods = [], {}
    by_method = {}
    for (method, setting), path in files.items():
        if method == "rt08":
            continue
        by_method.setdefault(method, []).append((setting, path))
    for method, settings in sorted(by_method.items()):
        evaluated = []
        for setting, path in settings:
            scores = evaluated_only(pd.read_parquet(path))
            oof, frozen = calibrate_method(method, scores, dev, group_cols, label_tables.get(method, {}), SOURCES)
            per_source = {}
            for source, (d, X, Y, Phat) in oof.items():
                table = lambda_table(Y, Phat, rho[source], targets[source])
                per_source[source] = table
                for row in table["escalation"]:
                    selection_rows.append({"method": method, "setting": setting, "source": source, **row})
            score = mean_gain(per_source)
            evaluated.append((score, setting, path, frozen, per_source, oof))
            print(f"[lock] {method} {setting}: mean dev gain {score:+.4f}", flush=True)
        score, setting, path, frozen, per_source, oof = max(evaluated, key=lambda e: e[0])
        # sensitivity: the selected setting priced with the group mean data prices (selection stays on 1:3:5)
        priced = {s: lambda_table(Y, Phat, rho[s], price_targets[s], prices[s]) for s, (d, X, Y, Phat) in oof.items()}
        lengthy = {s: lambda_table(Y, Phat, rho[s], length_targets[s], length_costs(length, d.sample_id, normalizer[s]))
                   for s, (d, X, Y, Phat) in oof.items()}
        out_path = output_dir / "calibrated" / f"{method}.parquet"
        frozen.to_parquet(out_path, index=False)
        methods[method] = {
            "selected_setting": setting, "kind": KIND[method], "dev_mean_gain": score,
            "settings_tried": len(evaluated), "scores_file": str(path.relative_to(run_dir)),
            "scores_sha256": C.sha256_file(path), "calibrated_file": f"calibrated/{method}.parquet",
            "calibrated_sha256": C.sha256_file(out_path),
            "lambda": lambda_record(per_source),
            "sensitivity_group_price": {"dev_mean_gain": mean_gain(priced), "lambda": lambda_record(priced)},
            "sensitivity_length": {"dev_mean_gain": mean_gain(lengthy), "lambda": lambda_record(lengthy)}}

    # RT-08: retrained old-definition R0 (v2 labels) against the operational R0 (v1), A split, two bases
    rt08 = {}
    if ("rt08", "default") in files:
        a_train = data["train"][data["train"].source_dataset.isin(A_SOURCES)]
        candidates = {"rt08": (evaluated_only(pd.read_parquet(files[("rt08", "default")])), "old_label_v2"),
                      "r0_operational": (evaluated_only(operational_r0(repo)), "old_label_v1")}
        for basis, cols in (("group", group_cols), ("representative", arm_cols)):
            basis_dev = dev[dev[cols].notna().all(axis=1)]
            b_rho, b_targets = {}, {}
            for source in A_SOURCES:
                P = basis_dev.loc[basis_dev.source_dataset.eq(source), cols].to_numpy(float)
                b_rho[source], b_targets[source] = estimate_rho(P), cost_targets(P)
            rt08[basis] = {"rho": b_rho, "cost_targets": b_targets, "models": {}}
            for name, (scores, label_col) in candidates.items():
                tables = {}
                for s in A_SOURCES:
                    tr = a_train[a_train.source_dataset.eq(s)]
                    tr = tr[tr[cols].notna().all(axis=1)]
                    tables[s] = class_table(tr[label_col], tr[cols].to_numpy(float), ("easy", "medium", "hard"))
                oof, frozen = calibrate_method(name, scores, dev, cols, tables, A_SOURCES)
                out_path = output_dir / "calibrated" / f"{name}-{basis}.parquet"
                frozen.to_parquet(out_path, index=False)
                lam = {s: lambda_table(Y, Phat, b_rho[s], b_targets[s]) for s, (d, X, Y, Phat) in oof.items()}
                rt08[basis]["models"][name] = {
                    "label_column": label_col, "calibrated_file": f"calibrated/{name}-{basis}.parquet",
                    "calibrated_sha256": C.sha256_file(out_path),
                    "dev_mean_gain": mean_gain(lam), "lambda": lambda_record(lam)}
        op = repo / OPERATIONAL_R0
        rt08["inputs"] = {"rt08_scores_sha256": C.sha256_file(files[("rt08", "default")]),
                          "operational_r0_scores": {p.name: C.sha256_file(p) for p in sorted(op.glob("*-512.parquet"))}}

    if by_method and seen != excluded:  # every listed id must be a scored question of this run
        raise SystemExit(f"{len(excluded - seen)} excluded ids are not in the scores")
    table = pd.DataFrame(selection_rows)
    table.to_parquet(output_dir / "selection-table.parquet", index=False)
    lock_record = {
        "experiment": "EV-10 07b_lock", "test_outcomes_read": False, "missing_scores": missing,
        "inputs": data["hashes"] | {"dev_rows": complete_dev.groupby("source_dataset").size().to_dict()},
        "exclusions": exclusions,
        "evaluation": {
            "tiers": list(TIERS), "service_cost": SERVICE_COST.tolist(), "kappa_grid": KAPPAS.tolist(),
            "cost_levels": {"fractions": list(LEVELS), "definition": "fraction of each source's range between the cheapest "
                            "and the dearest fixed start tier (expected cost with escalation, dev)",
                            "targets": targets},
            "sensitivity": {"group_price": {
                "definition": "per source, mean data price of each tier's group models (00_prepare train means; SPROUT "
                              "price table), high = 1; same adapters and selected settings, own cost targets and lambdas",
                "tier_cost": {s: prices[s].tolist() for s in SOURCES}, "targets": price_targets,
                "groups_sha256": groups_sha},
                "length": {
                "definition": "per question and tier: tier price (1:3:5) x L / mean dev L of the source, L = input "
                              "tokens + 5 x median output tokens over the source's data models (plan 2.3); same adapters "
                              "and selected settings, own cost targets and lambdas",
                "output_weight": OUTPUT_WEIGHT, "normalizer_mean_dev_length": normalizer, "targets": length_targets,
                "lengths_sha256": length_info["sha256"], "output_length_filled_with_dev_median": length_info["filled"]}},
            "rho": rho, "tau": TAU, "tau_curve": list(TAU_CURVE), "rules": list(RULES) + ["tau"],
            "adapter": "per source and tier isotonic regression on dev (increasing='auto'); class outputs are first "
                       "multiplied by the train table P(tier succeeds | class); single scores feed all three tiers",
            "selection": f"{FOLDS}-fold cross-fitted adapters on dev; lambda per (source, rule, kappa, level) closest to the "
                         "target cost; setting score = mean over sources, kappa grid and levels of the escalation-aware "
                         "rule's value minus the fixed-start envelope at the same dev cost",
            "comparison": "08_test: router value (resolved - kappa x failed attempts) against the best fixed start tier "
                          "or mixture at the router's test cost; kappa* = smallest kappa with gain > .01 and a paired "
                          "bootstrap 95% interval above 0",
            "lambda_grid": [float(LAMBDAS[0]), float(LAMBDAS[1]), float(LAMBDAS[-1]), len(LAMBDAS)],
        },
        "methods": methods, "rt08": rt08,
        "selection_table_sha256": C.sha256_file(output_dir / "selection-table.parquet"),
    }
    (output_dir / "selection-lock.json").write_text(json.dumps(lock_record, ensure_ascii=False, indent=1, default=float))
    (output_dir / "lock-summary.md").write_text(summary_markdown(lock_record, table))
    return lock_record


def summary_markdown(record: dict, table: pd.DataFrame) -> str:
    """Short human-readable view of the lock for the stop-and-report step."""
    lines = ["# 07 lock 요약", "", "dev에서 정한 내용이다. test 결과는 읽지 않았다.", ""]
    if record["missing_scores"]:
        lines += [f"**빠진 점수:** {', '.join(record['missing_scores'])}", ""]
    ev = record["evaluation"]
    label = {"non_english": "영어가 아닌 문항", "document_grounded": "문서 제공형(RAG형) 문항"}
    for name, en in record["exclusions"].items():
        lines += [f"**평가 제외 — {label.get(name, name)}:** {en['rows']}건(dev {en['dev_rows_removed']}건, 나머지 test). "
                  f"목록: `{en['file']}`", ""]
    gp, ln = ev["sensitivity"]["group_price"], ev["sensitivity"]["length"]
    lines += ["## 원천별 값", "", "비용 목표는 원천마다 가장 싼·가장 비싼 고정 시작 단계의 기대 비용 사이 25/50/75% 지점이다. "
              "주 기준은 Claude 가격 1:3:5(.2/.6/1.0), 민감도는 집단 모델의 평균 데이터 가격(high = 1)이다.", "",
              "| 원천 | dev 행 | ρ | 비용 목표(1:3:5) | 집단 평균 가격 low/medium/high | 비용 목표(집단 가격) | 비용 목표(길이 반영) |",
              "|---|---:|---:|---|---|---|---|"]
    for s in SOURCES:
        lines.append(f"| {s} | {record['inputs']['dev_rows'].get(s, 0)} | {ev['rho'][s]:.3f} | "
                     + " / ".join(f"{c:.3f}" for c in ev["cost_levels"]["targets"][s]) + " | "
                     + " / ".join(f"{c:.3f}" for c in gp["tier_cost"][s]) + " | "
                     + " / ".join(f"{c:.3f}" for c in gp["targets"][s]) + " | "
                     + " / ".join(f"{c:.3f}" for c in ln["targets"][s]) + " |")
    sel = {m: v["selected_setting"] for m, v in record["methods"].items()}
    chosen = table[table.apply(lambda r: sel.get(r["method"]) == r["setting"], axis=1)]
    gains = chosen.groupby(["method", "source"]).dev_gain.mean().unstack()
    lines += ["", "## 방법별 선택 설정과 dev 이득", "",
              "이득 = 상향 고려 규칙의 (최종 해결 − κ × 실패 횟수)에서 같은 dev 비용의 고정 시작 단계(혼합 포함) 값을 뺀 값. "
              "κ 격자와 비용 수준 평균. 설정은 1:3:5 기준으로 고르고, 마지막 두 열은 같은 설정을 집단 평균 가격과 길이 반영 비용으로 "
              "계산한 값이다.", "",
              "| 방법 | 선택 설정 | 시도 | " + " | ".join(SOURCES) + " | 평균 | 집단 가격 평균 | 길이 반영 평균 |",
              "|---|---|---:|" + "---:|" * (len(SOURCES) + 3)]
    for m, v in sorted(record["methods"].items(), key=lambda kv: -kv[1]["dev_mean_gain"]):
        cells = " | ".join(f"{gains.loc[m, s]:+.3f}" if s in gains.columns else "–" for s in SOURCES)
        lines.append(f"| {m} | `{v['selected_setting']}` | {v['settings_tried']} | {cells} | {v['dev_mean_gain']:+.3f} | "
                     f"{v['sensitivity_group_price']['dev_mean_gain']:+.3f} | {v['sensitivity_length']['dev_mean_gain']:+.3f} |")
    if record.get("rt08"):
        lines += ["", "## RT-08 (A 분할, dev 평균 이득)", "", "| 기준 | RT-08(v2 재학습) | 운영 R0(v1) |", "|---|---:|---:|"]
        for basis in ("group", "representative"):
            models = record["rt08"][basis]["models"]
            lines.append(f"| {basis} | {models['rt08']['dev_mean_gain']:+.3f} | {models['r0_operational']['dev_mean_gain']:+.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True, help="checkout holding the operational R0 scores")
    p.add_argument("--output-dir", type=Path, required=True, help="normally <run>/lock")
    p.add_argument("--allow-missing", action="store_true", help="dry run with the scores available so far")
    a = p.parse_args(argv)
    out = lock(a.run_dir, a.repo, a.output_dir, a.allow_missing)
    print(json.dumps({"methods": {m: {k: v[k] for k in ("selected_setting", "dev_mean_gain", "settings_tried")}
                                  for m, v in out["methods"].items()},
                      "rho": out["evaluation"]["rho"], "cost_targets": out["evaluation"]["cost_levels"]["targets"],
                      "missing": out["missing_scores"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
