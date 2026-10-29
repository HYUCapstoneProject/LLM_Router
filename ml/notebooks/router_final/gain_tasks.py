"""EV-10 11b_gain_tasks: first-attempt gain G per shared task type (descriptive, like 08's U task-type table).

Uses the gain lock exactly as 11_gain's test does: same calibrated scores, same decisions (tau = .8 and the one-shot
rule's three cost levels), main cost basis (1:3:5). Nothing is selected. Like 08's `by_category`, the fixed-start
envelope H(C) is rebuilt inside each (source, task type) cell, cells with fewer than MINIMUM questions are left
out, and the pooled value over sources is the row-weighted mean of the per-source cells. Intervals: question-group
bootstrap inside each cell (all routers share the same draws), pooled draw = row-weighted mean of the cell draws.

Writes <run>/gain-test-tasks (must not exist): by-task.parquet, by-task-pairs.parquet, manifest.json,
by-task-summary.md. R1-BGE stays out as in 11_gain.
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
import gain_eval as G  # noqa: E402
import lock as L  # noqa: E402
import test_eval as T  # noqa: E402

TIERS, SOURCES, TAU = L.TIERS, L.SOURCES, L.TAU
MINIMUM = 30  # same as 08's by_category
STREAM = 1000  # bootstrap streams 1000 + 10 x source + task, apart from 11_gain's per-source streams
PAIRS = (("r1_group", "r0_group"), ("r1_group", "knn"), ("r1_group", "graphrouter"), ("r0_group", "knn"),
         ("graphrouter", "knn"), ("r1_group", "routellm_released"))


def cell_rows(P: np.ndarray, sets: dict, draws: T.Draws) -> tuple:
    """G per (method, rule, param) in one cell, plus its bootstrap draws."""
    n = len(P)
    cmat = G.tier_cost(L.SERVICE_COST, n)
    cols, index, rows = [P[:, t] for t in range(3)] + [cmat[:, t] for t in range(3)], {}, []
    for name, ss in sets.items():
        for rule, param, st, _ in ss:
            if rule == "tau" and abs(param - TAU) > 1e-9:
                continue
            rows.append({"method": name, "rule": rule, "param": param, **G.gain_point(P, st)})
            index[(name, rule, param)] = (len(cols), len(cols) + 1)
            cols += [P[np.arange(n), st], cmat[np.arange(n), st]]
    B = draws.means(np.column_stack(cols))
    kept = {k: G.boot_gain(B, a, c, [0, 1, 2], [3, 4, 5]).astype(np.float32) for k, (a, c) in index.items()}
    return rows, kept


def evaluate(run_dir: Path, repo: Path, data_root: Path, output_dir: Path, rehearse: bool, iterations: int) -> dict:
    if output_dir.exists():
        raise SystemExit(f"{output_dir} exists (move it away only to redo a failed run)")
    t0 = time.time()
    glock = G.load_gain_lock(run_dir, repo)
    rec = glock["record"]
    truth = T.load_truth(run_dir, repo, data_root, glock, rehearse)
    frame, role = truth["frame"], truth["role"]
    print(f"[tasks] {role} rows evaluated: {truth['counts']['evaluated']}", flush=True)
    methods = sorted(m for m in rec["methods"] if m not in G.OMITTED)
    cal = {m: T.calibrated(glock["files"][m], frame, role) for m in methods}
    output_dir.mkdir(parents=True)

    cells, draws_kept = [], {}
    for si, source in enumerate(SOURCES):
        rows = frame[frame.source_dataset.eq(source)].reset_index(drop=True)
        ids = rows.sample_id.to_numpy()
        P = rows[[f"p_{t}" for t in TIERS]].to_numpy(float)
        needed = rows.needed.to_numpy()
        sets = {}
        for name in methods + ["needed_tier"]:
            Phat = None if name == "needed_tier" else T.phat_for(cal[name], ids)
            lam = rec["methods"][name]["lambda"][source] if name in rec["methods"] else None
            sets[name] = G.decision_sets(name, Phat, needed, lam)
        for ti, task in enumerate(K.TYPES):
            idx = np.flatnonzero(rows.task_type.eq(task).to_numpy())
            if len(idx) < MINIMUM:
                if len(idx):
                    cells.append({"source": source, "task_type": task, "rows": len(idx), "method": None})
                continue
            sub = {name: [(r, p, st[idx], b) for r, p, st, b in ss] for name, ss in sets.items()}
            draws = T.Draws(rows.group_id.iloc[idx], iterations, T.BOOT["seed"], STREAM + 10 * si + ti)
            got, kept = cell_rows(P[idx], sub, draws)
            for r in got:
                key = (r["method"], r["rule"], r["param"])
                lo, hi = T.interval(kept[key])
                cells.append({"source": source, "task_type": task, "rows": len(idx), **r,
                              "G_lo": float(lo), "G_hi": float(hi)})
                draws_kept[(source, task) + key] = kept[key]
        print(f"[tasks] {source} done ({time.time() - t0:.0f}s)", flush=True)

    out = pd.DataFrame(cells)
    pooled, pairs = pool(out, draws_kept)
    table = pd.concat([out, pooled], ignore_index=True)
    table.to_parquet(output_dir / "by-task.parquet", index=False)
    pairs.to_parquet(output_dir / "by-task-pairs.parquet", index=False)
    manifest = {"experiment": "EV-10 11b_gain_tasks", "rehearsal_on_dev": rehearse, "role_evaluated": role,
                "gain_lock_sha256": glock["sha256"], "outcomes_sha256": truth["test_sha256"],
                "counts": truth["counts"], "minimum_rows": MINIMUM,
                "bootstrap": T.BOOT | {"iterations": iterations, "streams": "1000 + 10 x source + task"},
                "methods": methods, "omitted": [m for m in G.OMITTED if m in rec["methods"]],
                "outputs": {p.name: C.sha256_file(p) for p in sorted(output_dir.glob("*.parquet"))},
                "seconds": round(time.time() - t0, 1)}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=float))
    (output_dir / "by-task-summary.md").write_text(summary_markdown(manifest, table, pairs))
    return manifest


def pool(cells: pd.DataFrame, kept: dict) -> tuple:
    """Row-weighted mean over the sources that have the cell (08's pooling), with pooled draws and router pairs."""
    done = cells[cells.method.notna()]
    rows, pair_rows = [], []
    for (task, rule, param), part in done.groupby(["task_type", "rule", "param"]):
        pooled_draws, pooled_g = {}, {}
        for method, m in part.groupby("method"):
            w = m.rows.to_numpy(float) / m.rows.sum()
            d = sum(wi * kept[(s, task, method, rule, param)] for wi, s in zip(w, m.source))
            num = {k: float(np.dot(w, m[k])) for k in ("A", "C", "H", "G", "start_low", "start_medium", "start_high")}
            pooled_draws[method], pooled_g[method] = d, num["G"]
            lo, hi = T.interval(d)
            rows.append({"source": "all", "task_type": task, "rows": int(m.rows.sum()), "method": method,
                         "rule": rule, "param": param, **num, "G_lo": float(lo), "G_hi": float(hi),
                         "sources": ",".join(m.source)})
        for a, b in PAIRS:
            if a in pooled_draws and b in pooled_draws:
                lo, hi = T.interval(pooled_draws[a] - pooled_draws[b])
                pair_rows.append({"task_type": task, "rule": rule, "param": param, "a": a, "b": b,
                                  "diff": pooled_g[a] - pooled_g[b], "lo": float(lo), "hi": float(hi)})
    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


def summary_markdown(manifest: dict, table: pd.DataFrame, pairs: pd.DataFrame) -> str:
    head = "11b 과제 유형별 G (dev 리허설 — test가 아니다)" if manifest["rehearsal_on_dev"] else "11b 과제 유형별 G"
    tau = table[table.source.eq("all") & table.rule.eq("tau") & np.isclose(table.param, TAU)]
    types = [t for t in K.TYPES if t in set(tau.task_type)]
    order = tau[tau.method.ne("needed_tier")].groupby("method").G.mean().sort_values(ascending=False).index.tolist()
    order += ["needed_tier"]
    n = tau.drop_duplicates("task_type").set_index("task_type").rows
    lines = [f"# {head}", "", "τ = .8 규칙, 기본 가격(1:3:5). 원천·과제 칸마다 고정 시작 기준선을 다시 만들고, "
             f"{MINIMUM}문항 미만 칸은 뺀 뒤 행 수로 가중 평균했다. 기술 통계(판정하지 않음).", "",
             "## G 점추정", "", "| 라우터 | " + " | ".join(f"{t} ({n[t]:,})" for t in types) + " |",
             "|---|" + "---:|" * len(types)]
    wide = tau.pivot_table(index="method", columns="task_type", values="G")
    for m in order:
        if m in wide.index:
            lines.append(f"| {m} | " + " | ".join(f"{wide.at[m, t]:+.3f}" for t in types) + " |")
    lines += ["", "## 95% 구간", "", "| 라우터 | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    ci = tau.set_index(["method", "task_type"])
    for m in order:
        if m in wide.index:
            lines.append(f"| {m} | " + " | ".join(
                f"[{ci.at[(m, t), 'G_lo']:+.3f}, {ci.at[(m, t), 'G_hi']:+.3f}]" if (m, t) in ci.index else "–"
                for t in types) + " |")
    pt = pairs[pairs.rule.eq("tau") & np.isclose(pairs.param, TAU)]
    if len(pt):
        lines += ["", "## 라우터 쌍 차이 (τ = .8)", "", "| 쌍 | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
        for (a, b), part in pt.groupby(["a", "b"], sort=False):
            p = part.set_index("task_type")
            lines.append(f"| {a} − {b} | " + " | ".join(
                f"{p.at[t, 'diff']:+.3f} [{p.at[t, 'lo']:+.3f}, {p.at[t, 'hi']:+.3f}]" if t in p.index else "–"
                for t in types) + " |")
    lines += ["", f"평가 행: {manifest['counts']['evaluated']}, bootstrap {manifest['bootstrap']['iterations']}회, "
              f"{manifest['seconds']}초", ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True, help="normally <run>/gain-test-tasks")
    p.add_argument("--rehearse-on-dev", action="store_true")
    p.add_argument("--iterations", type=int, default=T.BOOT["iterations"])
    args = p.parse_args(argv)
    out = evaluate(args.run_dir, args.repo, args.data_root, args.output_dir, args.rehearse_on_dev, args.iterations)
    print(json.dumps({k: out[k] for k in ("role_evaluated", "counts", "seconds")}, indent=1, default=float))


if __name__ == "__main__":
    main()
