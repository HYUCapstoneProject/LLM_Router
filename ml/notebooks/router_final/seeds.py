"""EV-10 09_seeds: H5 with seeds 17 and 73 added to the seed-42 run (plan 4.3: extra seeds only near a boundary).

Only H5 was near its boundary (R1-group minus R1-group-A, +.009 against +.01 on the external tests). kNN has no
seed, so only R1-group and R1-group-A are retrained (`ours.py train --seed`) and scored (`score_modernbert.py
--seed`). This step then treats each seed model exactly as 07b treated its method: the same adapter (per source and
tier isotonic, 5-fold cross-fitted on dev for the lambdas, full dev for the test), the locked rho and cost targets,
and its own lambda table. Nothing is re-selected. The test is evaluated as in 08 with the same bootstrap draws, and
H5 is judged again on the mean over the three seeds.

Writes <output>/ (normally <run>/seeds, which must not exist): calibrated/<model>.parquet, seeds-lock.json (dev
only), seeds-mean-gain.parquet, h5-seeds.json, seeds-summary.md, manifest.json.
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

SEEDS = (42, 17, 73)
PAIR = ("r1_group", "r1_group_a")  # mixed data, A split only


def seed_scores(run_dir: Path, seed: int, base: str) -> tuple:
    folder = run_dir / "scores-modernbert-seeds" / f"seed{seed}" / f"{base}_s{seed}"
    if not (folder / "_SUCCESS").exists():
        raise SystemExit(f"missing seed scores: {folder} (run 09_seeds training and scoring first)")
    path = folder / "scores.parquet"
    L.check(path, json.loads((folder / "metrics.json").read_text())["scores_sha256"])
    return pd.read_parquet(path), C.sha256_file(path)


def run(run_dir: Path, repo: Path, data_root: Path, output_dir: Path, rehearse: bool, iterations: int) -> dict:
    if output_dir.exists():
        raise SystemExit(f"{output_dir} exists")
    t0 = time.time()
    lock = T.load_lock(run_dir, repo)
    rec, ev = lock["record"], lock["record"]["evaluation"]
    # dev side: identical to 07b for these methods
    data = L.load_outcomes(run_dir)
    excluded, _ = L.evaluation_exclusions(repo)
    dev = data["dev"][~data["dev"].sample_id.isin(excluded)].reset_index(drop=True)
    group_cols = [f"p_{t}" for t in L.TIERS]
    output_dir.mkdir(parents=True)
    (output_dir / "calibrated").mkdir()
    models, seeds_lock = {}, {}
    for base in PAIR:
        models[(42, base)] = {"calibrated": T.calibrated(lock["files"][base], None, "dev" if rehearse else "test"),
                              "lambda": rec["methods"][base]["lambda"], "source": "07b lock"}
        for seed in SEEDS[1:]:
            scores, digest = seed_scores(run_dir, seed, base)
            scores = scores[~scores.sample_id.isin(excluded)].reset_index(drop=True)
            oof, frozen = L.calibrate_method(base, scores, dev, group_cols, {}, L.SOURCES)
            tables = {s: L.lambda_table(Y, Phat, ev["rho"][s], ev["cost_levels"]["targets"][s])
                      for s, (d, X, Y, Phat) in oof.items()}
            path = output_dir / "calibrated" / f"{base}_s{seed}.parquet"
            frozen.to_parquet(path, index=False)
            seeds_lock[f"{base}_s{seed}"] = {"scores_sha256": digest, "calibrated_sha256": C.sha256_file(path),
                                             "dev_mean_gain": L.mean_gain(tables), "lambda": L.lambda_record(tables)}
            role = "dev" if rehearse else "test"
            c = frozen[frozen.role.eq(role)].drop_duplicates("sample_id").set_index("sample_id")
            models[(seed, base)] = {"calibrated": c, "lambda": seeds_lock[f"{base}_s{seed}"]["lambda"], "source": "09"}
            print(f"[09] {base} seed {seed}: dev mean gain {seeds_lock[f'{base}_s{seed}']['dev_mean_gain']:+.4f}", flush=True)
    (output_dir / "seeds-lock.json").write_text(json.dumps(
        {"experiment": "EV-10 09_seeds lock (dev only)", "test_outcomes_read": False, "base_lock_sha256": lock["sha256"],
         "models": seeds_lock}, indent=1, default=float))
    # test side: the same evaluation and bootstrap streams as 08
    truth = T.load_truth(run_dir, repo, data_root, lock, rehearse)
    frame = truth["frame"]
    rows_out, draws = [], {}
    for si, source in enumerate(L.SOURCES):
        rows = frame[frame.source_dataset.eq(source)].reset_index(drop=True)
        P = rows[group_cols].to_numpy(float)
        d = T.Draws(rows.group_id, iterations, T.BOOT["seed"], si)
        for (seed, base), m in models.items():
            Phat = T.phat_for(m["calibrated"], rows.sample_id)
            sets = T.decision_sets(Phat, T.lambda_lookup(m["lambda"][source]), ev["rho"][source],
                                   ev["cost_levels"]["targets"][source], L.SERVICE_COST)
            records, kept, _ = T.evaluate_track1(P, sets, L.SERVICE_COST, d)
            mg = T.summarize_draws(kept)["mean_gain"]
            point = float(np.mean([r["gain"] for r in records if r["rule"] == "escalation"]))
            lo, hi = T.interval(mg)
            rows_out.append({"seed": seed, "method": base, "source": source, "mean_gain": point, "lo": lo, "hi": hi})
            draws[(seed, base, source)] = mg
        print(f"[09] {source} done ({time.time() - t0:.0f}s)", flush=True)
    table = pd.DataFrame(rows_out)
    table.to_parquet(output_dir / "seeds-mean-gain.parquet", index=False)
    h5 = judge_h5(table, draws)
    (output_dir / "h5-seeds.json").write_text(json.dumps(h5, ensure_ascii=False, indent=1, default=float))
    manifest = {"experiment": "EV-10 09_seeds", "rehearsal_on_dev": rehearse, "test_outcomes_read": not rehearse,
                "seeds": list(SEEDS), "methods": list(PAIR), "lock_sha256": lock["sha256"],
                "outcomes_sha256": truth["test_sha256"], "bootstrap": T.BOOT | {"iterations": iterations},
                "outputs": {p.name: C.sha256_file(p) for p in sorted(output_dir.glob("*.parquet"))},
                "seconds": round(time.time() - t0, 1)}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    (output_dir / "seeds-summary.md").write_text(summary(table, h5, rehearse))
    return manifest


def judge_h5(table: pd.DataFrame, draws: dict) -> dict:
    """H5 on the seed mean: mixed minus A-only mean gain > .01 on LLMRouterBench and SPROUT, >= -.01 on the A split."""
    per = {}
    for source in L.SOURCES:
        t = table[table.source.eq(source)].set_index(["seed", "method"]).mean_gain
        diff = float(np.mean([t[(s, PAIR[0])] for s in SEEDS]) - np.mean([t[(s, PAIR[1])] for s in SEEDS]))
        d = np.mean([draws[(s, PAIR[0], source)] for s in SEEDS], axis=0) - \
            np.mean([draws[(s, PAIR[1], source)] for s in SEEDS], axis=0)
        lo, hi = T.interval(d)
        per[source] = {"diff_seed_mean": diff, "lo": float(lo), "hi": float(hi),
                       "per_seed": {int(s): float(t[(s, PAIR[0])] - t[(s, PAIR[1])]) for s in SEEDS}}
    ext = all(per[s]["diff_seed_mean"] > T.MARGIN for s in T.EXTERNAL)
    a_ok = all(per[s]["diff_seed_mean"] >= -T.MARGIN for s in L.A_SOURCES)
    return {"rule": "mean over seeds 42/17/73: R1-group minus R1-group-A mean gain > .01 on LLMRouterBench and SPROUT, "
                    ">= -.01 on the A split (intervals: same group draws as 08, seed means taken inside each draw)",
            "per_source": per, "external_better": ext, "a_not_worse": a_ok, "supported": ext and a_ok}


def summary(table: pd.DataFrame, h5: dict, rehearse: bool) -> str:
    head = "09 seeds 요약 (dev 리허설 — test가 아니다)" if rehearse else "09 seeds 요약 (H5)"
    lines = [f"# {head}", "", "R1-집단(섞은 데이터) 대 R1-집단-A(A 분할만)의 평균 이득. seed 42는 07b lock, 17·73은 같은 방식으로 dev에서 보정했다.", "",
             "| 원천 | " + " | ".join(f"R1-집단 s{s}" for s in SEEDS) + " | " + " | ".join(f"R1-집단-A s{s}" for s in SEEDS)
             + " | 차이(seed 평균) [95%] |", "|---|" + "---:|" * (2 * len(SEEDS) + 1)]
    for source in L.SOURCES:
        t = table[table.source.eq(source)].set_index(["seed", "method"]).mean_gain
        p = h5["per_source"][source]
        lines.append(f"| {source} | " + " | ".join(f"{t[(s, PAIR[0])]:+.3f}" for s in SEEDS) + " | "
                     + " | ".join(f"{t[(s, PAIR[1])]:+.3f}" for s in SEEDS)
                     + f" | {p['diff_seed_mean']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}] |")
    lines += ["", f"- 외부 test에서 +.01 초과: {h5['external_better']}", f"- A test에서 −.01 이내: {h5['a_not_worse']}",
              f"- **H5 판정(seed 평균): {'지지' if h5['supported'] else '지지 안 됨'}**"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True, help="normally <run>/seeds")
    p.add_argument("--rehearse-on-dev", action="store_true")
    p.add_argument("--iterations", type=int, default=T.BOOT["iterations"])
    a = p.parse_args(argv)
    if not a.rehearse_on_dev and a.iterations != T.BOOT["iterations"]:
        raise SystemExit("the test run uses the pre-set bootstrap size")
    out = run(a.run_dir, a.repo, a.data_root or a.repo, a.output_dir, a.rehearse_on_dev, a.iterations)
    print(json.dumps({k: out[k] for k in ("seconds", "outputs")}, indent=1))


if __name__ == "__main__":
    main()
