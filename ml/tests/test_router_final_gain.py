import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/router_final"))
import common_track as C  # noqa: E402
import gain_eval as G  # noqa: E402
import gain_tasks as GT  # noqa: E402
import lock as L  # noqa: E402
import test_eval as T  # noqa: E402


class GainTest(unittest.TestCase):
    def test_fixed_start_on_the_baseline_has_zero_gain_and_below_it_negative(self):
        rng = np.random.default_rng(0)
        P = rng.uniform(size=(50, 3))
        P = P / P.mean(axis=0) * np.array([0.5, 0.75, 0.85])  # concave: every fixed tier on the baseline
        for t in range(3):
            self.assertAlmostEqual(G.gain_point(P, np.full(50, t))["G"], 0.0, places=12)
        P = P / P.mean(axis=0) * np.array([0.5, 0.6, 0.9])  # medium below the low-high chord
        self.assertAlmostEqual(G.gain_point(P, np.full(50, 1))["G"], 0.6 - 0.7, places=12)

    def test_gain_is_success_over_the_mixture_at_the_same_cost(self):
        # fixed first-attempt success .5 / .6 / .9 at .2 / .6 / 1.0: medium is below the low-high chord (.7)
        P = np.array([[1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 0, 1], [0.5, 1, 0.5]], float)
        P = P / P.mean(axis=0) * np.array([0.5, 0.6, 0.9])
        starts = np.array([0, 1, 2, 2, 0])
        res = G.gain_point(P, starts)
        self.assertAlmostEqual(res["C"], (0.2 + 0.6 + 1.0 + 1.0 + 0.2) / 5)
        self.assertAlmostEqual(res["H"], 0.5 + (0.9 - 0.5) * (res["C"] - 0.2) / 0.8)  # chord, medium skipped
        self.assertAlmostEqual(res["G"], res["A"] - res["H"])
        self.assertAlmostEqual(res["start_high"], 0.4)

    def test_dearer_fixed_tier_that_does_worse_does_not_lower_the_baseline(self):
        P = np.tile([0.6, 0.8, 0.7], (10, 1))  # high worse than medium
        res = G.gain_point(P, np.full(10, 2))
        self.assertAlmostEqual(res["H"], 0.8)  # medium is affordable at cost 1.0
        self.assertAlmostEqual(res["G"], -0.1)

    def test_oracle_picks_cheapest_success_and_gains(self):
        P = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]], float)
        oracle = np.array([0, 1, 2, 0])
        self.assertGreater(G.gain_point(P, oracle)["G"], 0.3)

    def test_per_question_costs(self):
        P = np.tile([0.5, 0.7, 0.9], (4, 1))
        cost = np.array([[0.1, 0.3, 0.5], [0.3, 0.9, 1.5]] * 2)
        res = G.gain_point(P, np.array([0, 2, 0, 2]), cost)
        self.assertAlmostEqual(res["C"], (0.1 + 1.5 + 0.1 + 1.5) / 4)

    def test_targets_are_quarters_of_the_first_attempt_range(self):
        P = np.random.default_rng(1).uniform(size=(20, 3))
        np.testing.assert_allclose(G.first_cost_targets(P), [0.4, 0.6, 0.8])

    def test_lambda_hits_cheaper_targets_with_larger_lambda(self):
        rng = np.random.default_rng(2)
        P = np.sort(rng.uniform(size=(400, 3)), axis=1)
        rows = G.lambda_first(P, P, G.first_cost_targets(P))
        lams = [r["lambda"] for r in rows]
        self.assertTrue(lams[0] >= lams[1] >= lams[2])
        for r in rows:
            self.assertLess(abs(r["dev_C"] - r["target_cost"]), 0.05)

    def test_boot_gain_matches_point_on_identity_draws(self):
        rng = np.random.default_rng(3)
        P = rng.uniform(size=(30, 3))
        st = rng.integers(0, 3, 30)
        c = np.broadcast_to(L.SERVICE_COST, P.shape)
        X = np.column_stack([P, c, P[np.arange(30), st], c[np.arange(30), st]])
        B = X.mean(axis=0)[None]
        self.assertAlmostEqual(float(G.boot_gain(B, 6, 7, [0, 1, 2], [3, 4, 5])[0]), G.gain_point(P, st)["G"], places=6)


def synthetic_rows(rng, role: str, n: int = 90) -> pd.DataFrame:
    parts = []
    for source in L.SOURCES:
        difficulty = rng.uniform(size=n)
        P = np.clip(np.column_stack([1.2 - 1.6 * difficulty, 1.4 - 1.2 * difficulty, 1.6 - difficulty])
                    + rng.normal(0, 0.1, (n, 3)), 0, 1)
        needed = np.where((P >= 0.8).any(axis=1), (P >= 0.8).argmax(axis=1), P.argmax(axis=1))
        parts.append(pd.DataFrame({
            "sample_id": [f"{source}:{role}{i}" for i in range(n)], "source_dataset": source, "excluded_v2": False,
            "needed_tier": np.array(L.TIERS)[needed], "difficulty": difficulty,
            **{f"p_{t}": P[:, k] for k, t in enumerate(L.TIERS)},
            **{f"arm_success_v2_{t}": P[:, k] for k, t in enumerate(L.TIERS)}}))
    return pd.concat(parts, ignore_index=True)


def write_scores(path: Path, frames: dict, noise: float, rng) -> Path:
    parts = []
    for role, f in frames.items():
        s = f[[f"p_{t}" for t in L.TIERS]].to_numpy() + rng.normal(0, noise, (len(f), 3))
        parts.append(pd.DataFrame({"sample_id": f.sample_id, "source_dataset": f.source_dataset, "role": role,
                                   **{f"score_{t}": s[:, k] for k, t in enumerate(L.TIERS)}}))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
    return path


def write_calibrated(path: Path, frames: dict, sources) -> str:
    parts = []
    for role, f in frames.items():
        f = f[f.source_dataset.isin(sources)]
        parts.append(f[["sample_id", "source_dataset"]].assign(
            role=role, **{f"p_hat_{t}": f[f"p_{t}"].clip(0, 1).to_numpy() for t in L.TIERS}))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
    return C.sha256_file(path)


class PipelineTest(unittest.TestCase):
    """lock -> test on a synthetic run (external inputs mocked): the whole path runs and stays consistent."""

    def test_lock_then_test(self):
        rng = np.random.default_rng(7)
        dev, test = synthetic_rows(rng, "dev"), synthetic_rows(rng, "test")
        with tempfile.TemporaryDirectory() as tmp:
            run, repo = Path(tmp) / "run", Path(tmp) / "repo"
            excl = repo / "excluded.csv"
            excl.parent.mkdir(parents=True)
            pd.DataFrame({"sample_id": [], "source_dataset": [], "role": []}).to_csv(excl, index=False)
            exclusions = {"non_english": {"file": "excluded.csv", "sha256": C.sha256_file(excl), "rows": 0}}
            frames = {"dev": dev, "test": test}
            files = {("knn", "good"): write_scores(run / "s/knn_good.parquet", frames, 0.05, rng),
                     ("knn", "noisy"): write_scores(run / "s/knn_noisy.parquet", frames, 0.6, rng),
                     ("r1_group", "default"): write_scores(run / "s/r1.parquet", frames, 0.1, rng),
                     ("r1_group_a", "default"): write_scores(run / "s/r1a.parquet", frames, 0.2, rng),
                     ("r1_bge", "default"): write_scores(run / "s/r1_bge.parquet", frames, 0.1, rng)}
            refs = {name: {"calibrated_file": f"calibrated/{name}-group.parquet",
                           "calibrated_sha256": write_calibrated(run / "lock" / "calibrated" / f"{name}-group.parquet",
                                                                 frames, L.A_SOURCES)}
                    for name in ("r0_operational", "rt08")}
            old = {"methods": {"knn": {"selected_setting": "noisy"}, "r1_group": {"selected_setting": "x"},
                               "r1_group_a": {"selected_setting": "x"},
                               "r1_bge": {"selected_setting": "x"}},
                   "exclusions": exclusions, "rt08": {"group": {"models": refs}},
                   "evaluation": {"sensitivity": {"length": {"lengths_sha256": "L",
                                                             "normalizer_mean_dev_length": {s: 100.0 for s in L.SOURCES}}}}}
            (run / "lock" / "selection-lock.json").write_text(json.dumps(old))
            seeds = {f"{b}_s{s}": {"calibrated_sha256": write_calibrated(run / "seeds" / "calibrated" / f"{b}_s{s}.parquet",
                                                                         frames, L.SOURCES)}
                     for b in G.PAIR for s in G.SEEDS}
            (run / "seeds" / "seeds-lock.json").write_text(json.dumps({"models": seeds}))
            blended = dev.assign(difficulty_label="easy", **{f"success_{t}": dev[f"p_{t}"] for t in L.TIERS})
            outcomes = {"dev": dev, "train": dev, "blended": blended, "hashes": {}}
            ids = pd.concat([dev, test]).sample_id
            length = pd.Series(rng.uniform(50, 150, len(ids)), index=ids.to_numpy())
            frame = test.assign(needed=test.needed_tier.map({t: k for k, t in enumerate(L.TIERS)}),
                                group_id=test.source_dataset + ":" + (test.index // 2).astype(str))
            pos = frame.groupby("source_dataset").cumcount()
            frame["task_type"] = np.where(pos < 40, "수학", np.where(pos < 80, "코드", "독해"))  # 독해: 10 rows, left out
            best = frame[[f"p_{t}" for t in L.TIERS]].max(axis=1)
            frame["stratum"] = np.where(best >= 0.8, "solvable", np.where(best > 0, "beyond_partial", "beyond_zero"))
            truth = {"frame": frame, "counts": {"evaluated": frame.groupby("source_dataset").size().to_dict()},
                     "role": "test", "test_sha256": "t", "input_sha256": {}}
            with mock.patch.object(L, "load_outcomes", return_value=outcomes), \
                    mock.patch.object(L, "evaluation_exclusions", return_value=(set(), exclusions)), \
                    mock.patch.object(L, "score_files", return_value=(files, [])), \
                    mock.patch.object(L, "group_prices", return_value=({s: np.array([0.05, 0.2, 1.0]) for s in L.SOURCES}, "g")), \
                    mock.patch.object(L, "length_table", return_value=(length, {"sha256": "L"})), \
                    mock.patch.object(T, "load_truth", return_value=truth):
                rec = G.lock(run, repo, run / "gain-lock", allow_missing=False)
                self.assertEqual(rec["methods"]["knn"]["selected_setting"], "good")
                self.assertTrue(rec["methods"]["knn"]["changed_from_07b"])
                self.assertIn("r1_bge", rec["methods"])  # locked, but left out of the results
                for rows in rec["methods"]["knn"]["lambda"].values():
                    self.assertEqual([r["level"] for r in rows], list(L.LEVELS))
                man = G.evaluate(run, repo, repo, run / "gain-test", rehearse=False, iterations=200)
                with self.assertRaises(SystemExit):  # opened once
                    G.evaluate(run, repo, repo, run / "gain-test", rehearse=False, iterations=200)
                tman = GT.evaluate(run, repo, repo, run / "gain-test-tasks", rehearse=False, iterations=200)
            tasks = pd.read_parquet(run / "gain-test-tasks" / "by-task.parquet")
            done = tasks[tasks.method.notna()]
            self.assertEqual(set(done.task_type), {"수학", "코드"})
            self.assertEqual(set(tasks[tasks.method.isna()].task_type), {"독해"})
            self.assertNotIn("r1_bge", set(done.method))
            self.assertTrue((done.G_lo <= done.G + 1e-6).all() and (done.G <= done.G_hi + 1e-6).all())
            pooled = done[done.source.eq("all")].set_index(["task_type", "method", "rule", "param"])
            per = done[done.source.ne("all")]
            for key, part in per.groupby(["task_type", "method", "rule", "param"]):
                self.assertAlmostEqual(pooled.loc[key, "G"], np.average(part.G, weights=part.rows), places=9)
            self.assertEqual(pooled.loc[("수학", "knn", "tau", L.TAU), "rows"], 40 * len(L.SOURCES))
            self.assertTrue(len(pd.read_parquet(run / "gain-test-tasks" / "by-task-pairs.parquet")))
            self.assertIn("| 수학 (", (run / "gain-test-tasks" / "by-task-summary.md").read_text())
            self.assertEqual(tman["omitted"], ["r1_bge"])
            out = run / "gain-test"
            macro = pd.read_parquet(out / "gain-macro.parquet")
            main = macro[macro.basis.eq("main") & macro.scope.eq("macro") & macro.rule.eq("tau")
                         & np.isclose(macro.param, L.TAU)].set_index("method")
            self.assertEqual(set(main.index), {"knn", "r1_group", "r1_group_a", "needed_tier"} | set(seeds))
            self.assertTrue(main.loc[list(seeds), "rank"].isna().all())  # seed models are not ranked
            h5 = pd.read_parquet(out / "seeds.parquet")
            self.assertEqual(set(h5.seed), {"42", "17", "73", "mean"})
            self.assertTrue((main.G_lo <= main.G).all() and (main.G <= main.G_hi).all())
            self.assertTrue(main.loc[["knn", "r1_group", "r1_group_a"], "rank"].notna().all())
            self.assertTrue(np.isnan(main.loc["needed_tier", "rank"]))
            a_split = macro[macro.scope.eq("a_split")]
            self.assertIn("r0_operational", set(a_split.method))
            self.assertEqual(set(macro.basis), {"main", "group_price", "length"})
            pairs = pd.read_parquet(out / "pairs.parquet")
            self.assertEqual(len(pairs[pairs.rule.eq("tau")]), 3)
            strata = pd.read_parquet(out / "by-stratum.parquet")
            self.assertTrue(set(strata.stratum) <= set(T.STRATA))
            self.assertTrue((out / "gain-summary.md").read_text().startswith("# 11 gain 결과 — test"))
            self.assertEqual(man["role_evaluated"], "test")
            self.assertEqual(man["omitted"], ["r1_bge"])
            self.assertNotIn("r1_bge", set(pd.read_parquet(out / "gain.parquet").method))


if __name__ == "__main__":
    unittest.main()
