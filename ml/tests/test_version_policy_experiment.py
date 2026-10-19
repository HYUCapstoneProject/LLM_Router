import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/version_policy"))
import compare  # noqa: E402
import plan as P  # noqa: E402

POLICY = {"cycle_size": 3, "history_replay_per_label": 32, "base_rehearsal_per_label": 32}


def frame(n, seed=0, sources=("a", "b")):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(n)],
        "group_id": [f"g{i // 2}" for i in range(n)],  # pairs of duplicates
        "source_dataset": [sources[(i // 2) % len(sources)] for i in range(n)],
        "difficulty_label": rng.choice(P.LABELS, size=n, p=[.55, .26, .19]),
    })


class SubsetTest(unittest.TestCase):
    def test_nested_and_group_complete(self):
        pool = frame(2000)
        small, large = P.nested_subset(pool, 300, 7), P.nested_subset(pool, 900, 7)
        self.assertTrue(set(small.sample_id) <= set(large.sample_id))
        for part in (small, large):
            sizes = pool[pool.group_id.isin(part.group_id)].groupby("group_id").size()
            self.assertTrue(part.groupby("group_id").size().sort_index().equals(sizes.sort_index()))
        self.assertAlmostEqual(len(large), 900, delta=4)
        self.assertAlmostEqual(large.source_dataset.eq("a").mean(), .5, delta=.01)

    def test_windows_split_groups_evenly(self):
        probe = frame(900)
        window = P.assign_windows(probe, 3, 1)
        self.assertEqual(sorted(window.unique()), [1, 2, 3])
        self.assertTrue((probe.assign(w=window).groupby("group_id").w.nunique() == 1).all())
        counts = window.value_counts()
        self.assertLessEqual(counts.max() - counts.min(), 4)


class BatchTest(unittest.TestCase):
    def test_targets_keep_every_wrong_row_and_restore_ratio(self):
        ratio = {"easy": .5, "medium": .3, "hard": .2}
        t = P.batch_targets({"easy": 10, "medium": 30, "hard": 20}, ratio)
        self.assertEqual(t, {"easy": 50, "medium": 30, "hard": 20})
        t = P.batch_targets({"hard": 5}, ratio)
        self.assertEqual(t, {"easy": 12, "medium": 8, "hard": 5})

    def test_batches_use_wrong_rows_and_never_reuse_reserve(self):
        pool = frame(3000)
        probe = pool.iloc[:600].assign(window=[1 + i % 3 for i in range(600)])
        reserve = pool.iloc[600:]
        rng = np.random.default_rng(3)
        pred = rng.integers(0, 3, size=600)
        ratio = P.label_ratio(pool)
        batches, report, left = P.build_batches(probe, pred, reserve, ratio, 3, "fill")
        wrong = probe[pred != P.label_ids(probe.difficulty_label)]
        self.assertEqual(set(batches[batches.origin.eq("wrong")].sample_id), set(wrong.sample_id))
        self.assertFalse(batches.sample_id.duplicated().any())
        self.assertTrue(set(batches[batches.origin.eq("fill")].sample_id) <= set(reserve.sample_id))
        for k in (1, 2, 3):
            mix = P.label_ratio(batches[batches["round"].eq(k)])
            for label in P.LABELS:
                self.assertAlmostEqual(mix[label], ratio[label], delta=.02)
        self.assertEqual(sum(left.values()) + int(batches.origin.eq("fill").sum()), len(reserve))

    def test_reserve_shortage_stops(self):
        pool = frame(400)
        probe = pool.iloc[:300].assign(window=1)
        reserve = pool.iloc[300:310]
        pred = np.full(300, 2)  # everything predicted hard: easy errors need easy fill
        with self.assertRaises(ValueError):
            P.build_batches(probe, pred, reserve, {"easy": .2, "medium": .2, "hard": .6}, 1, "fill")


class VersionPolicyTest(unittest.TestCase):
    def test_version_names_and_anchors_follow_operations(self):
        names = [P.version_name(k, 3) for k in range(1, 10)]
        self.assertEqual(names, ["1.1", "1.2", "2.0", "2.1", "2.2", "3.0", "3.1", "3.2", "4.0"])
        self.assertEqual([P.anchor_round(k, 3) for k in range(1, 10)], [0, 0, 0, 3, 3, 3, 6, 6, 6])

    def test_p1_and_p2_rows(self):
        seed_rows = frame(400).assign(sample_id=lambda f: "seed" + f.sample_id)
        batches = pd.concat([frame(100, k).assign(sample_id=lambda f, k=k: f"b{k}-" + f.sample_id, round=k)
                             for k in range(1, 6)], ignore_index=True)
        rows = P.p1_rows(5, batches, seed_rows, POLICY)  # cycle 2: rounds 4-5, history = rounds 1-3
        self.assertEqual(set(rows[rows.part.eq("cycle")]["round"]), {4, 5})
        self.assertEqual(int(rows.part.eq("cycle").sum()), 200)
        history = rows[rows.part.eq("history")]
        self.assertTrue(set(history["round"]) <= {1, 2, 3})
        self.assertTrue((history.difficulty_label.value_counts() <= 32).all())
        self.assertEqual(int(rows.part.eq("base").sum()), 96)
        first = P.p1_rows(2, batches, seed_rows, POLICY)
        self.assertEqual(int(first.part.eq("history").sum()), 0)
        self.assertEqual(int(first.part.eq("cycle").sum()), 200)
        full = P.p2_rows(5, batches, seed_rows)
        self.assertEqual(len(full), 400 + 500)

    def test_replay_order_is_stable(self):
        rows = frame(300)
        a = P.first_per_label(rows, 32)
        b = P.first_per_label(rows.sample(frac=1, random_state=1), 32)
        self.assertEqual(sorted(a.sample_id), sorted(b.sample_id))


class MetricTest(unittest.TestCase):
    def test_macro_f1_matches_definition(self):
        y = np.array([0, 0, 1, 2, 2, 2])
        pred = np.array([0, 1, 1, 2, 0, 2])
        f1 = P.macro_f1_from_confusion(P.confusion(y, pred))
        expected = np.mean([2 * .5 * .5 / 1, 2 * .5 * 1 / 1.5, 2 * 1 * (2 / 3) / (1 + 2 / 3)])
        self.assertAlmostEqual(float(f1), expected)
        self.assertEqual(float(P.macro_f1_from_confusion(P.confusion(np.zeros(4, int), np.zeros(4, int)))), 1 / 3)

    def test_select_sizes(self):
        curve = {1000: .30, 2000: .38, 4000: .45, 8000: .48, 16000: .50}
        s = P.select_sizes(curve, .20, .5, .03)
        self.assertEqual((s["n0"], s["stream_rows"]), (2000, 14000))
        self.assertTrue(s["ok"])
        flat = P.select_sizes({1000: .30, 2000: .30}, .25, .5, .03)
        self.assertFalse(flat["ok"])
        falling = P.select_sizes({1000: .20, 2000: .19}, .25, .5, .03)
        self.assertFalse(falling["ok"])

    def test_window_cap_keeps_fill_inside_reserve(self):
        share = {"easy": .5, "medium": .3, "hard": .2}
        fill = {"easy": .4, "medium": .02, "hard": 0.0}
        w = P.window_cap(10000, share, fill, 9, 1.0)
        for label in P.LABELS:
            self.assertLessEqual(9 * w * fill[label], (10000 - 9 * w) * share[label] + 1e-9)
        self.assertEqual(P.window_cap(900, share, {}, 9, 1.0), 100)

    def test_curve_interpolates_in_log_rows(self):
        curve = {1000: .30, 4000: .50}
        self.assertAlmostEqual(P.curve_at(curve, 2000), .40)
        self.assertAlmostEqual(P.curve_at(curve, 8000), .50)

    def test_bootstrap_identical_predictions_have_zero_difference(self):
        f = frame(400)
        pred = np.random.default_rng(0).integers(0, 3, 400)
        draws = compare.bootstrap_source_macro_f1(f, {"a": pred, "b": pred.copy()}, 200, 1)
        self.assertTrue(np.allclose(draws["a"], draws["b"]))
        observed = P.source_macro_f1(f, pred)["source_macro_f1"]
        self.assertAlmostEqual(float(np.median(draws["a"])), observed, delta=.05)

    def test_routing_outcomes_and_fixed_line(self):
        p = np.array([[.5, .8, 1.0]])
        o = compare.outcomes(p, np.array([0]), np.array([.2, .6, 1.0]))
        self.assertAlmostEqual(float(o["cost"][0]), .2 + .5 * .6 + .5 * .2 * 1.0)
        self.assertAlmostEqual(float(o["resolved"][0]), 1.0)
        points = [(.2, .5), (.6, .8), (1.0, .9)]
        self.assertAlmostEqual(compare.fixed_line(points, .4), .65)
        self.assertAlmostEqual(compare.fixed_line(points, .1), .5)
        self.assertAlmostEqual(compare.fixed_line(points, 2.0), .9)

    def test_noninferiority_labels(self):
        self.assertTrue(compare.noninferiority(-.002, [-.008, .004], .01).startswith("비열등"))
        self.assertTrue(compare.noninferiority(-.03, [-.04, -.02], .01).startswith("열등"))
        self.assertTrue(compare.noninferiority(-.01, [-.02, .0], .01).startswith("미확인"))


if __name__ == "__main__":
    unittest.main()
