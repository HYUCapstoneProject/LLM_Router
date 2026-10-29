import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/router_final"))
import gain_eval as G  # noqa: E402
import gain_tasks as GT  # noqa: E402
import test_eval as T  # noqa: E402


class GainTasksTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.P = rng.uniform(size=(80, 3))
        self.groups = pd.Series(np.arange(80) // 2).astype(str)
        good = np.argmax(self.P >= 0.8, axis=1)
        self.sets = {"a": [("tau", 0.8, good, True), ("tau", 0.5, np.zeros(80, int), False)],
                     "b": [("tau", 0.8, np.full(80, 2), True)]}

    def test_cell_matches_gain_point_and_skips_other_taus(self):
        rows, kept = GT.cell_rows(self.P, self.sets, T.Draws(self.groups, 200, 42, 0))
        self.assertEqual([(r["method"], r["param"]) for r in rows], [("a", 0.8), ("b", 0.8)])
        self.assertAlmostEqual(rows[0]["G"], G.gain_point(self.P, self.sets["a"][0][2])["G"])
        self.assertLessEqual(rows[1]["G"], 1e-12)  # a fixed tier never beats the fixed-start baseline
        self.assertEqual(kept[("a", "tau", 0.8)].shape, (200,))

    def test_pool_is_row_weighted_and_pairs_use_shared_draws(self):
        cells, kept = [], {}
        for source, n, ga, gb in (("s1", 100, 0.1, 0.0), ("s2", 300, 0.3, 0.1)):
            for m, g in (("r1_group", ga), ("r0_group", gb)):
                cells.append({"source": source, "task_type": "수학", "rows": n, "method": m, "rule": "tau",
                              "param": 0.8, "A": g, "C": 0.5, "H": 0.0, "G": g, "start_low": 1.0,
                              "start_medium": 0.0, "start_high": 0.0})
                kept[(source, "수학", m, "tau", 0.8)] = np.full(50, g, np.float32)
        pooled, pairs = GT.pool(pd.DataFrame(cells), kept)
        g = pooled.set_index("method").G
        self.assertAlmostEqual(g["r1_group"], 0.25)
        self.assertAlmostEqual(g["r0_group"], 0.075)
        row = pairs[(pairs.a == "r1_group") & (pairs.b == "r0_group")].iloc[0]
        self.assertAlmostEqual(row["diff"], 0.175)
        self.assertAlmostEqual(row["lo"], 0.175, places=6)


if __name__ == "__main__":
    unittest.main()
