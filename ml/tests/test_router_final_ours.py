import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/router_final"))
import ours  # noqa: E402


class OursTest(unittest.TestCase):
    def test_balanced_weight_gives_each_source_equal_total(self):
        frame = pd.DataFrame({"sample_id": range(10), "source_dataset": ["a"] * 8 + ["b"] * 2})
        weight = ours.balanced_weight(frame)
        totals = weight.groupby(frame.source_dataset).sum()
        self.assertAlmostEqual(totals["a"], totals["b"])
        self.assertAlmostEqual(weight.mean(), 1.0)

    def test_single_missing_tier_is_filled_from_neighbours(self):
        rows = pd.DataFrame({"p_low": [0.0, np.nan, 0.2, np.nan],
                             "p_medium": [np.nan, 0.6, 0.4, np.nan],
                             "p_high": [1.0, 0.8, np.nan, 0.5]})
        out, imputed, sparse = ours.impute_single_missing_tier(rows)
        self.assertEqual(out.loc[0, "p_medium"], 0.5)
        self.assertEqual(out.loc[1, "p_low"], 0.6)
        self.assertEqual(out.loc[2, "p_high"], 0.4)
        self.assertEqual(imputed.tolist(), [True, True, True, False])
        self.assertEqual(sparse.tolist(), [False, False, False, True])


if __name__ == "__main__":
    unittest.main()
