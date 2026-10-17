import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/router_final"))
import routing_latency as T  # noqa: E402


class SampleTest(unittest.TestCase):
    def test_fixed_dev_sample_per_source(self):
        scored = pd.DataFrame({"sample_id": [f"{s}:{i}" for s in ("a", "b") for i in range(10)] + ["c:0"],
                               "source_dataset": ["a"] * 10 + ["b"] * 10 + ["c"], "role": ["dev"] * 20 + ["test"]})
        inputs = {"scored": scored, "texts": [f"t{i}" for i in range(21)]}
        one, two = T.sample_questions(inputs, 3), T.sample_questions(inputs, 3)
        self.assertEqual(one.sample_id.tolist(), two.sample_id.tolist())
        self.assertEqual(one.groupby("source_dataset").size().to_dict(), {"a": 3, "b": 3})
        self.assertTrue((one.text == [inputs["texts"][r] for r in one.row]).all())

    def test_stats_report_median_and_p95(self):
        out = T.stats(list(range(1, 101)))
        self.assertAlmostEqual(out["median_ms"], 50.5)
        self.assertAlmostEqual(out["p95_ms"], np.percentile(range(1, 101), 95))

    def test_decision_stage_returns_a_tier(self):
        self.assertIn(T.DecisionStage(200)(np.array([[0.2, 0.6, 0.9]])), (0, 1, 2))


if __name__ == "__main__":
    unittest.main()
