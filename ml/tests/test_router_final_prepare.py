import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/router_final"))
import prepare as P  # noqa: E402


class GroupModelsTest(unittest.TestCase):
    def test_drops_dominated_models_and_orders_tiers(self):
        stats = pd.DataFrame({
            "model": ["a", "b", "c", "d", "e", "weak_expensive"],
            "quality": [0.50, 0.55, 0.70, 0.72, 0.85, 0.52],
            "cost": [1.0, 1.2, 4.0, 4.5, 40.0, 30.0],
        })
        g = P.group_models(stats, delta=0.03).set_index("model")
        self.assertFalse(g.loc["weak_expensive", "kept"])
        self.assertEqual(g.loc[["a", "b"], "tier"].tolist(), ["low", "low"])
        self.assertEqual(g.loc[["c", "d"], "tier"].tolist(), ["medium", "medium"])
        self.assertEqual(g.loc["e", "tier"], "high")

    def test_rejects_unordered_tiers(self):
        stats = pd.DataFrame({"model": list("abc"), "quality": [0.9, 0.5, 0.6], "cost": [1.0, 10.0, 100.0]})
        with self.assertRaises(ValueError):
            P.group_models(stats, delta=1.0)


class LabelTest(unittest.TestCase):
    def test_needed_tier_is_cheapest_sure_tier_else_most_likely(self):
        probs = np.array([[0.9, 1.0, 1.0], [0.5, 0.8, 1.0], [0.0, 0.4, 0.6], [0.3, 0.3, 0.1]])
        self.assertEqual(P.needed_tier(probs, 0.8).tolist(), [0, 1, 2, 0])

    def test_old_definition_label_boundaries(self):
        labels = P.old_definition_label(np.array([0.27, 0.2699, 0.64, 0.6399]), 0.27, 0.64)
        self.assertEqual(labels.tolist(), ["medium", "hard", "easy", "medium"])


class SplitTest(unittest.TestCase):
    def test_groups_stay_together_and_forced_groups_go_to_test(self):
        groups = pd.Series(["g1", "g1", "g2", "g3"] * 50 + [f"x{i}" for i in range(2000)])
        roles = P.assign_split(groups, {"train": 0.5, "dev": 0.15, "test": 0.35}, 7, forced_test={"g2"})
        frame = pd.DataFrame({"g": groups, "r": roles})
        self.assertTrue((frame.groupby("g")["r"].nunique() == 1).all())
        self.assertTrue((frame.loc[frame.g.eq("g2"), "r"] == "test").all())
        share = frame[frame.g.str.startswith("x")]["r"].value_counts(normalize=True)
        self.assertAlmostEqual(share["train"], 0.5, delta=0.04)
        self.assertAlmostEqual(share["test"], 0.35, delta=0.04)

    def test_split_is_deterministic(self):
        groups = pd.Series([f"q{i}" for i in range(100)])
        fractions = {"train": 0.5, "dev": 0.15, "test": 0.35}
        self.assertTrue(P.assign_split(groups, fractions, 3, set()).equals(P.assign_split(groups, fractions, 3, set())))


class OverlapKeyTest(unittest.TestCase):
    def test_question_body_ignores_options_and_formatting(self):
        a = "What is the wavelength of a 10-eV electron?\n\nA. 2.5\nB. 3.8"
        b = "what is the wavelength of a 10 eV electron?\n(A) 1.0"
        self.assertEqual(P.question_body(a), P.question_body(b))


if __name__ == "__main__":
    unittest.main()
