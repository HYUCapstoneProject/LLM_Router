import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/router_final"))
import lock as L  # noqa: E402


class OutcomeTest(unittest.TestCase):
    def test_escalation_expectations_match_hand_calculation(self):
        P = np.array([[0.5, 0.8, 1.0]])
        low = L.outcomes(P, np.array([0]))
        self.assertAlmostEqual(low["first"][0], 0.5)
        self.assertAlmostEqual(low["resolved"][0], 1 - 0.5 * 0.2 * 0.0)
        self.assertAlmostEqual(low["cost"][0], 0.2 + 0.5 * 0.6 + 0.5 * 0.2 * 1.0)
        self.assertAlmostEqual(low["fails"][0], 0.5 + 0.5 * 0.2 + 0.0)
        high = L.outcomes(P, np.array([2]))
        self.assertEqual((high["cost"][0], high["fails"][0], high["resolved"][0]), (1.0, 0.0, 1.0))

    def test_unsolvable_question_fails_every_attempt_from_its_start(self):
        out = L.outcomes(np.zeros((1, 3)), np.array([1]))
        self.assertEqual((out["resolved"][0], out["fails"][0]), (0.0, 2.0))
        self.assertAlmostEqual(out["cost"][0], 1.6)


class EnvelopeTest(unittest.TestCase):
    def setUp(self):
        self.points = [{"cost": 0.4, "resolved": 0.9, "fails": 0.5}, {"cost": 0.7, "resolved": 0.92, "fails": 0.2},
                       {"cost": 1.0, "resolved": 0.88, "fails": 0.1}]

    def test_mixture_interpolates_between_fixed_starts(self):
        v = L.envelope(self.points, 0.0, 0.55)
        self.assertAlmostEqual(v, 0.91)
        self.assertAlmostEqual(L.envelope(self.points, 0.0, 0.2), 0.9)  # below the cheapest start
        self.assertAlmostEqual(L.envelope(self.points, 0.0, 1.0), 0.92)  # dearer and worse start is not needed

    def test_failure_weight_changes_the_best_start(self):
        self.assertAlmostEqual(L.envelope(self.points, 1.0, 1.0), max(0.9 - 0.5, 0.92 - 0.2, 0.88 - 0.1))


class RuleTest(unittest.TestCase):
    def test_rho_is_zero_for_independent_and_one_for_nested_failures(self):
        rng = np.random.default_rng(0)
        independent = rng.uniform(size=(20000, 3))
        self.assertLess(abs(L.estimate_rho(independent)), 0.03)
        z = rng.uniform(size=(5000, 1))
        nested = np.hstack([z, np.minimum(1, z + 0.2), np.minimum(1, z + 0.4)])
        self.assertGreater(L.estimate_rho((nested > 0.5).astype(float)), 0.99)

    def test_tau_rule_takes_cheapest_sure_tier_else_most_likely(self):
        Phat = np.array([[0.9, 0.95, 1.0], [0.3, 0.85, 0.9], [0.2, 0.4, 0.3]])
        self.assertEqual(L.decide("tau", Phat, 0.0).tolist(), [0, 1, 1])

    def test_large_lambda_moves_to_cheaper_starts(self):
        Phat = np.array([[0.5, 0.7, 0.9]] * 4)
        chain = L.predicted_chain(Phat, 0.5)
        free = L.decide("escalation", Phat, 0.0, 1.0, chain)
        costly = L.decide("escalation", Phat, 10.0, 1.0, chain)
        self.assertTrue((costly <= free).all())

    def test_cost_targets_lie_inside_the_fixed_start_range(self):
        P = np.array([[0.6, 0.8, 0.9], [0.2, 0.5, 0.7]])
        costs = [p["cost"] for p in L.fixed_points(P)]
        targets = L.cost_targets(P)
        self.assertTrue(min(costs) < targets[0] < targets[1] < targets[2] < max(costs))


class CalibrationTest(unittest.TestCase):
    def test_class_outputs_use_the_train_table(self):
        table = L.class_table(pd.Series(["easy", "hard", "easy"]), np.array([[1, 1, 1], [0, 0, 1], [0.5, 1, 1.0]]),
                              ("easy", "medium", "hard"))
        np.testing.assert_allclose(table[0], [0.75, 1, 1])
        np.testing.assert_allclose(table[1], [0.5, 2 / 3, 1])  # no medium rows: overall mean
        X = L.features("class_probs", np.array([[10.0, -10.0, -10.0]]), table)
        np.testing.assert_allclose(X[0], [0.75, 1, 1], atol=1e-6)

    def test_single_score_feeds_every_tier_and_isotonic_follows_its_direction(self):
        X = L.features("single", np.array([[0.1, 9, 9], [0.9, 9, 9]]))
        self.assertEqual(X.tolist(), [[0.1] * 3, [0.9] * 3])
        rng = np.random.default_rng(1)
        x = rng.uniform(size=(400, 3))
        y = (rng.uniform(size=(400, 3)) < 1 - x).astype(float)  # decreasing relation
        p = L.fit_predict(x, y, np.array([[0.05] * 3, [0.95] * 3]))
        self.assertTrue((p[0] > p[1]).all())

    def test_folds_are_stable(self):
        ids = [f"q{i}" for i in range(50)]
        self.assertEqual(L.fold_of(ids).tolist(), L.fold_of(ids).tolist())
        self.assertEqual(set(L.fold_of(ids)), set(range(L.FOLDS)))


class EnglishOnlyTest(unittest.TestCase):
    def test_exclusion_list_holds_ids_only(self):
        table = pd.read_csv(ROOT.parent / L.NON_ENGLISH)
        self.assertEqual(list(table.columns),
                         ["sample_id", "source_dataset", "role", "source_family", "slice", "dominant_script"])
        self.assertTrue(table.sample_id.is_unique)
        self.assertTrue(table.role.isin(["dev", "test", "diagnostic"]).all())
        ids, record = L.english_exclusions(ROOT.parent)
        self.assertEqual(len(ids), record["rows"])
        self.assertEqual(sum(record["by_source_role"].values()), len(ids))
        self.assertIn("llmrouterbench/test", json.loads(json.dumps(record))["by_source_role"])

    def test_duplicate_ids_are_refused(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / L.NON_ENGLISH
            path.parent.mkdir(parents=True)
            pd.DataFrame({"sample_id": ["a", "a"], "source_dataset": ["s", "s"], "role": ["test", "test"]}).to_csv(path, index=False)
            with self.assertRaises(SystemExit):
                L.english_exclusions(Path(tmp))


if __name__ == "__main__":
    unittest.main()
