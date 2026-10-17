import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/router_final"))
import lock as L  # noqa: E402
import test_eval as T  # noqa: E402


class EnvelopeTest(unittest.TestCase):
    def test_vectorised_envelope_matches_the_lock(self):
        rng = np.random.default_rng(0)
        for _ in range(400):
            fc, fv = rng.uniform(0.2, 1.1, 3), rng.uniform(-0.5, 1.0, 3)
            x = rng.uniform(0.1, 1.2)
            points = [{"cost": c, "resolved": v, "fails": 0.0} for c, v in zip(fc, fv)]
            self.assertAlmostEqual(float(T.envelope_batch(fc[None], fv[None], np.array([x]))[0]),
                                   L.envelope(points, 0.0, x), places=10)

    def test_envelope_interpolates_mixtures(self):
        fc, fv = np.array([[0.4, 0.7, 1.0]]), np.array([[0.5, 0.6, 0.9]])  # middle point below the chord
        self.assertAlmostEqual(float(T.envelope_batch(fc, fv, np.array([0.7]))[0]), 0.7)


class DrawsTest(unittest.TestCase):
    def test_same_stream_gives_identical_draws_and_groups_move_together(self):
        groups = ["a", "a", "b", "c", "c", "c"]
        X = np.arange(12, dtype=float).reshape(6, 2)
        d1, d2 = T.Draws(groups, 50, 42, 0), T.Draws(groups, 50, 42, 0)
        np.testing.assert_array_equal(d1.means(X), d2.means(X))
        for _, W in d1.counts():
            self.assertTrue((W[:, 0] == W[:, 1]).all() and (W[:, 3] == W[:, 5]).all())
            np.testing.assert_array_equal(W[:, [0, 2, 3]].sum(axis=1), 3)  # three groups drawn each time

    def test_other_stream_differs(self):
        groups = list("abcdefgh")
        X = np.arange(8, dtype=float)[:, None]
        self.assertFalse(np.array_equal(T.Draws(groups, 30, 42, 0).means(X), T.Draws(groups, 30, 42, 1).means(X)))


class RuleTest(unittest.TestCase):
    def test_kappa_star_needs_margin_and_interval(self):
        rows = pd.DataFrame({"kappa": [0.0, 0.1, 0.2, 0.3], "gain": [0.0, 0.02, 0.02, 0.05],
                             "gain_lo": [-0.01, -0.001, 0.001, 0.02]})
        self.assertEqual(T.kappa_star(rows), 0.2)
        self.assertTrue(np.isnan(T.kappa_star(rows.assign(gain=0.005))))

    def test_track2_counts_calls_and_needed_accuracy(self):
        P = np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        M = T.evaluate_track2(P, np.array([1, 0]), np.array([0, 0]), L.SERVICE_COST)
        np.testing.assert_allclose(M[:, 0], [0, 1])  # needed-tier accuracy
        np.testing.assert_allclose(M[:, 3], [2, 1])  # calls
        np.testing.assert_allclose(M[:, 4], [0.8, 0.2])  # cost

    def test_spearman_is_rank_correlation(self):
        self.assertAlmostEqual(T.spearman(np.array([1.0, 2, 3, 4]), np.array([10.0, 20, 25, 100])), 1.0)


class CategoryTest(unittest.TestCase):
    def test_every_category_maps_to_a_listed_task_type(self):
        import categories as K
        self.assertTrue(set(K.TASK_TYPE.values()) <= set(K.TYPES))
        self.assertEqual(len(K.TYPES), len(set(K.TYPES)))
        sources = {s for s, _ in K.TASK_TYPE}
        self.assertEqual(sources, set(L.SOURCES))


if __name__ == "__main__":
    unittest.main()
