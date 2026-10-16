import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/router_final"))
import common_track as C  # noqa: E402


class SettingsTest(unittest.TestCase):
    def test_every_method_has_at_most_eight_distinct_settings(self):
        self.assertEqual(set(C.SETTINGS), set(C.METHODS))
        for method, settings in C.SETTINGS.items():
            ids = [C.setting_id(method, s) for s in settings]
            self.assertLessEqual(len(ids), 8, method)
            self.assertEqual(len(ids), len(set(ids)), method)
            self.assertIn(method, C.DEVIATIONS)

    def test_setting_id_is_readable(self):
        self.assertEqual(C.setting_id("mlp", {"hidden_layer_sizes": (100, 100)}), "hidden_layer_sizes=100-100")
        self.assertEqual(C.setting_id("r1_bge", {"hidden": 256, "source_weight": False}), "hidden=256,source_weight=off")
        self.assertEqual(C.setting_id("svm", {}), "default")


class DescriptionTest(unittest.TestCase):
    def test_each_tier_lists_its_members_in_every_source_and_its_price(self):
        groups = {s: {t: [f"{s}-{t}-model"] for t in C.TIERS} for s in C.SOURCES}
        texts = C.group_descriptions(groups, {"low": 0.2, "medium": 0.6, "high": 1.0})
        self.assertEqual(list(texts), list(C.TIERS))
        for tier in C.TIERS:
            for source in C.SOURCES:
                self.assertIn(f"{source}-{tier}-model", texts[tier])
            other = [t for t in C.TIERS if t != tier]
            self.assertNotIn(f"routerbench-{other[0]}-model", texts[tier])
        self.assertIn("1.0", texts["high"])


class ExternalPlumbingTest(unittest.TestCase):
    def test_array_cache_returns_rows_in_prompt_order(self):
        cache = C.ArrayCache(["a", "b", "c"], np.arange(6, dtype=np.float32).reshape(3, 2))
        self.assertEqual(cache.batch_get_embedding(("c", "a"), "bge").tolist(), [[4, 5], [0, 1]])

    def test_temporary_modules_restores_previous_entries(self):
        sys.modules["ev10_probe"] = "before"
        with C.temporary_modules({"ev10_probe": types.ModuleType("stub"), "ev10_new": types.ModuleType("x")}):
            self.assertIsInstance(sys.modules["ev10_probe"], types.ModuleType)
        self.assertEqual(sys.modules.pop("ev10_probe"), "before")
        self.assertNotIn("ev10_new", sys.modules)

    def test_verify_external_rejects_a_changed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            (dest / "repo").mkdir()
            (dest / "repo" / "a.py").write_text("x = 1\n")
            digest = hashlib.sha256(b"x = 1\n").hexdigest()
            spec = {"repo": {"repo": "o/r", "commit": "c", "license": "MIT", "files": {"a.py": digest}}}
            with mock.patch.dict(C.EXTERNAL, spec, clear=True):
                self.assertEqual(C.verify_external(dest)["repo"]["commit"], "c")
                (dest / "repo" / "a.py").write_text("x = 2\n")
                with self.assertRaises(SystemExit):
                    C.verify_external(dest)


class MethodHelpersTest(unittest.TestCase):
    def test_routing_frame_lists_tiers_cheapest_first_so_ties_pick_low(self):
        inputs = {
            "vectors": np.eye(3, dtype=np.float32),
            "row": pd.Series([0, 1, 2], index=["q0", "q1", "d0"]),
            "train": {"blended": pd.DataFrame({"sample_id": ["q0", "q1"], "source_dataset": ["sprout"] * 2,
                                               "sample_weight": [1.0, 1.0],
                                               "success_low": [1.0, 0.0], "success_medium": [1.0, 0.5],
                                               "success_high": [1.0, 1.0]})},
            "scored": pd.DataFrame({"sample_id": ["d0"], "source_dataset": ["sprout"], "role": ["dev"]}),
        }
        ctx = C.Context(inputs, np.zeros((3, 3), np.float32), "cpu")
        frame = ctx.routing_frame()
        self.assertEqual(frame.model_name.tolist(), ["low", "medium", "high"] * 2)
        best = frame.groupby("query").apply(lambda g: g.loc[g.performance.idxmax(), "model_name"], include_groups=False)
        self.assertEqual(best.to_dict(), {"q0": "low", "q1": "high"})
        self.assertEqual(ctx.query.tolist(), [[0, 0, 1]])

    def test_cold_start_vector_is_mean_of_nearest_training_questions(self):
        train = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0], [0.0, 1.0], [50.0, 50.0]])
        cold = C.cold_start_vectors(train, np.array([[0.2, 0.2]]), k=3)
        np.testing.assert_allclose(cold, [[1 / 3, 1 / 3]])

    def test_avengers_scores_follow_rank_weighted_cluster_mixture(self):
        from sklearn.preprocessing import Normalizer
        router = types.SimpleNamespace(
            normalizer=Normalizer().fit(np.eye(2)), cluster_centers=np.array([[1.0, 0.0], [0.0, 1.0]]),
            config=types.SimpleNamespace(top_k=1, beta=9.0),
            cluster_rankings={0: {"ranking": ["low", "medium", "high"]}, 1: {"ranking": ["high", "low", "medium"]}})
        scores = C.avengers_scores(router, np.array([[1.0, 0.1], [0.1, 1.0]]), list(C.TIERS))
        np.testing.assert_allclose(scores, [[1, 1 / 2, 1 / 3], [1 / 2, 1 / 3, 1]])
        router.config.top_k = 2
        mixed = C.avengers_scores(router, np.array([[1.0, 1.0]]), list(C.TIERS))  # equidistant: equal weights
        np.testing.assert_allclose(mixed, [[0.75, 5 / 12, 2 / 3]])


class GraphRouterModelNodeTest(unittest.TestCase):
    def test_missing_records_take_the_group_probability_so_group_means_match(self):
        success = pd.DataFrame({"a": [1.0, np.nan, 0.0], "b": [0.0, 1.0, np.nan], "c": [np.nan, np.nan, 1.0],
                                "d": [0.0, 1.0, 1.0]})
        target = pd.DataFrame({"success_low": [0.5, 1.0, 0.0], "success_medium": [0.25, 0.75, 1.0],
                               "success_high": [0.0, 1.0, 1.0]})
        filled, missing = C.fill_missing_with_target(success, ["low", "low", "medium", "high"], target)
        self.assertEqual(filled["b"].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(filled["c"].tolist(), [0.25, 0.75, 1.0])
        self.assertEqual(int(missing.to_numpy().sum()), 4)
        target.loc[0, "success_low"] = 0.9  # inconsistent with the observed records
        with self.assertRaises(SystemExit):
            C.fill_missing_with_target(success, ["low", "low", "medium", "high"], target)

    def test_tier_score_is_the_mean_over_the_tier_models(self):
        pred = np.array([[0.2, 0.4, 0.9, 1.0], [0.0, 0.0, 0.5, 0.0]])
        np.testing.assert_allclose(C.tier_means(pred, ["low", "low", "medium", "high"]), [[0.3, 0.9, 1.0], [0.0, 0.5, 0.0]])

    def test_model_descriptions_give_price_relative_to_the_pool(self):
        members = {s: [{"model": f"{s}-cheap", "tier": "low", "cost": 1.0}, {"model": f"{s}-dear", "tier": "high", "cost": 4.0}]
                   for s in C.SOURCES}
        texts = C.model_descriptions(members)
        self.assertIn("relative price 0.250", texts["sprout"][0])
        self.assertIn("high price tier", texts["sprout"][1])


class WriteSettingTest(unittest.TestCase):
    def _ctx(self):
        ctx = types.SimpleNamespace(device="cpu", scored=pd.DataFrame({
            "sample_id": ["a", "b"], "source_dataset": ["sprout", "sprout"], "role": ["dev", "test"]}))
        return ctx

    def test_writes_scores_and_flags_constant_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "svm" / "default"
            info = {"fit_seconds": 1.0, "score_seconds": 0.5, "train_rows": 10}
            record = C.write_setting(folder, self._ctx(), "svm", {}, np.ones((2, 3)), {}, info)
            self.assertTrue(record["degenerate"])
            self.assertTrue((folder / "_SUCCESS").exists())
            table = pd.read_parquet(folder / "scores.parquet")
            self.assertEqual(list(table.columns), ["sample_id", "source_dataset", "role", "score_low", "score_medium", "score_high"])
            record = C.write_setting(Path(tmp) / "x", self._ctx(), "svm", {}, np.array([[0, 1, 2], [1, 2, 3.0]]), {}, info)
            self.assertFalse(record["degenerate"])

    def test_only_pool_specific_methods_may_leave_routerarena_unscored(self):
        ctx = types.SimpleNamespace(device="cpu", scored=pd.DataFrame({
            "sample_id": ["a", "b", "r"], "source_dataset": ["sprout", "sprout", "routerarena"], "role": ["dev", "test", "diagnostic"]}))
        scores = np.array([[0, 1, 2], [1, 2, 3.0], [np.nan] * 3])
        info = {"fit_seconds": 1.0, "score_seconds": 0.5, "train_rows": 10}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                C.write_setting(Path(tmp) / "a", ctx, "graphrouter", {}, scores, {}, info)
            record = C.write_setting(Path(tmp) / "b", ctx, "graphrouter", {}, scores, {},
                                     info | {"diagnostic_unscored": True, "external_key": "graphrouter"})
            self.assertFalse(record["degenerate"])
            self.assertEqual(record["external"]["repo"], "ulab-uiuc/GraphRouter")
            dev_nan = scores.copy(); dev_nan[0, 0] = np.nan
            with self.assertRaises(SystemExit):
                C.write_setting(Path(tmp) / "c", ctx, "graphrouter", {}, dev_nan, {}, info | {"diagnostic_unscored": True})

    def test_rejects_wrong_shape_and_non_finite_scores(self):
        info = {"fit_seconds": 1.0, "score_seconds": 0.5, "train_rows": 10}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                C.write_setting(Path(tmp) / "a", self._ctx(), "svm", {}, np.ones((3, 3)), {}, info)
            with self.assertRaises(SystemExit):
                C.write_setting(Path(tmp) / "b", self._ctx(), "svm", {}, np.array([[0, 1, np.nan], [1, 1, 1]]), {}, info)


if __name__ == "__main__":
    unittest.main()
