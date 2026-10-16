import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/router_final"))
import finetune_track as F  # noqa: E402


class SettingsTest(unittest.TestCase):
    def test_settings_are_distinct_and_documented(self):
        for method, settings in F.SETTINGS.items():
            ids = [F.C.setting_id(method, s) for s in settings]
            self.assertLessEqual(len(ids), 8)
            self.assertEqual(len(ids), len(set(ids)))
            self.assertIn(method, F.DEVIATIONS)
        self.assertEqual({s["config"] for s in F.SETTINGS["routerdc"]}, set(F.ROUTERDC_CONFIGS))

    def test_routerdc_keeps_top_and_last_k_within_three_tiers(self):
        self.assertLessEqual(F.ROUTERDC_COMMON["top_k"], 3)
        self.assertLessEqual(F.ROUTERDC_COMMON["last_k"], 3)


class SoftLabelTest(unittest.TestCase):
    def test_three_classes_follow_group_probabilities(self):
        probs = F.soft_three_class(np.array([1.0, 0.0, 0.5, 1.0]), np.array([1.0, 1.0, 0.5, 0.0]))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        np.testing.assert_allclose(probs[0], [0, 1, 0])  # both groups solve: tie
        np.testing.assert_allclose(probs[1], [1, 0, 0])  # only the strong (high) group solves
        np.testing.assert_allclose(probs[2], [0.25, 0.5, 0.25])
        np.testing.assert_allclose(probs[3], [0, 0, 1])  # only the weak (low) group solves

    def test_single_score_is_strong_win_rate(self):
        probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        scores, extra = F.routellm_scores(probs)
        np.testing.assert_allclose(scores, [[0.7] * 3, [0.1] * 3])
        np.testing.assert_allclose(1 - probs[:, 1:].sum(axis=1), scores[:, 0])  # BERTRouter: 1 - P(tie) - P(weak)
        self.assertEqual(sorted(extra), ["p_strong_win", "p_tie", "p_weak_win"])


class PaddingTest(unittest.TestCase):
    def test_trim_drops_columns_that_are_padding_everywhere(self):
        import torch
        tokens = {"input_ids": torch.tensor([[5, 6, 0, 0], [7, 0, 0, 0]]),
                  "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]])}
        batch = F.trim_padding([tokens, torch.zeros(2, 3)])
        self.assertEqual(batch[0]["input_ids"].tolist(), [[5, 6], [7, 0]])
        self.assertEqual(batch[0]["attention_mask"].tolist(), [[1, 1], [1, 0]])

    def test_length_order_sorts_by_text_length(self):
        self.assertEqual(F.length_order(["aaa", "a", "aa"]).tolist(), [1, 2, 0])


if __name__ == "__main__":
    unittest.main()
