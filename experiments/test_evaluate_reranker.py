import unittest

from evaluate_reranker import (
    encode_text_pair,
    revision_from_snapshot_path,
    compute_joint_metrics,
    compute_metrics,
    percentile,
    select_joint_min_false_hits,
    select_joint_thresholds,
    select_min_false_hits,
    select_threshold,
)


class ModelMetadataTest(unittest.TestCase):
    def test_extracts_huggingface_revision_from_snapshot_path(self):
        path = "cache/models--GPTCache--albert-duplicate-onnx/snapshots/abc123/model.onnx"

        self.assertEqual(revision_from_snapshot_path(path), "abc123")

    def test_returns_none_without_snapshot_segment(self):
        self.assertIsNone(revision_from_snapshot_path("cache/model.onnx"))

    def test_uses_modern_tokenizer_call_when_encode_plus_is_unavailable(self):
        class ModernTokenizer:
            def __call__(self, query, candidate, **kwargs):
                return {"pair": (query, candidate), "kwargs": kwargs}

        encoded = encode_text_pair(ModernTokenizer(), "query", "candidate")

        self.assertEqual(encoded["pair"], ("query", "candidate"))
        self.assertEqual(encoded["kwargs"]["padding"], "longest")
        self.assertTrue(encoded["kwargs"]["return_token_type_ids"])


class ComputeMetricsTest(unittest.TestCase):
    def test_computes_confusion_and_rates(self):
        labels = ["positive", "positive", "negative", "negative"]
        scores = [0.95, 0.75, 0.85, 0.10]

        result = compute_metrics(labels, scores, threshold=0.80)

        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["tn"], 1)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["false_hit_rate"], 0.5)

    def test_rejects_unknown_label(self):
        with self.assertRaisesRegex(ValueError, "label"):
            compute_metrics(["maybe"], [0.5], threshold=0.5)

    def test_rejects_non_finite_score(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_metrics(["positive"], [float("nan")], threshold=0.5)


class ThresholdSelectionTest(unittest.TestCase):
    def test_selects_highest_recall_with_zero_false_hits(self):
        labels = ["positive", "positive", "positive", "negative", "negative"]
        scores = [0.95, 0.90, 0.60, 0.85, 0.20]

        result = select_threshold(labels, scores, max_false_hits=0)

        self.assertAlmostEqual(result["threshold"], 0.90)
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["fp"], 0)

    def test_returns_none_when_recall_gate_cannot_be_met(self):
        labels = ["positive", "negative"]
        scores = [0.4, 0.9]

        result = select_threshold(
            labels,
            scores,
            max_false_hits=0,
            min_recall=1.0,
        )

        self.assertIsNone(result)

    def test_minimizes_false_hits_at_recall_gate_then_maximizes_recall(self):
        labels = ["positive", "positive", "positive", "negative", "negative"]
        scores = [0.90, 0.80, 0.70, 0.85, 0.10]

        result = select_min_false_hits(labels, scores, min_recall=2 / 3)

        self.assertAlmostEqual(result["threshold"], 0.70)
        self.assertEqual(result["tp"], 3)
        self.assertEqual(result["fp"], 1)

    def test_joint_thresholds_can_remove_complementary_false_hits(self):
        labels = ["positive", "positive", "negative", "negative"]
        similarities = [0.90, 0.80, 0.95, 0.70]
        reranker_scores = [0.80, 0.90, 0.60, 0.95]

        result = select_joint_thresholds(
            labels,
            similarities,
            reranker_scores,
            max_false_hits=0,
            min_recall=1.0,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["fp"], 0)
        verified = compute_joint_metrics(
            labels,
            similarities,
            reranker_scores,
            result["similarity_threshold"],
            result["reranker_threshold"],
        )
        self.assertEqual(verified["recall"], 1.0)

    def test_joint_min_false_hits_prefers_fewer_errors_at_recall_gate(self):
        labels = ["positive", "positive", "negative", "negative"]
        similarities = [0.90, 0.80, 0.95, 0.70]
        reranker_scores = [0.80, 0.90, 0.60, 0.95]

        result = select_joint_min_false_hits(
            labels,
            similarities,
            reranker_scores,
            min_recall=1.0,
        )

        self.assertEqual(result["tp"], 2)
        self.assertEqual(result["fp"], 0)


class PercentileTest(unittest.TestCase):
    def test_uses_nearest_rank(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5], 0.95), 5)

    def test_empty_input_is_zero(self):
        self.assertEqual(percentile([], 0.95), 0.0)


if __name__ == "__main__":
    unittest.main()
