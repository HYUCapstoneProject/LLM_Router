from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_router_labeling.shared.object_store import LocalObjectStore
from llm_router_labeling.shared.observability import write_observability_json
from llm_router_labeling.training.feedback_retraining_cli import _publish_metrics


class FeedbackObservabilityTest(unittest.TestCase):
    def test_external_batch_and_candidate_identify_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalObjectStore(Path(directory))
            root = "gs://test/runs/feedback-1/observability-v1"
            metrics = {
                "batch_id": "continual-cycle-1",
                "schema_version": "feedback-export-v0.1",
                "candidate_model_version": "internal-candidate",
                "accepted_event_count": 3,
            }
            original = dict(metrics)
            args = argparse.Namespace(
                observability_run_uri=root,
                batch_id="feedback-1",
                candidate_model_version="router-candidate",
            )
            with patch("llm_router_labeling.shared.observability.build_object_store", return_value=store):
                _publish_metrics(args, relative_uri="metrics/feedback-prepare.json", metrics=metrics)
                # Repeating the same stage is idempotent.
                _publish_metrics(args, relative_uri="metrics/feedback-prepare.json", metrics=metrics)
            value = json.loads(store.read_text(root + "/metrics/feedback-prepare.json"))
            self.assertEqual(value["batch_id"], "feedback-1")
            self.assertEqual(value["source_batch_id"], "continual-cycle-1")
            self.assertEqual(value["candidate_model_version"], "router-candidate")
            self.assertEqual(value["schema_version"], "retraining-observability-v1")
            self.assertEqual(value["source_schema_version"], "feedback-export-v0.1")
            self.assertEqual(value["accepted_event_count"], 3)
            self.assertEqual(metrics, original)

    def test_observability_schema_is_reserved_without_losing_source_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalObjectStore(Path(directory))
            uri = "gs://test/result.json"
            write_observability_json(uri, {"batch_id": "feedback-1"}, object_store=store)
            value = json.loads(store.read_text(uri))
            self.assertEqual(value["schema_version"], "retraining-observability-v1")
            self.assertNotIn("source_schema_version", value)


if __name__ == "__main__":
    unittest.main()
