from __future__ import annotations

import unittest
from pathlib import Path


class DataProcessingImageContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ml_root = Path(__file__).resolve().parents[1]

    def test_package_exposes_only_data_commands(self) -> None:
        pyproject = (self.ml_root / "docker" / "data-processing.pyproject.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn("routerbench-labeling", pyproject)
        self.assertIn("routerarena-preprocessing", pyproject)
        self.assertIn("escalation-feedback-preprocessing", pyproject)
        self.assertNotIn("modernbert-train", pyproject)
        self.assertNotIn("routerarena-evaluate", pyproject)

    def test_image_excludes_model_runtime(self) -> None:
        dockerfile = (self.ml_root / "docker" / "data-processing.Dockerfile").read_text(
            encoding="utf-8"
        )
        self.assertIn("USER 10001:10001", dockerfile)
        self.assertIn("routerbench_labeling.py", dockerfile)
        self.assertIn("routerarena_preprocessing.py", dockerfile)
        self.assertIn("feedback_retraining_preprocessing.py", dockerfile)
        self.assertNotIn("modernbert_training.py", dockerfile)
        self.assertNotIn("routerarena_routing_evaluator.py", dockerfile)
        self.assertNotIn("torch", dockerfile)
        self.assertNotIn("transformers", dockerfile)


if __name__ == "__main__":
    unittest.main()
