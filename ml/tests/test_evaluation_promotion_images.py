from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


class EvaluationPromotionImageContractTest(unittest.TestCase):
    def test_evaluation_image_contains_only_inference_command(self) -> None:
        ml_root = Path(__file__).resolve().parents[1]
        package = tomllib.loads(
            (ml_root / "docker" / "modernbert-evaluation.pyproject.toml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            package["project"]["scripts"],
            {
                "routerarena-evaluate": (
                    "llm_router_labeling.evaluation.routerarena_evaluation_cli:main"
                )
            },
        )
        dockerfile = (
            ml_root / "docker" / "modernbert-evaluation.Dockerfile"
        ).read_text(encoding="utf-8")
        self.assertNotIn("modernbert_training.py", dockerfile)
        self.assertNotIn("feedback_retraining", dockerfile)

    def test_promotion_image_has_no_model_or_data_dependency(self) -> None:
        ml_root = Path(__file__).resolve().parents[1]
        package = tomllib.loads(
            (ml_root / "docker" / "model-promotion.pyproject.toml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            set(package["project"]["scripts"]),
            {"modernbert-package", "modernbert-deploy-gate", "modernbert-publish"},
        )
        dockerfile = (ml_root / "docker" / "model-promotion.Dockerfile").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("requirements-training.lock", dockerfile)
        self.assertNotIn("requirements-lock.txt", dockerfile)
        self.assertNotIn("/training/", dockerfile)
        self.assertNotIn("/evaluation/", dockerfile)


if __name__ == "__main__":
    unittest.main()
