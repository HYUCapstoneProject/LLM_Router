from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import tomllib


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
        self.assertEqual(package["project"]["dependencies"], [])
        copied = [
            line.split()[1]
            for line in dockerfile.splitlines()
            if line.startswith("COPY ") and line.split()[1].endswith(".py")
        ]
        training_modules = {Path(name).name for name in copied if "/training/" in name}
        self.assertEqual(training_modules, {"__init__.py", "routing_policy.py"})
        self.assertNotIn("torch", dockerfile)
        self.assertNotIn("transformers", dockerfile)
        self.assertNotIn("pandas", dockerfile)
        # Import the exact image's Python files with site-packages disabled.
        # This catches missing internal modules and accidental ML dependencies.
        with tempfile.TemporaryDirectory() as directory:
            isolated = Path(directory)
            for name in copied:
                source = ml_root.parent / name
                target = isolated / Path(name).relative_to("ml/src")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            for entry in package["project"]["scripts"].values():
                module = entry.split(":")[0]
                result = subprocess.run(
                    [sys.executable, "-S", "-m", module, "--help"],
                    cwd=isolated,
                    env={
                        **os.environ,
                        "PYTHONPATH": str(isolated),
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
        self.assertNotIn("/evaluation/", dockerfile)


if __name__ == "__main__":
    unittest.main()
