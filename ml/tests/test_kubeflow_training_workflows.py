from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("kfp"), "kfp SDK is not installed")
class KubeflowTrainingWorkflowsTest(unittest.TestCase):
    def test_initial_and_retraining_workflows_compile_with_expected_boundaries(
        self,
    ) -> None:
        import yaml

        from ml.kubeflow.compile_training_workflows import compile_workflows

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = compile_workflows(Path(temporary_directory))
            retraining_yaml = paths["feedback_retraining_pipeline"].read_text(
                encoding="utf-8"
            )
            initial_yaml = paths["initial_training_pipeline"].read_text(
                encoding="utf-8"
            )
            qualification_yaml = paths["model_qualification_pipeline"].read_text(
                encoding="utf-8"
            )
            retraining = yaml.safe_load(retraining_yaml)
            qualification = yaml.safe_load(qualification_yaml)
            initial = yaml.safe_load(initial_yaml)

        self.assertEqual(
            set(paths),
            {
                "feedback_prepare_component",
                "feedback_merge_component",
                "modernbert_training_component",
                "routerarena_evaluation_component",
                "model_package_component",
                "deploy_gate_component",
                "initial_training_pipeline",
                "feedback_retraining_pipeline",
                "model_qualification_pipeline",
            },
        )
        self.assertIn("modernbert-router-initial-training", initial_yaml)
        self.assertIn("modernbert-router-feedback-retraining", retraining_yaml)
        self.assertIn("modernbert-router-model-qualification", qualification_yaml)
        self.assertIn("feedback_training.parquet", retraining_yaml)
        self.assertIn("/training.parquet", retraining_yaml)

        tasks = retraining["root"]["dag"]["tasks"]
        self.assertIn(
            "feedback-prepare-component",
            tasks["feedback-merge-component"]["dependentTasks"],
        )
        self.assertIn(
            "feedback-merge-component",
            tasks["modernbert-training-component"]["dependentTasks"],
        )
        self.assertIn(
            "modernbert-training-component",
            tasks["routerarena-evaluation-component"]["dependentTasks"],
        )
        self.assertIn(
            "model-package-component",
            tasks["deploy-gate-component"]["dependentTasks"],
        )
        initial_tasks = initial["root"]["dag"]["tasks"]
        self.assertIn(
            "modernbert-training-component",
            initial_tasks["routerarena-evaluation-component"]["dependentTasks"],
        )
        self.assertIn(
            "model-package-component",
            initial_tasks["deploy-gate-component"]["dependentTasks"],
        )

        executors = retraining["deploymentSpec"]["executors"]
        prepare = executors["exec-feedback-prepare-component"]["container"]
        merge = executors["exec-feedback-merge-component"]["container"]
        training = executors["exec-modernbert-training-component"]["container"]
        self.assertEqual(prepare["command"], ["escalation-feedback-preprocessing"])
        self.assertEqual(merge["command"], ["escalation-feedback-preprocessing"])
        self.assertNotIn("accelerator", prepare["resources"])
        self.assertNotIn("accelerator", merge["resources"])
        self.assertEqual(prepare["resources"]["resourceCpuRequest"], "1")
        self.assertEqual(merge["resources"]["resourceCpuRequest"], "2")
        self.assertEqual(
            training["resources"]["accelerator"]["resourceType"],
            "nvidia.com/gpu",
        )
        self.assertEqual(training["resources"]["accelerator"]["count"], "1")

        qualification_tasks = qualification["root"]["dag"]["tasks"]
        self.assertIn(
            "routerarena-evaluation-component",
            qualification_tasks["model-package-component"]["dependentTasks"],
        )
        self.assertIn(
            "model-package-component",
            qualification_tasks["deploy-gate-component"]["dependentTasks"],
        )
        qualification_executors = qualification["deploymentSpec"]["executors"]
        evaluation = qualification_executors[
            "exec-routerarena-evaluation-component"
        ]["container"]
        package = qualification_executors["exec-model-package-component"][
            "container"
        ]
        gate = qualification_executors["exec-deploy-gate-component"]["container"]
        self.assertEqual(evaluation["command"], ["routerarena-evaluate"])
        self.assertEqual(
            evaluation["resources"]["accelerator"]["resourceType"],
            "nvidia.com/gpu",
        )
        self.assertNotIn("accelerator", package["resources"])
        self.assertNotIn("accelerator", gate["resources"])
        self.assertEqual(gate["command"], ["modernbert-deploy-gate"])

    def test_component_ir_uses_one_cpu_image_with_separate_commands(self) -> None:
        from ml.kubeflow.compile_training_workflows import compile_workflows

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = compile_workflows(Path(temporary_directory))
            prepare = paths["feedback_prepare_component"].read_text(encoding="utf-8")
            merge = paths["feedback_merge_component"].read_text(encoding="utf-8")

        self.assertIn("llm-router-data-processing:py312-v2", prepare)
        self.assertIn("llm-router-data-processing:py312-v2", merge)
        self.assertIn("prepare", prepare)
        self.assertIn("merge", merge)
        self.assertNotIn("nvidia.com/gpu", prepare)
        self.assertNotIn("nvidia.com/gpu", merge)


if __name__ == "__main__":
    unittest.main()
