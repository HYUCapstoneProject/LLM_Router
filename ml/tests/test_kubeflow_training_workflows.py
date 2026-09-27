from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(
    importlib.util.find_spec("kfp")
    and importlib.util.find_spec("google_cloud_pipeline_components"),
    "KFP and Google Cloud Pipeline Components are required",
)
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
                "routerbench_v2_preprocessing_component",
                "feedback_prepare_component",
                "feedback_merge_component",
                "vertex_modernbert_training_job_component",
                "vertex_modernbert_continual_training_job_component",
                "routerarena_evaluation_component",
                "model_package_component",
                "deploy_gate_component",
                "model_publish_component",
                "model_auto_approve_component",
                "initial_training_pipeline",
                "feedback_retraining_pipeline",
                "feedback_direct_retraining_pipeline",
                "model_qualification_pipeline",
            },
        )
        self.assertIn("modernbert-router-initial-training", initial_yaml)
        self.assertIn("modernbert-router-feedback-retraining", retraining_yaml)
        self.assertIn("modernbert-router-model-qualification", qualification_yaml)
        self.assertIn("feedback_training.parquet", retraining_yaml)
        self.assertIn("/training.parquet", retraining_yaml)
        self.assertIn("minimum-sufficient-tier-v2", retraining_yaml)
        self.assertIn("modernbert_t4_balanced_v2.json", retraining_yaml)

        tasks = retraining["root"]["dag"]["tasks"]
        self.assertIn(
            "feedback-prepare-component",
            tasks["feedback-merge-component"]["dependentTasks"],
        )
        self.assertIn(
            "feedback-merge-component",
            tasks["modernbert-continual-training-component"]["dependentTasks"],
        )
        self.assertIn(
            "modernbert-continual-training-component",
            tasks["routerarena-evaluation-component"]["dependentTasks"],
        )
        self.assertIn(
            "model-package-component",
            tasks["deploy-gate-component"]["dependentTasks"],
        )
        # Publishing must run after the gate so a rejected candidate never
        # reaches the GCS prefix the serving deployment can be pointed at.
        self.assertIn(
            "deploy-gate-component",
            tasks["model-publish-component"]["dependentTasks"],
        )
        self.assertIn(
            "model-publish-component",
            tasks["model-auto-approve-component"]["dependentTasks"],
        )
        initial_tasks = initial["root"]["dag"]["tasks"]
        self.assertIn(
            "routerbench-v2-preprocessing-component",
            initial_tasks["modernbert-training-component"]["dependentTasks"],
        )
        self.assertIn(
            "modernbert-training-component",
            initial_tasks["routerarena-evaluation-component"]["dependentTasks"],
        )
        self.assertIn(
            "model-package-component",
            initial_tasks["deploy-gate-component"]["dependentTasks"],
        )
        self.assertIn(
            "deploy-gate-component",
            initial_tasks["model-publish-component"]["dependentTasks"],
        )
        self.assertIn(
            "model-publish-component",
            initial_tasks["model-auto-approve-component"]["dependentTasks"],
        )
        publish = retraining["deploymentSpec"]["executors"][
            "exec-model-publish-component"
        ]["container"]
        self.assertEqual(publish["command"], ["modernbert-publish"])
        self.assertIn("publish-candidate", publish["args"])
        self.assertNotIn("accelerator", publish["resources"])
        approve = retraining["deploymentSpec"]["executors"][
            "exec-model-auto-approve-component"
        ]["container"]
        self.assertEqual(approve["command"], ["modernbert-publish"])
        self.assertIn("approve", approve["args"])
        self.assertIn("--publish-result", approve["args"])
        self.assertIn("vertex-pipeline-deploy-gate", approve["args"])
        self.assertNotIn("accelerator", approve["resources"])

        # package_metadata_uri is one static file shared by every run, so the
        # package step must be told the run's actual candidate_model_version;
        # otherwise the deploy gate's package-version check always fails.
        package = retraining["deploymentSpec"]["executors"][
            "exec-model-package-component"
        ]["container"]
        self.assertIn("--model-version", package["args"])
        model_version_index = package["args"].index("--model-version")
        self.assertEqual(
            package["args"][model_version_index + 1],
            "{{$.inputs.parameters['candidate_model_version']}}",
        )
        initial_executors = initial["deploymentSpec"]["executors"]
        routerbench = initial_executors["exec-routerbench-v2-preprocessing-component"][
            "container"
        ]
        self.assertEqual(routerbench["command"], ["routerbench-labeling"])
        self.assertIn("build-full", routerbench["args"])
        self.assertNotIn("accelerator", routerbench["resources"])
        self.assertEqual(routerbench["resources"]["resourceCpuRequest"], "2")

        executors = retraining["deploymentSpec"]["executors"]
        prepare = executors["exec-feedback-prepare-component"]["container"]
        merge = executors["exec-feedback-merge-component"]["container"]
        training = executors["exec-modernbert-continual-training-component"][
            "container"
        ]
        self.assertEqual(prepare["command"], ["escalation-feedback-preprocessing"])
        self.assertEqual(merge["command"], ["escalation-feedback-preprocessing"])
        self.assertIn("--training-data-strategy", merge["args"])
        self.assertEqual(
            tasks["feedback-merge-component"]["inputs"]["parameters"][
                "training_data_strategy"
            ]["runtimeValue"]["constant"],
            "compact_rehearsal",
        )
        self.assertNotIn("accelerator", prepare["resources"])
        self.assertNotIn("accelerator", merge["resources"])
        self.assertEqual(prepare["resources"]["resourceCpuRequest"], "1")
        self.assertEqual(merge["resources"]["resourceCpuRequest"], "2")
        self.assertEqual(
            training["command"],
            [
                "python3",
                "-u",
                "-m",
                "google_cloud_pipeline_components.container.v1.custom_job.launcher",
            ],
        )
        worker_pool = retraining["components"][
            "comp-modernbert-continual-training-component"
        ]["inputDefinitions"]["parameters"]["worker_pool_specs"]["defaultValue"][0]
        self.assertEqual(worker_pool["machine_spec"]["machine_type"], "n1-standard-4")
        self.assertEqual(
            worker_pool["machine_spec"]["accelerator_type"],
            "NVIDIA_TESLA_T4",
        )
        self.assertEqual(worker_pool["machine_spec"]["accelerator_count"], 1.0)
        self.assertEqual(worker_pool["container_spec"]["command"][:2], ["python3", "-c"])
        self.assertIn("cuda", worker_pool["container_spec"]["args"])
        self.assertIn("--initial-model-path", worker_pool["container_spec"]["args"])
        self.assertIn("--initialization-mode", worker_pool["container_spec"]["args"])
        self.assertNotIn(
            "retryPolicy", tasks["modernbert-continual-training-component"]
        )
        self.assertNotIn("nvidia.com/gpu", retraining_yaml)

        qualification_tasks = qualification["root"]["dag"]["tasks"]
        self.assertIn(
            "routerarena-evaluation-component",
            qualification_tasks["model-package-component"]["dependentTasks"],
        )
        self.assertIn(
            "model-package-component",
            qualification_tasks["deploy-gate-component"]["dependentTasks"],
        )
        self.assertIn(
            "model-publish-component",
            qualification_tasks["model-auto-approve-component"]["dependentTasks"],
        )
        qualification_executors = qualification["deploymentSpec"]["executors"]
        evaluation = qualification_executors["exec-routerarena-evaluation-component"][
            "container"
        ]
        package = qualification_executors["exec-model-package-component"]["container"]
        gate = qualification_executors["exec-deploy-gate-component"]["container"]
        self.assertEqual(evaluation["command"], ["routerarena-evaluate"])
        self.assertNotIn("accelerator", evaluation["resources"])
        self.assertIn("cpu", evaluation["args"])
        self.assertNotIn("accelerator", package["resources"])
        self.assertNotIn("accelerator", gate["resources"])
        self.assertEqual(gate["command"], ["modernbert-deploy-gate"])

    def test_component_ir_uses_one_cpu_image_with_separate_commands(self) -> None:
        from ml.kubeflow.compile_training_workflows import compile_workflows

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = compile_workflows(Path(temporary_directory))
            routerbench = paths["routerbench_v2_preprocessing_component"].read_text(
                encoding="utf-8"
            )
            prepare = paths["feedback_prepare_component"].read_text(encoding="utf-8")
            merge = paths["feedback_merge_component"].read_text(encoding="utf-8")
            vertex_training = paths[
                "vertex_modernbert_training_job_component"
            ].read_text(encoding="utf-8")

        self.assertIn("routerbench-labeling", routerbench)
        self.assertIn("build-full", routerbench)
        self.assertNotIn("nvidia.com/gpu", routerbench)
        self.assertIn("llm-router-data-processing:py312-v2", prepare)
        self.assertIn("llm-router-data-processing:py312-v2", merge)
        self.assertIn("prepare", prepare)
        self.assertIn("merge", merge)
        self.assertNotIn("nvidia.com/gpu", prepare)
        self.assertNotIn("nvidia.com/gpu", merge)

        self.assertIn("NVIDIA_TESLA_T4", vertex_training)
        self.assertIn("n1-standard-4", vertex_training)
        self.assertIn("CustomJob", vertex_training)
        self.assertNotIn("nvidia.com/gpu", vertex_training)


if __name__ == "__main__":
    unittest.main()
