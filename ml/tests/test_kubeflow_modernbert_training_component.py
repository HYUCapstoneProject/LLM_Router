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
class KubeflowModernBertTrainingTest(unittest.TestCase):
    def test_component_and_pipeline_compile_to_ir_yaml(self) -> None:
        import yaml

        from ml.kubeflow.compile_modernbert_training import compile_definitions

        with tempfile.TemporaryDirectory() as temporary_directory:
            component_path, pipeline_path = compile_definitions(
                Path(temporary_directory)
            )
            component_yaml = component_path.read_text(encoding="utf-8")
            pipeline_yaml = pipeline_path.read_text(encoding="utf-8")
            component_definition = yaml.safe_load(component_yaml)
            pipeline_definition = yaml.safe_load(pipeline_yaml)

        self.assertIn("modernbert-train", component_yaml)
        self.assertIn("llm-router-modernbert-training:cu124-v1", component_yaml)
        self.assertIn("training_dataset", component_yaml)
        self.assertIn("training_config", component_yaml)
        self.assertIn("training_path_suffix", component_yaml)
        self.assertIn("trained_model", component_yaml)
        self.assertIn("modernbert_runtime_cu124_v1.json", component_yaml)
        self.assertIn("cuda", component_yaml)
        worker_pool = pipeline_definition["components"][
            "comp-modernbert-training-component"
        ]["inputDefinitions"]["parameters"]["worker_pool_specs"]["defaultValue"][0]
        self.assertEqual(worker_pool["machine_spec"]["machine_type"], "n1-standard-4")
        self.assertEqual(
            worker_pool["machine_spec"]["accelerator_type"],
            "NVIDIA_TESLA_T4",
        )
        self.assertEqual(worker_pool["machine_spec"]["accelerator_count"], 1.0)
        self.assertEqual(worker_pool["container_spec"]["command"][:2], ["python3", "-c"])
        self.assertIn("cuda", worker_pool["container_spec"]["args"])
        training_executor = pipeline_definition["deploymentSpec"]["executors"][
            "exec-modernbert-training-component"
        ]["container"]
        self.assertIn("custom_job.launcher", " ".join(training_executor["command"]))
        self.assertNotIn("resources", training_executor)
        self.assertNotIn("nvidia.com/gpu", pipeline_yaml)
        self.assertIn("NVIDIA_TESLA_T4", component_yaml)
        self.assertIn("CustomJob", component_yaml)
        self.assertEqual(
            component_definition["pipelineInfo"]["name"],
            "custom-training-job",
        )
        self.assertIn("routerbench_zero_shot_uri", pipeline_yaml)
        self.assertIn("routerbench_five_shot_uri", pipeline_yaml)
        self.assertIn("routerbench_labeling_config_uri", pipeline_yaml)
        self.assertIn("routerbench_labeling_v2.json", pipeline_yaml)
        self.assertIn("modernbert_t4_balanced_v2.json", pipeline_yaml)
        self.assertIn("routerbench-v2-preprocessing-component", pipeline_yaml)
        self.assertIn("training_config_uri", pipeline_yaml)
        self.assertIn("training_service_account", pipeline_yaml)
        self.assertIn("training_base_output_directory", pipeline_yaml)

    def test_vertex_worker_receives_real_path_for_bundle_and_direct_artifact(self) -> None:
        """Execute the compiled worker command, not only the original component."""
        import json
        import os
        import subprocess
        import sys

        import yaml

        from ml.kubeflow.compile_training_workflows import compile_workflows

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = compile_workflows(root / "ir")
            probe = root / "modernbert-train"
            probe.write_text(
                f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n"
            )
            probe.chmod(0o755)
            for pipeline, component in (
                ("initial_training_pipeline", "comp-modernbert-training-component"),
                ("feedback_retraining_pipeline", "comp-modernbert-continual-training-component"),
            ):
                ir = yaml.safe_load(paths[pipeline].read_text())
                worker = ir["components"][component]["inputDefinitions"]["parameters"][
                    "worker_pool_specs"
                ]["defaultValue"][0]["container_spec"]
                self.assertNotIn("Concat", json.dumps(worker))
                self.assertEqual(worker["args"][0], "{{$.inputs.artifacts['training_dataset'].path}}")
                self.assertEqual(worker["args"][1], "{{$.inputs.parameters['training_path_suffix']}}")
                for suffix in ("", "/training.parquet"):
                    with self.subTest(pipeline=pipeline, suffix=suffix):
                        base = "/gcs/test bucket/quote'and$(not-a-command)/artifact"
                        resolved = [base, suffix, *worker["args"][2:]]
                        result = subprocess.run(
                            [sys.executable, *worker["command"][1:], *resolved],
                            env={**os.environ, "PATH": str(root) + os.pathsep + os.environ["PATH"]},
                            check=True, capture_output=True, text=True,
                        )
                        argv = json.loads(result.stdout)
                        self.assertEqual(argv[:2], ["--training-path", base + suffix])
                        self.assertEqual(argv[2:], resolved[2:])


if __name__ == "__main__":
    unittest.main()
