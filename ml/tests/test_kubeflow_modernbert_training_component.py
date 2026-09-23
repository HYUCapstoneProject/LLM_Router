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
        self.assertEqual(worker_pool["container_spec"]["command"], ["modernbert-train"])
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


if __name__ == "__main__":
    unittest.main()
