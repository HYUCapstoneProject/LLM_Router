from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("kfp"), "kfp SDK is not installed")
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
            pipeline_definition = yaml.safe_load(pipeline_yaml)

        self.assertIn("modernbert-train", component_yaml)
        self.assertIn("llm-router-modernbert-training:cu124-v1", component_yaml)
        self.assertIn("training_dataset", component_yaml)
        self.assertIn("training_config", component_yaml)
        self.assertIn("training_path_suffix", component_yaml)
        self.assertIn("trained_model", component_yaml)
        self.assertIn("modernbert_runtime_cu124_v1.json", component_yaml)
        self.assertIn("cuda", component_yaml)
        resources = pipeline_definition["deploymentSpec"]["executors"][
            "exec-modernbert-training-component"
        ]["container"]["resources"]
        self.assertEqual(resources["accelerator"]["resourceType"], "nvidia.com/gpu")
        self.assertEqual(resources["accelerator"]["count"], "1")
        self.assertEqual(resources["resourceCpuRequest"], "4")
        self.assertEqual(resources["resourceMemoryRequest"], "24Gi")
        self.assertIn("training_dataset_uri", pipeline_yaml)
        self.assertIn("training_config_uri", pipeline_yaml)
        self.assertIn("training_image", pipeline_yaml)


if __name__ == "__main__":
    unittest.main()
