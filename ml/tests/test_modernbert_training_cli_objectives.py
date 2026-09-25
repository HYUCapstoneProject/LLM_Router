from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llm_router_labeling.objective_comparison.objectives import primary_row_losses
from llm_router_labeling.training.modernbert_training import ModernBertTrainingError
from llm_router_labeling.training.modernbert_training_cli import build_parser, main
from test_modernbert_training import training_frame, valid_config

try:
    import torch
except ImportError:
    torch = None


def _tiny_backend():
    class TinyTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, texts, *, return_tensors=None, **_kwargs):
            if isinstance(texts, str):
                return {"input_ids": [max(1, len(texts) % 11)]}
            values = [[max(1, len(text) % 11)] for text in texts]
            return {
                "input_ids": torch.tensor(values, dtype=torch.float32)
                if return_tensors == "pt"
                else values
            }

        def save_pretrained(self, directory):
            (Path(directory) / "tokenizer_config.json").write_text("{}")

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(1, 3)
            self.config = SimpleNamespace(
                id2label={0: "easy", 1: "medium", 2: "hard"},
                label2id={"easy": 0, "medium": 1, "hard": 2},
            )

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            model = cls()
            weights = Path(str(source)) / "pytorch_model.bin"
            if weights.is_file():
                values = json.loads((weights.parent / "config.json").read_text())
                values["id2label"] = {
                    int(key): value for key, value in values["id2label"].items()
                }
                model.config = SimpleNamespace(**values)
                model.load_state_dict(
                    torch.load(weights, weights_only=True, map_location="cpu")
                )
            elif "id2label" in kwargs:
                model.config.id2label = kwargs["id2label"]
                model.config.label2id = kwargs["label2id"]
            return model

        def forward(self, input_ids):
            return SimpleNamespace(logits=self.classifier(input_ids))

        def save_pretrained(self, directory, **_kwargs):
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), directory / "pytorch_model.bin")
            (directory / "config.json").write_text(json.dumps(vars(self.config)))

    def scheduler_factory(*_args, **_kwargs):
        return SimpleNamespace(step=lambda: None)

    return (
        torch,
        torch.nn.functional,
        SimpleNamespace(__version__="test"),
        (TinyModel, TinyTokenizer),
        scheduler_factory,
    )


class ModernBertObjectiveCliParserTest(unittest.TestCase):
    def test_default_and_explicit_objectives(self) -> None:
        self.assertEqual(build_parser().parse_args([]).objective, "ordinal")
        for objective in ("ordinal", "pointwise", "pairwise"):
            self.assertEqual(
                build_parser().parse_args(["--objective", objective]).objective,
                objective,
            )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["--objective", "regret"])


@unittest.skipUnless(torch is not None, "optional torch dependency is not installed")
class ModernBertObjectiveCliCpuTest(unittest.TestCase):
    def _inputs(self, root: Path, *, complete_targets: bool = True) -> list[str]:
        config = {**valid_config(), "model_id": "offline-tiny-model", "max_length": 512}
        (root / "config.json").write_text(json.dumps(config))
        frame = training_frame()
        if complete_targets:
            for tier, success, quality in (
                ("low", [1, 0, 0], [1.0, 0.1, 0.2]),
                ("medium", [1, 1, 0], [1.0, 0.8, 0.3]),
                ("high", [1, 1, 1], [1.0, 0.9, 0.9]),
            ):
                frame[f"success_{tier}"] = success
                frame[f"quality_{tier}"] = quality
        frame.to_parquet(root / "training.parquet", index=False)
        return [
            "modernbert-train",
            "--config",
            str(root / "config.json"),
            "--training-path",
            str(root / "training.parquet"),
            "--output-dir",
            str(root / "output"),
            "--device",
            "cpu",
        ]

    def _patches(self, arguments: list[str], backend):
        patches = ExitStack()
        patches.enter_context(patch.object(sys, "argv", arguments))
        patches.enter_context(redirect_stdout(io.StringIO()))
        patches.enter_context(
            patch(
                "llm_router_labeling.training.modernbert_training_cli.load_runtime_contract",
                return_value=object(),
            )
        )
        patches.enter_context(
            patch(
                "llm_router_labeling.training.modernbert_training._require_training_backends",
                return_value=backend,
            )
        )
        patches.enter_context(
            patch(
                "llm_router_labeling.training.modernbert_training.collect_runtime_versions",
                return_value={},
            )
        )
        patches.enter_context(
            patch(
                "llm_router_labeling.training.modernbert_training.validate_runtime_contract",
                return_value={
                    "runtime_version": "offline-cpu-fixture",
                    "contract_passed": True,
                },
            )
        )
        return patches

    def test_cli_runs_selected_loss_and_preserves_default_ordinal_artifact(
        self,
    ) -> None:
        for objective in ("ordinal", "pointwise", "pairwise"):
            with (
                self.subTest(objective=objective),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                arguments = self._inputs(root)
                if objective != "ordinal":
                    arguments.extend(["--objective", objective])
                with (
                    self._patches(arguments, _tiny_backend()),
                    patch(
                        "llm_router_labeling.objective_comparison.objectives.primary_row_losses",
                        wraps=primary_row_losses,
                    ) as losses,
                    patch(
                        "torch.nn.functional.cross_entropy",
                        wraps=torch.nn.functional.cross_entropy,
                    ) as cross_entropy,
                ):
                    main()
                output_config = json.loads(
                    (root / "output" / "config.json").read_text()
                )
                training_config = json.loads(
                    (root / "output" / "training_config.json").read_text()
                )
                if objective == "ordinal":
                    losses.assert_not_called()
                    self.assertTrue(cross_entropy.called)
                    self.assertNotIn("objective_method", training_config)
                    self.assertNotIn("router_objective", output_config)
                    self.assertEqual(
                        output_config["id2label"],
                        {"0": "easy", "1": "medium", "2": "hard"},
                    )
                else:
                    self.assertTrue(losses.called)
                    self.assertEqual(
                        {call.args[0] for call in losses.call_args_list}, {objective}
                    )
                    cross_entropy.assert_not_called()
                    self.assertEqual(training_config["objective_method"], objective)
                    self.assertEqual(output_config["router_objective"], objective)
                    self.assertEqual(
                        output_config["id2label"],
                        {"0": "low", "1": "medium", "2": "high"},
                    )

    def test_cli_does_not_bypass_anchor_objective_mismatch(self) -> None:
        for objective, anchor_objective in (
            ("ordinal", "pointwise"),
            ("pointwise", "ordinal"),
            ("pairwise", "pointwise"),
        ):
            with (
                self.subTest(objective=objective),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                arguments = self._inputs(root)
                backend = _tiny_backend()
                model = backend[3][0]()
                model.config.router_objective = anchor_objective
                model.save_pretrained(root / "anchor")
                arguments.extend(
                    [
                        "--objective",
                        objective,
                        "--initialization-mode",
                        "approved_anchor",
                        "--initial-model-path",
                        str(root / "anchor"),
                    ]
                )
                with (
                    self._patches(arguments, backend),
                    self.assertRaisesRegex(
                        ModernBertTrainingError, "checkpoint objective mismatch"
                    ),
                ):
                    main()
                self.assertFalse((root / "output").exists())

    def test_nonordinal_cli_still_rejects_unobserved_targets(self) -> None:
        for objective in ("pointwise", "pairwise"):
            with (
                self.subTest(objective=objective),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                arguments = self._inputs(root, complete_targets=False)
                arguments.extend(["--objective", objective])
                with (
                    self._patches(arguments, _tiny_backend()),
                    self.assertRaisesRegex(
                        ModernBertTrainingError, f"Missing observed {objective} targets"
                    ),
                ):
                    main()
                self.assertFalse((root / "output").exists())

    def test_training_docker_inventory_imports_and_executes_both_losses(self) -> None:
        repo = Path(__file__).resolve().parents[2]
        dockerfile = (repo / "ml/docker/modernbert-training.Dockerfile").read_text()
        with tempfile.TemporaryDirectory() as directory:
            staged = Path(directory)
            for line in dockerfile.splitlines():
                if line.startswith("COPY ml/src/"):
                    _, source, _ = line.split()
                    target = staged / Path(source).relative_to("ml/src")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(repo / source, target)
            script = """
import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
import torch
from setuptools import find_namespace_packages
from llm_router_labeling.objective_comparison import objectives
from llm_router_labeling.training.modernbert_training_cli import build_parser
assert pathlib.Path(objectives.__file__).is_relative_to(sys.argv[1])
assert 'llm_router_labeling.objective_comparison.data' not in sys.modules
assert 'llm_router_labeling.objective_comparison' in find_namespace_packages(sys.argv[1])
for method in ('pointwise', 'pairwise'):
    assert build_parser().parse_args(['--objective', method]).objective == method
    logits = torch.zeros((2, 3), requires_grad=True)
    losses = objectives.primary_row_losses(
        method, logits, difficulty_ids=None,
        success_targets=torch.tensor([[1., 0., 1.], [0., 1., 1.]]),
        quality_targets=torch.tensor([[0., 0.5, 1.], [1., 0.5, 0.]]),
    )
    assert losses.shape == (2,) and torch.isfinite(losses).all()
    losses.mean().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
print(json.dumps({'container_inventory_import_and_losses': 'passed'}))
"""
            result = subprocess.run(
                [sys.executable, "-I", "-c", script, str(staged)],
                cwd=staged,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(
                json.loads(result.stdout)["container_inventory_import_and_losses"],
                "passed",
            )


if __name__ == "__main__":
    unittest.main()
