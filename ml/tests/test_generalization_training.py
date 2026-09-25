from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from llm_router_labeling.generalization.train_cli import main
from llm_router_labeling.generalization.training import (
    ARMS,
    plan_four_arms,
    run_four_arms,
)
from llm_router_labeling.objective_comparison.objectives import (
    objective_loss,
    regret_row_losses,
)
from llm_router_labeling.shared.dataset_io import sha256_file, write_json
from llm_router_labeling.training.modernbert_checkpointing import (
    CheckpointOptions,
    ModernBertCheckpointError,
)
from llm_router_labeling.training.modernbert_runtime import load_runtime_contract
from llm_router_labeling.training.modernbert_training import run_training

try:
    import torch
except ImportError:
    torch = None


def fixture(root, *, evaluation_seam=False):
    config = {
        "experiment_version": "generalization-cpu-fixture",
        "seed": 42,
        "arms": [
            {"name": name, "method": method, "view": view}
            for name, (method, view) in ARMS.items()
        ],
        "balancing": {"max_sample_weight": 3.0},
        "backbone": {
            "model_id": "offline-tiny-model",
            "revision": "a" * 40,
            "max_length": 16,
        },
        "training": {
            "max_optimizer_steps": 3,
            "initialization": "fresh-pinned-pretrained-only",
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "gradient_checkpointing": False,
            "learning_rate": 0.02,
            "weight_decay": 0.0,
            "warmup_ratio": 0.0,
            "precision": "fp32",
            "device": "cpu",
            "attention_implementation": "eager",
            "gradient_clip_norm": 1.0,
            "regret_lambdas": [0.0, 0.5, 2.0],
            "train_sampling_strategy": "shuffle",
            "checkpoint_selection": "fixed_budget_final",
            "checkpoint_every_optimizer_steps": 1,
        },
    }
    if evaluation_seam:
        config["evaluation"] = {
            "relative_cost_budgets": [0.0, 0.5, 1.0],
            "penalty_grid": [0.0, 1.0, 100.0],
            # This fixture exercises file and inference seams, not a selected
            # production budget. Predeclare the full range to ensure eligibility.
            "default_relative_cost_budget": 1.0,
            "selection_metric": "source_macro_locked_policy_quality_auc",
            "require_selection_budget_feasible": True,
            "lambda_selection_role": "calibration",
            "arm_selection_role": "selection",
            "task_holdout_requires_selection_lock": True,
        }
    frames = {}
    role_sizes = [("natural", 5), ("calibration", 3), ("selection", 3)]
    if evaluation_seam:
        role_sizes = [("natural", 5), ("calibration", 6), ("selection", 6),
                      ("task_holdout", 6)]
    for role, n in role_sizes:
        entries = []
        for i in range(n):
            row = {
                "sample_id": f"{role}-{i}",
                "model_input_text": f"{role} question {'x' * i}",
                "source_dataset": "a" if i < 3 else "b",
                "category": "t",
                "task_group_id": "unseen" if role == "task_holdout" else "t",
                "generalization_role": "train" if role == "natural" else role,
                "duplicate_group_id": f"group-{role}-{i}",
                "split": "train" if role == "natural" else role,
                "difficulty_label": ["easy", "medium", "hard"][i % 3],
                "sample_weight": 1.0,
            }
            for j, tier in enumerate(("low", "medium", "high")):
                row[f"quality_{tier}"] = (
                    j / 2 if evaluation_seam else float(((i + j) % 3) / 2)
                )
                row[f"success_{tier}"] = float((i + j) % 3 > 0)
                row[f"decision_cost_{tier}"] = j / 2
                row[f"actual_cost_{tier}"] = (j + 1) * 0.001
                row[f"model_{tier}"] = f"model-{j}"
            entries.append(row)
        frames[role] = pd.DataFrame(entries)
    frames["balanced"] = frames["natural"].copy()
    frames["balanced"]["sample_weight"] = [0.5, 0.5, 1.0, 1.0, 2.0]
    dataset = root / "dataset"
    dataset.mkdir()
    from llm_router_labeling.generalization.data import (
        SCHEMA_VERSION,
        canonical_config_sha256,
    )

    artifacts, roles = {}, {}
    stored_roles = [
        ("natural", "train"),
        ("calibration", "calibration"),
        ("selection", "selection"),
    ]
    if evaluation_seam:
        stored_roles.append(("task_holdout", "task_holdout"))
    for view, role in stored_roles:
        frame = frames[view]
        frame["natural_sample_weight"] = 1.0
        frame["balanced_sample_weight"] = (
            frames["balanced"].sample_weight.to_numpy() if role == "train" else 1.0
        )
        path = dataset / f"{role}.parquet"
        frame.to_parquet(path, index=False)
        artifacts[path.name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": len(frame),
        }
        roles[role] = {"file": path.name, "rows": len(frame)}
    frames["balanced"] = frames["natural"].copy()
    frames["balanced"]["sample_weight"] = frames["balanced"].balanced_sample_weight
    # Pre-holdout training must work without even downloading the held-out file.
    if not evaluation_seam:
        roles["task_holdout"] = {"file": "task_holdout.parquet", "rows": 1}
        artifacts["task_holdout.parquet"] = {"sha256": "a" * 64, "bytes": 1, "rows": 1}
    write_json(dataset / "config.json", config)
    write_json(dataset / "arms.json", {"selection_role": "train"})
    write_json(dataset / "split-audit.json", {"effective_holdout_tasks": ["unseen"]})
    pd.DataFrame({"model": ["tiny"]}).to_parquet(
        dataset / "candidate-statistics.parquet", index=False
    )
    for name in (
        "config.json",
        "arms.json",
        "split-audit.json",
        "candidate-statistics.parquet",
    ):
        path = dataset / name
        artifacts[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": canonical_config_sha256(config),
        "roles": roles,
        "artifacts": artifacts,
    }
    write_json(dataset / "manifest.json", manifest)
    write_json(
        dataset / "_SUCCESS",
        {
            "schema_version": SCHEMA_VERSION,
            "manifest_sha256": sha256_file(dataset / "manifest.json"),
        },
    )
    config_path = root / "config.json"
    write_json(config_path, config)
    runtime_path = root / "runtime.json"
    write_json(
        runtime_path,
        {
            "runtime_version": "fixture-v1",
            "allowed_python_minors": ["3.12"],
            **{
                f"{key}_version": "test"
                for key in (
                    "torch",
                    "transformers",
                    "numpy",
                    "pandas",
                    "pyarrow",
                    "cuda",
                )
            },
        },
    )
    return (
        config,
        manifest,
        frames,
        {
            "dataset_dir": dataset,
            "config_path": config_path,
            "runtime_config_path": runtime_path,
        },
    )


def tiny_backend():
    initial_weights = []

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, texts, *, return_tensors=None, **_kwargs):
            if isinstance(texts, str):
                return {"input_ids": [max(1, len(texts) % 13)]}
            ids = [[max(1, len(text) % 13)] for text in texts]
            return {
                "input_ids": torch.tensor(ids, dtype=torch.float32)
                if return_tensors == "pt"
                else ids
            }

        def save_pretrained(self, path):
            write_json(Path(path) / "tokenizer_config.json", {})

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(1, 3)
            self.dropout = torch.nn.Dropout(0.2)
            self.config = SimpleNamespace(
                id2label={0: "easy", 1: "medium", 2: "hard"},
                label2id={"easy": 0, "medium": 1, "hard": 2},
            )

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            model = cls()
            if str(source) == "offline-tiny-model":
                model.config.id2label = kwargs["id2label"]
                model.config.label2id = kwargs["label2id"]
                initial_weights.append(
                    {k: v.clone() for k, v in model.state_dict().items()}
                )
            else:
                path = Path(source)
                cfg = json.loads((path / "config.json").read_text())
                cfg["id2label"] = {int(k): v for k, v in cfg["id2label"].items()}
                model.config = SimpleNamespace(**cfg)
                model.load_state_dict(
                    torch.load(
                        path / "pytorch_model.bin",
                        weights_only=True,
                        map_location="cpu",
                    )
                )
            return model

        def forward(self, input_ids):
            return SimpleNamespace(logits=self.classifier(self.dropout(input_ids)))

        def save_pretrained(self, path, **_kwargs):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), path / "pytorch_model.bin")
            write_json(path / "config.json", vars(self.config))

    class Scheduler:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

        def state_dict(self):
            return {"steps": self.steps}

        def load_state_dict(self, values):
            self.steps = values["steps"]

    def scheduler(*_args, **_kwargs):
        return Scheduler()

    backend = (
        torch,
        torch.nn.functional,
        SimpleNamespace(__version__="test"),
        (Model, Tokenizer),
        scheduler,
    )
    return backend, initial_weights


@unittest.skipUnless(torch is not None, "optional torch dependency")
class GeneralizationTrainingTest(unittest.TestCase):
    def patches(self, config, manifest, frames, backend, *, mock_inputs=True):
        stack = ExitStack()
        if mock_inputs:
            stack.enter_context(
                patch(
                    "llm_router_labeling.generalization.training._inputs",
                    return_value=(config, manifest, frames),
                )
            )
        stack.enter_context(
            patch(
                "llm_router_labeling.training.modernbert_training._require_training_backends",
                return_value=backend,
            )
        )
        stack.enter_context(
            patch(
                "llm_router_labeling.generalization.training._require_transformers",
                return_value=(*backend[3], backend[4]),
            )
        )
        stack.enter_context(
            patch(
                "llm_router_labeling.training.modernbert_training.collect_runtime_versions",
                return_value={
                    "python_minor": "3.12",
                    "torch_base_version": "test",
                    "transformers_version": "test",
                    "numpy_version": "test",
                    "pandas_version": "test",
                    "pyarrow_version": "test",
                    "cuda_version": None,
                    "cuda_available": False,
                },
            )
        )
        return stack

    def test_r3_matches_gap_weighted_ce_including_cost_switches_and_ties(self):
        logits = torch.tensor(
            [[0.1, 0.4, -0.2], [0.3, 0.2, 0.4]], dtype=torch.float64, requires_grad=True
        )
        quality = torch.tensor([[0.2, 0.7, 1.0], [0.5, 0.5, 0.5]], dtype=torch.float64)
        costs = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
        penalties = [0.0, 0.5, 2.0]
        expected = []
        for i in range(2):
            terms = []
            for penalty in penalties:
                utility = quality[i] - penalty * costs[i]
                ordered = utility.sort(descending=True).values
                gap = max(float(ordered[0] - ordered[1]), 1e-4)
                terms.append(
                    torch.nn.functional.cross_entropy(
                        (logits[i] - penalty * costs[i])[None, :],
                        utility.argmax()[None],
                    )
                    * gap
                )
            expected.append(torch.stack(terms).mean())
        expected = torch.stack(expected)
        actual = regret_row_losses(
            logits,
            quality_targets=quality,
            decision_costs=costs,
            regret_lambdas=penalties,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
        weights = torch.tensor([0.5, 1.5], dtype=torch.float64)
        reduced = objective_loss(
            "regret",
            logits,
            difficulty_ids=None,
            success_targets=None,
            quality_targets=quality,
            decision_costs=costs,
            sample_weights=weights,
            regret_lambdas=penalties,
            pairwise_tie_weight=0.25,
        )
        torch.testing.assert_close(reduced, (expected * weights).sum() / weights.sum())
        gradient = torch.autograd.grad(reduced, logits)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(gradient[1].abs().sum() > 0)
        for bad in ([], [-0.1], [float("nan")], [float("inf")], [0.0, 0.0]):
            with self.subTest(grid=bad), self.assertRaises(ValueError):
                regret_row_losses(
                    logits,
                    quality_targets=quality,
                    decision_costs=costs,
                    regret_lambdas=bad,
                )

    def test_four_arms_same_init_budget_natural_eval_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest, frames, kwargs = fixture(root)
            backend, initial = tiny_backend()
            with self.patches(config, manifest, frames, backend, mock_inputs=False):
                # Dry-run must not load a model or create a training artifact.
                with patch(
                    "llm_router_labeling.training.modernbert_training._require_training_backends",
                    side_effect=AssertionError("model loading in plan"),
                ):
                    plan = plan_four_arms(**kwargs)
                    self.assertFalse(plan["task_holdout_accessed"])
                    with redirect_stdout(io.StringIO()) as output:
                        main(
                            [
                                "plan",
                                "--dataset-dir",
                                str(kwargs["dataset_dir"]),
                                "--config",
                                str(kwargs["config_path"]),
                                "--runtime-contract",
                                str(kwargs["runtime_config_path"]),
                            ]
                        )
                    self.assertEqual(
                        json.loads(output.getvalue())["arm_order"], list(ARMS)
                    )
                full = root / "full"
                result = run_four_arms(**kwargs, output_dir=full)
                self.assertTrue(result["comparison_complete"])
                self.assertEqual(len(initial), 4)
                for weights in initial[1:]:
                    for name, value in initial[0].items():
                        self.assertTrue(torch.equal(value, weights[name]))
                for prefix in ("r1", "r3"):
                    natural = torch.load(
                        full / f"{prefix}-natural/model/pytorch_model.bin",
                        weights_only=True,
                    )
                    balanced = torch.load(
                        full / f"{prefix}-balanced/model/pytorch_model.bin",
                        weights_only=True,
                    )
                    self.assertTrue(
                        any(
                            not torch.equal(natural[name], balanced[name])
                            for name in natural
                        )
                    )
                for arm in ARMS:
                    metrics = json.loads(
                        (full / arm / "model/metrics.json").read_text()
                    )
                    self.assertEqual(metrics["optimization"]["optimizer_steps"], 3)
                    self.assertEqual(
                        metrics["optimization"]["train_sampling_strategy"], "shuffle"
                    )
                    for role in ("calibration", "selection"):
                        scores = pd.read_parquet(full / arm / f"{role}-scores.parquet")
                        self.assertEqual(
                            scores.sample_id.tolist(), frames[role].sample_id.tolist()
                        )
                        self.assertTrue(scores.sample_weight.eq(1).all())
                        self.assertNotIn("model_input_text", scores)
                        self.assertEqual(set(scores.research_arm), {arm})
                # Interrupt R3 in the middle and exactly at the final checkpoint.
                for target_step in (2, 3):
                    interrupted = root / f"resume-{target_step}"

                    def stop(arm, path, target_step=target_step):
                        if (
                            arm == "r3-balanced"
                            and path.name == f"step-{target_step:08d}"
                        ):
                            raise RuntimeError("simulated interruption")

                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        run_four_arms(
                            **kwargs, output_dir=interrupted, on_checkpoint_saved=stop
                        )
                    run_four_arms(**kwargs, output_dir=interrupted)
                    for arm in ARMS:
                        expected = torch.load(
                            full / arm / "model/pytorch_model.bin", weights_only=True
                        )
                        actual = torch.load(
                            interrupted / arm / "model/pytorch_model.bin",
                            weights_only=True,
                        )
                        for name in expected:
                            self.assertTrue(
                                torch.equal(expected[name], actual[name]),
                                (arm, target_step, name),
                            )
                    resumed = json.loads(
                        (interrupted / "r3-balanced/model/metrics.json").read_text()
                    )
                    self.assertEqual(
                        resumed["optimization"]["resumed_from_step"], target_step
                    )
                    self.assertEqual(resumed["optimization"]["optimizer_steps"], 3)
                # A changed R3 grid must not reuse its optimizer checkpoint.
                from llm_router_labeling.generalization.training import _training_config

                training = _training_config(
                    config, len(frames["balanced"]), "r3-balanced"
                )
                with self.assertRaises(ModernBertCheckpointError):
                    run_training(
                        full / "r3-balanced/training.parquet",
                        root / "wrong-grid",
                        training,
                        runtime_contract=load_runtime_contract(
                            kwargs["runtime_config_path"]
                        ),
                        objective_method="regret",
                        regret_lambdas=[0.0, 1.0],
                        checkpoint_options=CheckpointOptions(
                            checkpoint_dir=root / "unused",
                            save_every_optimizer_steps=1,
                            resume_from=full / "r3-balanced/checkpoints/step-00000003",
                        ),
                    )
                # Even a fully completed root must revalidate child score files.
                with (full / "r1-natural/selection-scores.parquet").open(
                    "ab"
                ) as stream:
                    stream.write(b"tamper")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    run_four_arms(**kwargs, output_dir=full)

    def test_real_four_arm_training_lock_cpu_holdout_and_report_seam(self):
        from llm_router_labeling.generalization import evaluation as ev

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest, frames, kwargs = fixture(root, evaluation_seam=True)
            backend, initial = tiny_backend()
            dataset = kwargs["dataset_dir"]
            holdout_path = dataset / "task_holdout.parquet"
            # The sealed manifest contains the future file's real digest, but
            # training and selection cannot open or inspect the held-out rows.
            unavailable_holdout = root / "not-downloaded-task-holdout.parquet"
            holdout_path.rename(unavailable_holdout)
            dataset_sha = sha256_file(dataset / "manifest.json")
            runs, lock, scores, report = (root / name for name in
                                         ("runs", "lock", "scores", "report"))
            evaluation_args = {
                "dataset_dir": dataset,
                "runs_dir": runs,
                "config_path": kwargs["config_path"],
            }
            with self.patches(config, manifest, frames, backend, mock_inputs=False):
                run_four_arms(**kwargs, output_dir=runs)
                selected = ev.lock_selection(**evaluation_args, output_dir=lock)
                self.assertFalse(holdout_path.exists())
                self.assertFalse(selected["task_holdout_accessed"])
                self.assertFalse(selected["production_approved"])
                self.assertEqual(set(selected["policies"]), set(ARMS))
                self.assertEqual(set(selected["baselines"]), {"a", "b"})
                self.assertTrue(all(
                    set(policy) == {"a", "b"}
                    for policy in selected["policies"].values()
                ))
                self.assertIn(selected["selected_arm"], selected["eligible_arms"])
                lock_sha = sha256_file(lock / "manifest.json")
                model_shas = {
                    arm: sha256_file(runs / arm / "model/pytorch_model.bin")
                    for arm in ARMS
                }
                # Only the explicit inference stage receives the held-out file.
                unavailable_holdout.rename(holdout_path)
                with patch.object(
                    ev, "_require_transformers",
                    return_value=(*backend[3], backend[4]),
                ):
                    inferred = ev.infer_holdout(
                        **evaluation_args, lock_dir=lock, output_dir=scores,
                        device="cpu",
                    )
                self.assertEqual(len(initial), 4, "inference must reload final weights")
                self.assertEqual(inferred["selected_arm"], selected["selected_arm"])
                self.assertEqual(inferred["selection_lock_sha256"], lock_sha)
                self.assertTrue(inferred["task_holdout_accessed"])
                self.assertFalse(inferred["production_approved"])
                self.assertEqual(len(inferred["artifacts"]), 4)
                for arm, (method, _) in ARMS.items():
                    scored = pd.read_parquet(scores / f"{arm}-scores.parquet")
                    pd.testing.assert_frame_equal(
                        scored[frames["task_holdout"].drop(columns="model_input_text").columns],
                        frames["task_holdout"].drop(columns="model_input_text"),
                    )
                    self.assertEqual(set(scored.research_arm), {arm})
                    self.assertEqual(set(scored.objective_method), {method})
                    self.assertTrue(scored.sample_weight.eq(1.0).all())
                    # Independently reload the actual trained model to ensure
                    # the evaluator writes its logits, not hand-made fixtures.
                    model = backend[3][0].from_pretrained(runs / arm / "model")
                    model.eval()
                    encoded = backend[3][1]()(
                        frames["task_holdout"].model_input_text.tolist(),
                        return_tensors="pt",
                    )
                    with torch.no_grad():
                        expected = model(**encoded).logits
                    torch.testing.assert_close(
                        torch.tensor(scored[[f"score_{tier}" for tier in ev.TIERS]].to_numpy()),
                        expected,
                    )
                evaluated = ev.evaluate_holdout(
                    **evaluation_args, lock_dir=lock, scores_dir=scores,
                    output_dir=report,
                )
                self.assertEqual(evaluated["selected_arm"], selected["selected_arm"])
                self.assertFalse(evaluated["arm_reselected_on_holdout"])
                self.assertFalse(evaluated["production_approved"])
                self.assertEqual(evaluated["selection_lock_sha256"], lock_sha)
                self.assertEqual(set(evaluated["reports"]), set(ARMS))
                for arm, metrics in evaluated["reports"].items():
                    self.assertIsNotNone(metrics[ev.METRIC])
                    for source, source_metrics in metrics["per_source"].items():
                        self.assertEqual(source_metrics["rows"], 3)
                        for budget, point in source_metrics["budgets"].items():
                            self.assertEqual(
                                point["penalty"], selected["policies"][arm][source][budget]
                            )
                            self.assertEqual(set(point["per_task"]), {"unseen"})
                        baselines = source_metrics["train_fitted_baselines"]
                        self.assertEqual(
                            baselines["train_source_task_with_source_fallback"],
                            baselines["train_best_single_within_source_pool"],
                        )
                for sealed in (lock, scores, report):
                    self.assertEqual(
                        (sealed / "_SUCCESS").read_text().strip(),
                        sha256_file(sealed / "manifest.json"),
                    )
                self.assertEqual(sha256_file(lock / "manifest.json"), lock_sha)
                self.assertEqual(sha256_file(dataset / "manifest.json"), dataset_sha)
                for arm in ARMS:
                    self.assertEqual(
                        sha256_file(runs / arm / "model/pytorch_model.bin"),
                        model_shas[arm],
                    )
                # A changed model is rejected before any holdout read or model
                # construction, even though the pre-holdout lock still exists.
                with (runs / "r3-balanced/model/pytorch_model.bin").open("ab") as stream:
                    stream.write(b"tamper")
                with patch.object(ev, "load_role") as load_role, patch.object(
                    ev, "_require_transformers"
                ) as require_backend:
                    with self.assertRaisesRegex(ValueError, "checksum"):
                        ev.infer_holdout(
                            **evaluation_args, lock_dir=lock,
                            output_dir=root / "rejected-scores", device="cpu",
                        )
                    load_role.assert_not_called()
                    require_backend.assert_not_called()
                self.assertFalse((root / "rejected-scores").exists())

    def test_two_partial_sessions_merge_without_retraining_and_reject_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest, frames, kwargs = fixture(root)
            backend, initial = tiny_backend()
            with self.patches(config, manifest, frames, backend, mock_inputs=False):
                first, second, merged = root / "first", root / "second", root / "merged"
                left = ["r1-natural", "r3-natural"]
                right = ["r1-balanced", "r3-balanced"]
                for target, arms in ((first, left), (second, right)):
                    result = run_four_arms(**kwargs, output_dir=target, arms_to_run=arms)
                    self.assertFalse(result["comparison_complete"])
                    self.assertEqual(set(result["completed_arms"]), set(arms))
                    self.assertFalse((target / "_SUCCESS").exists())
                    self.assertFalse((target / "manifest.json").exists())
                    self.assertTrue((target / "partial-manifest.json").is_file())
                    self.assertFalse(any((target / arm).exists() for arm in ARMS if arm not in arms))
                self.assertEqual((first / "run-plan.json").read_bytes(),
                                 (second / "run-plan.json").read_bytes())
                self.assertEqual(len(initial), 4)
                merged.mkdir()
                shutil.copy2(first / "run-plan.json", merged / "run-plan.json")
                for source, arms in ((first, left), (second, right)):
                    for arm in arms:
                        shutil.copytree(source / arm, merged / arm)
                with patch(
                    "llm_router_labeling.generalization.training.run_training",
                    side_effect=AssertionError("Completed arm retrained during merge"),
                ):
                    result = run_four_arms(**kwargs, output_dir=merged)
                    self.assertTrue(result["comparison_complete"])
                    self.assertEqual(result["arms"], list(ARMS))
                    self.assertTrue((merged / "_SUCCESS").is_file())
                with (merged / "r3-balanced/selection-scores.parquet").open("ab") as handle:
                    handle.write(b"tampered")
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    run_four_arms(**kwargs, output_dir=merged, arms_to_run=left)
                for invalid in ([], ["r0-natural"], ["r1-natural", "r1-natural"]):
                    with self.assertRaisesRegex(ValueError, "unique known"):
                        run_four_arms(**kwargs, output_dir=root / "invalid", arms_to_run=invalid)

    def test_reject_difficulty_class_balancing_and_changed_resume_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest, frames, kwargs = fixture(root)
            backend, _ = tiny_backend()
            with self.patches(config, manifest, frames, backend):
                config["training"]["train_sampling_strategy"] = (
                    "balanced_effective_batch"
                )
                with self.assertRaisesRegex(ValueError, "never difficulty balancing"):
                    plan_four_arms(**kwargs)
                config["training"]["train_sampling_strategy"] = "shuffle"
                output = root / "partial"
                output.mkdir()
                write_json(output / "run-plan.json", plan_four_arms(**kwargs))
                config["training"]["regret_lambdas"] = [0.0, 1.0]
                with self.assertRaisesRegex(ValueError, "Resume contract changed"):
                    run_four_arms(**kwargs, output_dir=output)


if __name__ == "__main__":
    unittest.main()
