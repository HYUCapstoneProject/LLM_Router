"""Contract tests use synthetic rows only; no research training/cloud calls."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "notebooks/r1_to_r0_transfer/experiment.py"
spec = importlib.util.spec_from_file_location("transfer_experiment", HELPER)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def frame(role, n=9):
    return pd.DataFrame({
        "sample_id": [f"{role}-{i}" for i in range(n)],
        "duplicate_group_id": [f"group-{role}-{i}" for i in range(n)],
        "model_input_text": [f"question {role} {i}" for i in range(n)],
        "difficulty_label": [experiment.LABELS[i % 3] for i in range(n)],
        "source_dataset": ["source-a" if i % 2 else "source-b" for i in range(n)],
        "category": ["reasoning"] * n,
    })


class TransferContracts(unittest.TestCase):
    def setUp(self):
        self.config = experiment.read(HELPER.parent / "experiment_config.json")
        self.frames = {role: frame(role, 300 if role == "feedback" else 99)
                       for role in experiment.ROLES}
        self.frames["feedback"]["difficulty_label"] = "hard"
        self.log = self.frames["feedback"][["sample_id"]].assign(initial_label="easy", final_label="hard")

    def test_cross_role_identity_and_text_leakage_rejected(self):
        experiment.validate_frames(self.frames)
        self.frames["dev"].loc[0, "model_input_text"] = "  QUESTION TRAIN 0  "
        with self.assertRaisesRegex(ValueError, "Duplicate/leakage"):
            experiment.validate_frames(self.frames)

    def test_duplicate_groups_rejected(self):
        self.frames["dev"].loc[0, "duplicate_group_id"] = self.frames["train"].iloc[0].duplicate_group_id
        with self.assertRaisesRegex(ValueError, "duplicate_group_id"):
            experiment.validate_frames(self.frames)

    def test_shared_passage_groups_stay_in_same_evaluation_role(self):
        self.frames["routerarena"].loc[1, "duplicate_group_id"] = self.frames["routerarena"].iloc[0].duplicate_group_id
        experiment.validate_frames(self.frames)
        self.frames["routerarena"].loc[1, "duplicate_group_id"] = self.frames["train"].iloc[0].duplicate_group_id
        with self.assertRaisesRegex(ValueError, "duplicate_group_id"):
            experiment.validate_frames(self.frames)

    def test_same_order_and_no_future_feedback(self):
        phases = experiment.make_phases(self.config, self.frames, self.log)
        shuffled = {k: v.sample(frac=1, random_state=7) for k, v in self.frames.items()}
        # Stable feedback selection and rehearsal do not depend on source row order.
        again = experiment.make_phases(self.config, shuffled, self.log.sample(frac=1, random_state=8))
        for phase in list(phases)[1:]:
            pd.testing.assert_frame_equal(phases[phase], again[phase])
        ordered = experiment.stable_sample(self.frames["feedback"], 300, self.config["stream_seed"])
        future = set(ordered.iloc[50:].sample_id)
        first = phases["cycle-01"]
        self.assertFalse(set(first.sample_id) & future)
        self.assertEqual(int(first.split.eq("train").sum()), 32 + 96)
        self.assertLessEqual(int(phases["cycle-06"].split.eq("train").sum()), 32 + 64 + 96)
        for data in phases.values():
            self.assertTrue(data.sample_weight.eq(1).all())
            self.assertFalse(set(data.sample_id) & set(self.frames["gate"].sample_id))

    def test_short_window_fails_instead_of_changing_budget(self):
        self.log["initial_label"] = "hard"
        with self.assertRaisesRegex(ValueError, "eligible rows"):
            experiment.make_phases(self.config, self.frames, self.log)

    def test_log_labels_cannot_silently_change_reference(self):
        self.log.loc[0, "final_label"] = "medium"
        with self.assertRaisesRegex(ValueError, "proxy labels"):
            experiment.make_phases(self.config, self.frames, self.log)

    def test_missing_real_source_pins_fails_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "Fill source pins"):
            experiment.load_inputs(HELPER.parent / "experiment_config.json",
                                   HELPER.parent / "source_pins.example.json", Path("unused"))

    def test_frozen_policy_and_arena_semantics(self):
        data = frame("eval", 2)
        logits = np.array([[0., 1., 2.], [0., 1., 2.]])
        policy = {"sources": {x: {"lambda": 1., "decision_costs": [0., .9, 1.]}
                              for x in data.source_dataset}}
        np.testing.assert_array_equal(experiment.predictions(logits, data, "r1_frozen", policy, "dev"), [0, 0])
        np.testing.assert_array_equal(experiment.predictions(logits, data, "r1_frozen", policy, "routerarena"), [2, 2])
        np.testing.assert_array_equal(experiment.predictions(logits, data, "r0_continued", policy, "dev"), [2, 2])

    def test_modernbert_head_reset_preserves_encoder(self):
        import torch
        from transformers import ModernBertConfig, ModernBertForSequenceClassification
        config = ModernBertConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                                  num_hidden_layers=2, num_attention_heads=2, num_labels=3,
                                  pad_token_id=0, bos_token_id=1, eos_token_id=2,
                                  cls_token_id=1, sep_token_id=2,
                                  max_position_embeddings=64, local_attention=16,
                                  _attn_implementation="eager")
        config.router_objective = "pointwise"
        model = ModernBertForSequenceClassification(config)
        before = {k: v.clone() for k, v in model.model.state_dict().items()}
        old_head = model.classifier.weight.detach().clone()
        experiment.reset_r1_head(model, 44)
        for name, tensor in model.model.state_dict().items():
            self.assertTrue(torch.equal(before[name], tensor), name)
        self.assertFalse(torch.equal(old_head, model.classifier.weight))
        self.assertEqual(model.config.router_objective, "ordinal")
        self.assertEqual(model.config.id2label, {0: "easy", 1: "medium", 2: "hard"})

    def test_holdouts_require_lock_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = {"config": self.config, "pins": {"models": {"r1": {"path": tmp}}}}
            args = SimpleNamespace(run_dir=tmp, arm="r1_frozen", seed=0, phase="baseline",
                                   role="gate", allow_final_evaluation=False)
            with patch.object(experiment, "load_plan", return_value=plan), \
                    patch.object(experiment, "verify_source"), patch.object(experiment, "infer") as infer:
                with self.assertRaisesRegex(ValueError, "selection lock"):
                    experiment.score(args)
                infer.assert_not_called()

    def test_selection_rejects_predictions_from_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plan.json").write_text('{}')
            receipt = root / "scores/r1_to_r0/seed-42/cycle-06/dev/8192/receipt.json"
            experiment.exact_json(receipt, {"plan_sha256": experiment.sha(root / "plan.json"),
                                            "model_digest": "old"})
            with self.assertRaisesRegex(ValueError, "another plan"):
                experiment.dev_score(root, "r1_to_r0", 42, "cycle-06", "new")

    def test_shared_trainer_contract_and_resume_completed_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text('{}')
            runtime = ROOT / "config/modernbert_runtime_cu124_v1.json"
            phases = experiment.make_phases(self.config, self.frames, self.log)
            (root / "shared-data").mkdir()
            hashes = {}
            for phase, data in phases.items():
                path = root / "shared-data" / f"{phase}.parquet"
                data.to_parquet(path, index=False)
                hashes[phase] = experiment.sha(path)
            pins = {"models": {"r0": {"path": str(source), "files": {"config.json": experiment.sha(source / "config.json")},
                       "cohort": {"backbone_revision": "a" * 40}}},
                    "runtime_contract": {"path": str(runtime), "sha256": experiment.sha(runtime)}}
            plan = {"config": self.config, "pins": pins, "phase_sha256": hashes,
                    "phase_train_rows": {k: int(v.split.eq("train").sum()) for k,v in phases.items()},
                    "helper_sha256": experiment.sha(HELPER), "source_code_sha256": {}}
            experiment.exact_json(root / "plan.json", plan)
            experiment.exact_json(root / "plan.sha256.json", {"sha256": experiment.sha(root / "plan.json")})
            def fake_train(data, output, config, **kwargs):
                self.assertEqual(config.max_length, 512)
                self.assertEqual(config.max_train_steps, 200)
                self.assertEqual(kwargs["objective_method"], "ordinal")
                self.assertEqual(kwargs["initial_model_dir"], source)
                output.mkdir()
                (output / "model.safetensors").write_bytes(b"test fixture only")
                return SimpleNamespace(metrics={"optimization": {"optimizer_steps": 200, "training_seconds": 1.}})
            args = SimpleNamespace(run_dir=root, arm="r0_continued", seed=42, until_phase="bridge")
            with patch("llm_router_labeling.training.modernbert_training.run_training", side_effect=fake_train) as call:
                experiment.train(args)
                experiment.train(args)
                self.assertEqual(call.call_count, 1)
            experiment.exact_json(root / "selection-lock.json", {})
            with self.assertRaisesRegex(ValueError, "locked"):
                experiment.train(args)

    def test_notebook_code_and_clean_outputs(self):
        notebooks = sorted(HELPER.parent.glob("*.ipynb"))
        self.assertEqual(len(notebooks), 5)
        for path in notebooks:
            notebook = experiment.read(path)
            self.assertEqual(notebook["nbformat"], 4)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
                    compile("".join(cell["source"]), str(path), "exec")


if __name__ == "__main__":
    unittest.main()
