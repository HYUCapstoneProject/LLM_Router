"""Contract tests for the F0/F1/F2 retraining-effect harness; synthetic rows only."""
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
HELPER = ROOT / "notebooks/retraining_effect/experiment.py"
spec = importlib.util.spec_from_file_location("retraining_effect_experiment", HELPER)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def frame(role, n=99, outcomes=True):
    data = pd.DataFrame({
        "sample_id": [f"{role}-{i}" for i in range(n)],
        "duplicate_group_id": [f"group-{role}-{i}" for i in range(n)],
        "model_input_text": [f"question {role} {i}" for i in range(n)],
        "difficulty_label": [experiment.LABELS[i % 3] for i in range(n)],
        "source_dataset": ["source-a" if i % 2 else "source-b" for i in range(n)],
        "category": ["reasoning"] * n,
    })
    if outcomes:
        for tier, cost in zip(experiment.TIERS, (.001, .01, .1)):
            data[f"success_{tier}"] = 1.0
            data[f"quality_{tier}"] = 1.0
            data[f"actual_cost_{tier}"] = cost
    return data


class RetrainingEffectContracts(unittest.TestCase):
    def setUp(self):
        self.config = experiment.read(HELPER.parent / "experiment_config.json")
        self.frames = {role: frame(role, 600 if role == "feedback" else 99, role in experiment.OUTCOME_ROLES)
                       for role in experiment.INPUT_ROLES}
        feedback = self.frames["feedback"]
        # Low tier fails everywhere; medium solves two thirds, high solves the rest.
        feedback["success_low"] = 0.0
        feedback["quality_low"] = 0.0
        feedback["success_medium"] = [0.0 if i % 3 == 0 else 1.0 for i in range(len(feedback))]
        self.log = feedback[["sample_id"]].assign(initial_label="easy", logit_easy=1., logit_medium=0., logit_hard=0.)

    def test_simulator_replays_only_called_tiers(self):
        data = frame("sim", 5)
        cases = [(0, [1, 1, 1]), (0, [0, 1, 1]), (0, [0, 0, 0]), (2, [1, 1, 0]), (1, [0, 0, 1])]
        for row, (_, success) in enumerate(cases):
            for tier, value in zip(experiment.TIERS, success):
                data.loc[row, f"success_{tier}"] = float(value)
            data.loc[row, "quality_medium"] = .5
        sim = experiment.simulate(data, [c[0] for c in cases])
        self.assertEqual(sim.outcome.tolist(), ["first_try", "escalated_success", "fail_all", "fail_all", "escalated_success"])
        self.assertEqual(sim.resolved_tier.tolist(), [0, 1, -1, -1, 2])
        self.assertEqual(sim.call_depth.tolist(), [1, 2, 3, 1, 2])
        np.testing.assert_allclose(sim.simulated_cost, [.001, .011, .111, .1, .11])
        np.testing.assert_allclose(sim.initial_quality, [1., 1., 1., 1., .5])

    def test_updates_follow_operational_version_rule_without_future_rows(self):
        updates, stream, events, _ = experiment.make_updates(self.config, self.frames, self.log)
        window_of = dict(zip(stream.sample_id, stream.window))
        base_ids = set(self.frames["train"].sample_id)
        for update in range(1, 7):
            name = f"update-{update:02d}"
            cycle_start = 0 if update <= 3 else 3
            for arm in updates:
                data = updates[arm][name]
                train = data[data.split.eq("train")]
                feedback_windows = {window_of[x] for x in train.sample_id if x in window_of}
                self.assertTrue(all(w <= update for w in feedback_windows), (arm, name))
                self.assertEqual(int(data.split.eq("validation").sum()), 48)
                self.assertTrue(data.sample_weight.eq(1).all())
                self.assertFalse(set(data.sample_id) & set(self.frames["gate"].sample_id))
            f1 = updates["f1_cycle_only"][name]
            f1_train = f1[f1.split.eq("train")]
            self.assertEqual({window_of[x] for x in f1_train.sample_id}, set(range(cycle_start + 1, update + 1)))
            self.assertNotIn("easy", set(f1_train.difficulty_label))
            f2 = updates["f2_replay"][name]
            f2_train = f2[f2.split.eq("train")]
            self.assertEqual(len(set(f2_train.sample_id) & base_ids), 96)
            history = {window_of[x] for x in f2_train.sample_id if x in window_of} - set(range(cycle_start + 1, update + 1))
            self.assertEqual(history, set() if update <= 3 else {1, 2, 3})

    def test_feedback_label_is_resolved_tier_and_every_event_is_used(self):
        updates, stream, events, _ = experiment.make_updates(self.config, self.frames, self.log)
        self.assertEqual(len(events), len(self.frames["feedback"]))
        expected = np.where(self.frames["feedback"].set_index("sample_id").loc[events.sample_id].success_medium.eq(1), "medium", "hard")
        self.assertEqual(events.difficulty_label.tolist(), expected.tolist())
        first = updates["f1_cycle_only"]["update-01"]
        self.assertEqual(int(first.split.eq("train").sum()), int(stream.window.eq(1).sum()))
        self.assertGreater(int(first.split.eq("train").sum()), 32)
        self.assertTrue(first[first.split.eq("train")].label_origin.eq(experiment.FEEDBACK_ORIGIN).all())

    def test_order_independent_preparation(self):
        updates, _, _, _ = experiment.make_updates(self.config, self.frames, self.log)
        shuffled = {k: v.sample(frac=1, random_state=7).reset_index(drop=True) for k, v in self.frames.items()}
        again, _, _, _ = experiment.make_updates(self.config, shuffled, self.log.sample(frac=1, random_state=8))
        for arm in updates:
            for name in updates[arm]:
                pd.testing.assert_frame_equal(updates[arm][name], again[arm][name])

    def test_f4_preparation_is_order_independent(self):
        config = self.f4_setup()
        updates, _, _, _ = experiment.make_updates(config, self.frames, self.log)
        shuffled = {k: v.sample(frac=1, random_state=3).reset_index(drop=True) for k, v in self.frames.items()}
        again, _, _, _ = experiment.make_updates(config, shuffled, self.log.sample(frac=1, random_state=4))
        for name in updates["f4_balanced_easy"]:
            pd.testing.assert_frame_equal(updates["f4_balanced_easy"][name], again["f4_balanced_easy"][name])

    def test_short_window_fails_before_fitting(self):
        self.frames["feedback"]["success_low"] = 1.0
        with self.assertRaisesRegex(ValueError, "fewer than 32"):
            experiment.make_updates(self.config, self.frames, self.log)

    def test_f1_matches_f2_optimizer_steps(self):
        updates, _, _, _ = experiment.make_updates(self.config, self.frames, self.log)
        budget = experiment.training_budget(self.config, updates)
        for name in updates["f2_replay"]:
            f1, f2 = budget["f1_cycle_only"][name], budget["f2_replay"][name]
            self.assertEqual(f2["epochs"], 3)
            self.assertEqual(f1["max_train_steps"], f2["max_train_steps"])
            self.assertEqual(f2["max_train_steps"], experiment.steps_per_epoch(f2["train_rows"], self.config) * 3)
            self.assertGreaterEqual(experiment.steps_per_epoch(f1["train_rows"], self.config) * f1["epochs"], f1["max_train_steps"])

    def test_parent_schedule(self):
        self.assertEqual([experiment.parent_update(i, self.config) for i in range(1, 7)],
                         ["anchor"] * 3 + ["update-03"] * 3)

    def test_missing_outcomes_are_not_imputed(self):
        experiment.validate_frames(self.frames)
        self.frames["gate"].loc[0, "success_high"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            experiment.validate_frames(self.frames)

    def test_feedback_cap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = dict(self.config, feedback_per_window_cap=32)
            (Path(tmp) / "config.json").write_text(json.dumps(config))
            (Path(tmp) / "pins.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "no per-window cap"):
                experiment.load_inputs(Path(tmp) / "config.json", Path(tmp) / "pins.json", Path(tmp))

    def test_train_restarts_cycle_from_its_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text("{}")
            runtime = ROOT / "config/modernbert_runtime_cu124_v1.json"
            updates, _, _, _ = experiment.make_updates(self.config, self.frames, self.log)
            hashes = {}
            for arm, phases in updates.items():
                for name, data in phases.items():
                    path = root / "shared-data" / arm / f"{name}.parquet"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    data.to_parquet(path, index=False)
                    hashes[f"{arm}/{name}"] = experiment.sha(path)
            pins = {"models": {"r0": {"path": str(source), "files": {"config.json": experiment.sha(source / "config.json")},
                                      "cohort": {"backbone_revision": "a" * 40}}},
                    "runtime_contract": {"path": str(runtime), "sha256": experiment.sha(runtime)}}
            plan = {"config": self.config, "pins": pins, "data_sha256": hashes,
                    "training_budget": experiment.training_budget(self.config, updates),
                    "helper_sha256": experiment.sha(HELPER), "source_code_sha256": {}}
            experiment.exact_json(root / "plan.json", plan)
            experiment.exact_json(root / "plan.sha256.json", {"sha256": experiment.sha(root / "plan.json")})
            seen = {}

            def fake_train(data, output, config, **kwargs):
                seen[output.parent.name] = (kwargs["initial_model_dir"], config.max_train_steps, config.epochs)
                self.assertEqual(kwargs["objective_method"], "ordinal")
                self.assertEqual(config.max_length, 512)
                output.mkdir()
                (output / "model.safetensors").write_bytes(output.parent.name.encode())
                return SimpleNamespace(metrics={"optimization": {"optimizer_steps": config.max_train_steps,
                                                                 "training_seconds": 1.}})
            args = SimpleNamespace(run_dir=root, arm="f1_cycle_only", seed=42, until_update="update-06")
            with patch("llm_router_labeling.training.modernbert_training.run_training", side_effect=fake_train) as call:
                experiment.train(args)
                experiment.train(args)
                self.assertEqual(call.call_count, 6)
            arm_root = root / "arms/f1_cycle_only/seed-42"
            for name in ("update-01", "update-02", "update-03"):
                self.assertEqual(seen[name][0], source)
            for name in ("update-04", "update-05", "update-06"):
                self.assertEqual(seen[name][0], arm_root / "update-03" / "model")
            for name, (_, steps, epochs) in seen.items():
                self.assertEqual(steps, plan["training_budget"]["f1_cycle_only"][name]["max_train_steps"])
            experiment.exact_json(root / "final-lock.json", {})
            with self.assertRaisesRegex(ValueError, "locked"):
                experiment.train(args)

    def test_final_scoring_requires_lock_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = {"config": self.config, "pins": {}}
            args = SimpleNamespace(run_dir=tmp, arm="f0_frozen", seed=0, role="gate")
            with patch.object(experiment, "load_plan", return_value=plan), patch.object(experiment, "infer") as infer:
                with self.assertRaisesRegex(ValueError, "final-lock"):
                    experiment.score_final(args)
                infer.assert_not_called()

    def test_final_lock_requires_window_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plan.json").write_text("{}")
            for arm in experiment.run_arms(self.config):
                for seed in self.config["seeds"]:
                    directory = root / "arms" / arm / f"seed-{seed}" / "update-06"
                    (directory / "model").mkdir(parents=True)
                    (directory / "model" / "w").write_text("x")
                    experiment.exact_json(directory / "completed.json",
                                          {"model_files": experiment.model_inventory(directory / "model")})
            with patch.object(experiment, "load_plan", return_value={"config": self.config}):
                with self.assertRaisesRegex(ValueError, "prequential"):
                    experiment.final_lock(SimpleNamespace(run_dir=root))
            self.assertFalse((root / "final-lock.json").exists())

    def test_paired_bootstrap_on_constant_shift(self):
        data = frame("boot", 30)
        data["left"] = np.linspace(0, 1, 30)
        data["right"] = data.left + .1
        value, low, high = experiment.paired_bootstrap(data, "left", "right", 50, 1)
        self.assertAlmostEqual(value, .1)
        self.assertAlmostEqual(low, .1)
        self.assertAlmostEqual(high, .1)

    def test_label_metrics_count_over_escalation(self):
        data = frame("m", 6, outcomes=False)
        result = experiment.label_metrics(data, [2, 1, 2, 2, 1, 2])
        self.assertAlmostEqual(result["easy_to_hard_rate_all_rows"], 2 / 6)
        self.assertAlmostEqual(result["predicted_easy_share"], 0.)
        self.assertAlmostEqual(result["over_route_rate"], 2 / 6)

    def f4_setup(self):
        config = experiment.read(HELPER.parent / "experiment_config_f4.json")
        # Half of the requests are solved by the low tier first try (confirmed easy).
        self.frames["feedback"]["success_low"] = [1.0 if i % 2 else 0.0 for i in range(600)]
        return config

    def test_f4_balances_labels_with_fresh_arrived_easy_rows(self):
        config = self.f4_setup()
        updates, stream, events, supplement = experiment.make_updates(config, self.frames, self.log)
        self.assertEqual(list(updates), ["f4_balanced_easy"])
        window_of = dict(zip(stream.sample_id, stream.window))
        confirmed = set(stream[stream.outcome.eq("first_try") & stream.initial_tier.eq(0)].sample_id)
        used = set()
        for update in range(1, 7):
            name = f"update-{update:02d}"
            data = updates["f4_balanced_easy"][name]
            train = data[data.split.eq("train")]
            added = train[train.label_origin.eq(experiment.CONFIRMED_EASY_ORIGIN)]
            self.assertTrue(set(added.sample_id) <= confirmed)
            self.assertTrue(all(window_of[x] <= update for x in added.sample_id), name)
            self.assertFalse(set(added.sample_id) & used, f"easy rows reused in {name}")
            used |= set(added.sample_id)
            self.assertTrue(added.difficulty_label.eq("easy").all())
            counts = train.difficulty_label.value_counts()
            info = supplement[name]
            self.assertEqual(info["added_easy"], len(added))
            self.assertEqual(info["added_easy"] + info["shortfall"], info["target_easy"] - info["base_easy"])
            if info["shortfall"]:
                self.assertEqual(info["added_easy"], info["pool_available"])
            self.assertEqual(int(counts["easy"]), info["base_easy"] + info["added_easy"])
            self.assertEqual(info["target_easy"], (int(counts["medium"]) + int(counts["hard"]) + 1) // 2)
            feedback_rows = train[train.label_origin.eq(experiment.FEEDBACK_ORIGIN)]
            self.assertTrue(set(feedback_rows.sample_id) <= set(events.sample_id))
        budget = experiment.training_budget(config, updates)["f4_balanced_easy"]
        for name, data in updates["f4_balanced_easy"].items():
            rows = int(data.split.eq("train").sum())
            self.assertEqual(budget[name], {"train_rows": rows, "epochs": 3,
                                            "max_train_steps": experiment.steps_per_epoch(rows, config) * 3})

    def test_f4_records_shortfall_instead_of_reusing_easy_rows(self):
        config = self.f4_setup()
        self.frames["feedback"]["success_low"] = [1.0 if i % 25 == 1 else 0.0 for i in range(600)]
        updates, _, _, supplement = experiment.make_updates(config, self.frames, self.log)
        self.assertGreater(sum(x["shortfall"] for x in supplement.values()), 0)
        seen = []
        for data in updates["f4_balanced_easy"].values():
            seen += data[data.label_origin.eq(experiment.CONFIRMED_EASY_ORIGIN)].sample_id.tolist()
        self.assertEqual(len(seen), len(set(seen)))

    def test_arm_selection_is_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "pins.json").write_text("{}")
            for arms, message in [(["f1_cycle_only"], "prepare them together"), (["f9"], "Unknown")]:
                config = dict(self.config, arms=arms)
                (Path(tmp) / "config.json").write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, message):
                    experiment.load_inputs(Path(tmp) / "config.json", Path(tmp) / "pins.json", Path(tmp))

    def test_reference_run_must_share_stream_and_logger(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp)
            (ref / "shared-data").mkdir()
            stream = pd.DataFrame({"sample_id": ["a", "b"], "window": [1, 2], "initial_label": ["easy", "hard"],
                                   "outcome": ["first_try", "fail_all"], "resolved_tier": [0, -1]})
            stream.to_parquet(ref / "shared-data" / "stream-windows.parquet", index=False)
            stream_sha = experiment.sha(ref / "shared-data" / "stream-windows.parquet")
            ref_plan = {"config": self.config, "logger_receipt": {"log_sha256": "a"},
                        "data_sha256": {"stream-windows": stream_sha}}
            experiment.exact_json(ref / "plan.json", ref_plan)
            experiment.exact_json(ref / "plan.sha256.json", {"sha256": experiment.sha(ref / "plan.json")})
            current = {"config": self.config, "logger_receipt": {"log_sha256": "a"},
                       "data_sha256": {"stream-windows": stream_sha}}
            sources = experiment.reference_sources(current, ref, stream)
            self.assertEqual(set(sources), {"f1_cycle_only", "f2_replay"})
            changed = stream.assign(window=[1, 1])
            with self.assertRaisesRegex(ValueError, "different stream"):
                experiment.reference_sources(current, ref, changed)
            current["logger_receipt"] = {"log_sha256": "b"}
            with self.assertRaisesRegex(ValueError, "different stream"):
                experiment.reference_sources(current, ref)

    def test_notebook_code_and_clean_outputs(self):
        notebooks = sorted(HELPER.parent.glob("*.ipynb"))
        self.assertEqual(len(notebooks), 7)
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
