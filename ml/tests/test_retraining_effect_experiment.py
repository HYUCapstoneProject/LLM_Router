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

    def test_f3_preparation_is_order_independent(self):
        config = self.f3_setup()
        updates, _, _, _ = experiment.make_updates(config, self.frames, self.log)
        shuffled = {k: v.sample(frac=1, random_state=3).reset_index(drop=True) for k, v in self.frames.items()}
        again, _, _, _ = experiment.make_updates(config, shuffled, self.log.sample(frac=1, random_state=4))
        for name in updates["f3_balanced_easy"]:
            pd.testing.assert_frame_equal(updates["f3_balanced_easy"][name], again["f3_balanced_easy"][name])

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

    def f3_setup(self):
        config = experiment.read(HELPER.parent / "experiment_config_f3.json")
        # Half of the requests are solved by the low tier first try (confirmed easy).
        self.frames["feedback"]["success_low"] = [1.0 if i % 2 else 0.0 for i in range(600)]
        return config

    def test_f3_balances_labels_with_fresh_arrived_easy_rows(self):
        config = self.f3_setup()
        updates, stream, events, supplement = experiment.make_updates(config, self.frames, self.log)
        self.assertEqual(list(updates), ["f3_balanced_easy"])
        window_of = dict(zip(stream.sample_id, stream.window))
        confirmed = set(stream[stream.outcome.eq("first_try") & stream.initial_tier.eq(0)].sample_id)
        used = set()
        for update in range(1, 7):
            name = f"update-{update:02d}"
            data = updates["f3_balanced_easy"][name]
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
        budget = experiment.training_budget(config, updates)["f3_balanced_easy"]
        for name, data in updates["f3_balanced_easy"].items():
            rows = int(data.split.eq("train").sum())
            self.assertEqual(budget[name], {"train_rows": rows, "epochs": 3,
                                            "max_train_steps": experiment.steps_per_epoch(rows, config) * 3})

    def test_f3_records_shortfall_instead_of_reusing_easy_rows(self):
        config = self.f3_setup()
        self.frames["feedback"]["success_low"] = [1.0 if i % 25 == 1 else 0.0 for i in range(600)]
        updates, _, _, supplement = experiment.make_updates(config, self.frames, self.log)
        self.assertGreater(sum(x["shortfall"] for x in supplement.values()), 0)
        seen = []
        for data in updates["f3_balanced_easy"].values():
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

    def f4_setup(self):
        config = experiment.read(HELPER.parent / "experiment_config_f4.json")
        # Half of the requests are routed easy (they escalate); the rest alternate medium/hard.
        n = len(self.frames["feedback"])
        self.log["initial_label"] = ["easy" if i % 2 == 0 else ("medium" if i % 4 == 1 else "hard") for i in range(n)]
        return config

    def test_f4_learns_both_directions_with_original_labels_and_f2_rehearsal(self):
        config = self.f4_setup()
        updates, stream, _, _ = experiment.make_updates(config, self.frames, self.log)
        self.assertEqual(list(updates), ["f4_bidirectional"])
        window_of = dict(zip(stream.sample_id, stream.window))
        original = dict(zip(self.frames["feedback"].sample_id, self.frames["feedback"].difficulty_label))
        logged = dict(zip(self.log.sample_id, self.log.initial_label))
        base_ids = set(self.frames["train"].sample_id)
        directions = set()
        for update in range(1, 7):
            name = f"update-{update:02d}"
            cycle_start = 0 if update <= 3 else 3
            data = updates["f4_bidirectional"][name]
            train = data[data.split.eq("train")]
            picked = train[train.label_origin.eq(experiment.DISAGREEMENT_ORIGIN)]
            for sample_id, label in zip(picked.sample_id, picked.difficulty_label):
                self.assertEqual(label, original[sample_id])
                self.assertNotEqual(logged[sample_id], original[sample_id])
                directions.add(experiment.LABELS.index(logged[sample_id]) > experiment.LABELS.index(label))
            windows = {window_of[x] for x in picked.sample_id}
            self.assertTrue(all(w <= update for w in windows), name)
            current = {x for x in picked.sample_id if window_of[x] > cycle_start}
            expected = {x for x in stream[stream.window.between(cycle_start + 1, update)].sample_id if logged[x] != original[x]}
            self.assertEqual(current, expected)
            history = picked[[window_of[x] <= cycle_start for x in picked.sample_id]]
            self.assertTrue(history.difficulty_label.value_counts().le(32).all())
            self.assertEqual(len(set(train.sample_id) & base_ids), 96)
            self.assertEqual(int(data.split.eq("validation").sum()), 48)
        self.assertEqual(directions, {True, False})
        budget = experiment.training_budget(config, updates)["f4_bidirectional"]
        for name, data in updates["f4_bidirectional"].items():
            rows = int(data.split.eq("train").sum())
            self.assertEqual(budget[name], {"train_rows": rows, "epochs": 3,
                                            "max_train_steps": experiment.steps_per_epoch(rows, config) * 3})
        summary = experiment.selection_summary(stream, config)
        self.assertEqual(sum(sum(v.values()) for v in summary["per_window"].values()),
                         sum(logged[x] != original[x] for x in stream.sample_id))

    def test_evaluation_roles_may_share_text_but_never_with_training_roles(self):
        frames = dict(self.frames, test=frame("test", 30))
        frames["test"].loc[0, "model_input_text"] = frames["routerarena"].model_input_text.iloc[0]
        overlaps = experiment.validate_frames(frames)
        self.assertEqual(overlaps, {"test~routerarena:text": 1})
        frames["test"].loc[1, "model_input_text"] = frames["train"].model_input_text.iloc[0]
        with self.assertRaisesRegex(ValueError, "leakage in test"):
            experiment.validate_frames(frames)

    def test_extra_inputs_must_be_declared_and_pinned(self):
        config = experiment.read(HELPER.parent / "experiment_config_f4.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame("test", 9).to_parquet(root / "test.parquet", index=False)
            manifest = {"roles": {"test": {"path": "test.parquet", "sha256": experiment.sha(root / "test.parquet")}}}
            (root / "extra.json").write_text(json.dumps(manifest))
            frames, pins = experiment.load_extra_inputs(config, root / "extra.json")
            self.assertEqual(len(frames["test"]), 9)
            self.assertEqual(pins["test"]["rows"], 9)
            with self.assertRaisesRegex(ValueError, "needs --extra-inputs"):
                experiment.load_extra_inputs(config, None)
            with self.assertRaisesRegex(ValueError, "extra_input_roles"):
                experiment.load_extra_inputs(dict(config, extra_input_roles=[]), None)
            with self.assertRaisesRegex(ValueError, "declares no"):
                experiment.load_extra_inputs(self.config, root / "extra.json")
            manifest["roles"]["test"]["sha256"] = "0" * 64
            (root / "extra.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                experiment.load_extra_inputs(config, root / "extra.json")

    def test_mixing_frontier_and_first_choice(self):
        points = [(0.0, .5), (.1, .8), (1.0, .9)]
        self.assertAlmostEqual(experiment.mixing_frontier(points, .05), .65)
        self.assertAlmostEqual(experiment.mixing_frontier(points, .55), .85)
        data = frame("route", 4)
        for tier, quality in zip(experiment.TIERS, (0., 1., 1.)):
            data[f"quality_{tier}"] = quality
        # Rows alternate source-b/source-a: source-b always gets low, source-a always medium.
        rows = {r["source"]: r for r in experiment.routing_by_source(data, [0, 1, 0, 1])}
        self.assertAlmostEqual(rows["source-b"]["initial_quality"], 0.)
        self.assertAlmostEqual(rows["source-b"]["relative_cost"], 0.)
        self.assertAlmostEqual(rows["source-a"]["initial_quality"], 1.)
        self.assertAlmostEqual(rows["source-a"]["relative_cost"], (.01 - .001) / (.1 - .001))
        self.assertAlmostEqual(rows["source-a"]["gain_over_mixing"], 0.)

    def test_compare_final_reports_f0_versus_retrained_arm(self):
        config = experiment.read(HELPER.parent / "experiment_config_f4.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plan.json").write_text("{}")
            plan_sha = experiment.sha(root / "plan.json")
            experiment.exact_json(root / "final-lock.json", {"plan_sha256": plan_sha})
            stream = self.frames["feedback"].drop(columns="model_input_text").assign(
                window=[i % 6 + 1 for i in range(600)], initial_label="medium")
            (root / "shared-data").mkdir()
            stream.to_parquet(root / "shared-data" / "stream-windows.parquet", index=False)
            test = frame("test", 60)
            arena = frame("routerarena", 30, outcomes=False)
            logits = lambda n, label: np.eye(3)[[label] * n]
            for arm, seeds, label in [("f0_frozen", [0], 0), ("f4_bidirectional", config["seeds"], 1)]:
                for seed in seeds:
                    for role, data in (("test", test), ("routerarena", arena)):
                        experiment.write_scores(root / "scores" / arm / f"seed-{seed}" / "final" / role, data,
                                                logits(len(data), label), {"arm": arm, "seed": seed, "role": role},
                                                plan_sha, "d", "m", 0.)
                    if arm != "f0_frozen":
                        for window in range(2, 7):
                            rows = self.frames["feedback"][stream.window.eq(window).to_numpy()]
                            experiment.write_scores(root / "scores" / arm / f"seed-{seed}" / f"window-{window:02d}", rows,
                                                    logits(len(rows), label), {"arm": arm, "seed": seed}, plan_sha, "d", "m", 0.)
            plan = {"config": config, "pins": {"extra_inputs": {"evaluation_only_text_overlaps": {}}}}
            with patch.object(experiment, "load_plan", return_value=plan):
                experiment.compare_final(SimpleNamespace(run_dir=root, reference_run=None))
            report = root / "report-final"
            status = experiment.read(report / "status.json")
            self.assertTrue(status["complete"], status["missing"])
            contrasts = pd.read_csv(report / "paired-comparisons.csv")
            correct = contrasts[(contrasts.role.eq("test")) & contrasts.scope.eq("all") & contrasts.metric.eq("correct")]
            # Labels cycle easy/medium/hard: F0 always easy, F4 always medium -> same accuracy.
            self.assertAlmostEqual(float(correct.difference.iloc[0]), 0.)
            cost = contrasts[(contrasts.role.eq("test")) & contrasts.scope.eq("all") & contrasts.metric.eq("relative_cost")]
            self.assertGreater(float(cost.difference.iloc[0]), 0)
            self.assertTrue((report / "test-quality-cost.png").is_file())
            self.assertIn("f4_bidirectional", (report / "REPORT.md").read_text())

    def f5_setup(self):
        config = experiment.read(HELPER.parent / "experiment_config_f5.json")
        n = len(self.frames["feedback"])
        # Labels cycle easy/medium/hard; the logger is right on two thirds of the easy rows only.
        self.log["initial_label"] = ["easy" if i % 3 == 0 else ("medium" if i % 3 == 2 and i % 2 else "easy")
                                     for i in range(n)]
        return config

    def test_f5_keeps_every_error_and_restores_window_label_proportions(self):
        config = self.f5_setup()
        updates, stream, _, _ = experiment.make_updates(config, self.frames, self.log)
        selected, summary = experiment.matched_prior_selection(stream, config)
        logged = dict(zip(self.log.sample_id, self.log.initial_label))
        for window, rows in stream.groupby("window"):
            picked = selected[selected.window.eq(window)]
            errors = set(rows[rows.initial_label.ne(rows.difficulty_label)].sample_id)
            self.assertTrue(errors <= set(picked.sample_id))
            fills = picked[picked.label_origin.eq(experiment.AGREEMENT_FILL_ORIGIN)]
            self.assertTrue(all(logged[x] == y for x, y in zip(fills.sample_id, fills.difficulty_label)))
            self.assertFalse(picked.sample_id.duplicated().any())
            shares = rows.difficulty_label.value_counts(normalize=True)
            mix = picked.difficulty_label.value_counts()
            for label in experiment.LABELS:
                info = summary[str(window)][label]
                if not info["shortfall"]:
                    self.assertLessEqual(abs(mix.get(label, 0) - shares[label] * len(picked)), 2, (window, label))
        base_ids = set(self.frames["train"].sample_id)
        window_of = dict(zip(stream.sample_id, stream.window))
        for update in range(1, 7):
            name = f"update-{update:02d}"
            cycle_start = 0 if update <= 3 else 3
            train = updates["f5_matched_prior"][name]
            train = train[train.split.eq("train")]
            base = train[train.sample_id.isin(base_ids)]
            self.assertEqual(len(base), 96)
            history = [x for x in train.sample_id if x in window_of and window_of[x] <= cycle_start]
            self.assertEqual(len(history), 0 if update <= 3 else 96)
            self.assertTrue(all(window_of[x] <= update for x in train.sample_id if x in window_of))
        budget = experiment.training_budget(config, updates)["f5_matched_prior"]
        for name, data in updates["f5_matched_prior"].items():
            rows = int(data.split.eq("train").sum())
            self.assertEqual(budget[name]["max_train_steps"], experiment.steps_per_epoch(rows, config) * 3)

    def test_proportional_sample_follows_label_shares(self):
        data = frame("prop", 100)
        data["difficulty_label"] = ["easy"] * 50 + ["medium"] * 37 + ["hard"] * 13
        picked = experiment.proportional_sample(data, 96, 7)
        self.assertEqual(len(picked), 96)
        self.assertEqual(picked.difficulty_label.value_counts().to_dict(), {"easy": 48, "medium": 36, "hard": 12})

    def test_compare_final_adds_reference_finals_on_identical_data(self):
        config4 = experiment.read(HELPER.parent / "experiment_config_f4.json")
        config5 = experiment.read(HELPER.parent / "experiment_config_f5.json")
        test = frame("test", 60)
        arena = frame("routerarena", 30, outcomes=False)
        logits = lambda n, label: np.eye(3)[[label] * n]
        stream = self.frames["feedback"].drop(columns="model_input_text").assign(
            window=[i % 6 + 1 for i in range(600)], initial_label="medium", outcome="first_try", resolved_tier=1)
        with tempfile.TemporaryDirectory() as tmp:
            runs = {}
            for name, config, arm, label in [("f4", config4, "f4_bidirectional", 1), ("f5", config5, "f5_matched_prior", 2)]:
                root = Path(tmp) / name
                (root / "shared-data").mkdir(parents=True)
                stream.to_parquet(root / "shared-data" / "stream-windows.parquet", index=False)
                plan = {"config": config, "logger_receipt": {"log_sha256": "a"},
                        "data_sha256": {"stream-windows": experiment.sha(root / "shared-data" / "stream-windows.parquet")},
                        "pins": {"data_sha256": {"routerarena": "arena"},
                                 "extra_inputs": {"roles": {"test": {"sha256": "test"}}, "evaluation_only_text_overlaps": {}}}}
                experiment.exact_json(root / "plan.json", plan)
                experiment.exact_json(root / "plan.sha256.json", {"sha256": experiment.sha(root / "plan.json")})
                plan_sha = experiment.sha(root / "plan.json")
                experiment.exact_json(root / "final-lock.json", {"plan_sha256": plan_sha})
                for a, seeds, lab in [("f0_frozen", [0], 0), (arm, config["seeds"], label)]:
                    for seed in seeds:
                        for role, data in (("test", test), ("routerarena", arena)):
                            experiment.write_scores(root / "scores" / a / f"seed-{seed}" / "final" / role, data,
                                                    logits(len(data), lab), {"arm": a, "seed": seed, "role": role},
                                                    plan_sha, "d", "m", 0.)
                        if a != "f0_frozen":
                            for window in range(2, 7):
                                rows = self.frames["feedback"][stream.window.eq(window).to_numpy()]
                                experiment.write_scores(root / "scores" / a / f"seed-{seed}" / f"window-{window:02d}", rows,
                                                        logits(len(rows), lab), {"arm": a, "seed": seed}, plan_sha, "d", "m", 0.)
                runs[name] = (root, plan)
            root, plan = runs["f5"]
            with patch.object(experiment, "load_plan", return_value=plan):
                experiment.compare_final(SimpleNamespace(run_dir=root, reference_run=[runs["f4"][0]]))
            report = root / "report-final"
            self.assertTrue(experiment.read(report / "status.json")["complete"])
            contrasts = pd.read_csv(report / "paired-comparisons.csv")
            self.assertIn("f5_matched_prior_minus_f4_bidirectional", set(contrasts.contrast))
            self.assertIn("f4_bidirectional_minus_f0_frozen", set(contrasts.contrast))
            metrics = pd.read_csv(report / "final-metrics.csv")
            self.assertEqual(set(metrics.arm), {"f0_frozen", "f4_bidirectional", "f5_matched_prior"})
            found = experiment.reference_finals(plan, [runs["f4"][0]], ["test", "routerarena"])
            self.assertEqual(set(found), {("f4_bidirectional", "test"), ("f4_bidirectional", "routerarena")})
            # A reference that scored different test data only contributes RouterArena.
            other = Path(tmp) / "other"
            other.mkdir()
            ref_plan = experiment.read(runs["f4"][0] / "plan.json")
            ref_plan["pins"]["extra_inputs"]["roles"]["test"]["sha256"] = "other"
            experiment.exact_json(other / "plan.json", ref_plan)
            found = experiment.reference_finals(plan, [other], ["test", "routerarena"])
            self.assertEqual(set(found), {("f4_bidirectional", "routerarena")})

    def test_notebook_code_and_clean_outputs(self):
        notebooks = sorted(HELPER.parent.glob("*.ipynb"))
        self.assertEqual(len(notebooks), 13)
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
