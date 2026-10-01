"""R0 retraining-effect research harness (F0/F1/F2). Local/Colab only.

Never submits cloud jobs, never changes CURRENT and never publishes a model.
The contract is docs/ml/RETRAINING_EFFECT_EXPERIMENT.md: a fixed logger (the
operational 1.0 anchor) replays recorded benchmark outcomes to create upward
escalation feedback; F1 learns only the current cycle's feedback, F2 adds the
operational history replay and base rehearsal, F3 adds to F2 confirmed-easy
rows (logger routed easy and the low tier solved it first try) drawn without
reuse so the three labels are as even as possible. Heavy ML imports happen
only for training/inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ("easy", "medium", "hard")
TIERS = ("low", "medium", "high")
ARMS = ("f1_cycle_only", "f2_replay", "f3_balanced_easy")
DEFAULT_ARMS = ["f1_cycle_only", "f2_replay"]
FROZEN = "f0_frozen"
INPUT_ROLES = ("train", "dev", "gate", "feedback", "forget", "routerarena")
FINAL_ROLES = ("gate", "forget", "routerarena")
OUTCOME_ROLES = ("feedback", "gate", "forget")
OUTCOME_COLUMNS = tuple(f"{kind}_{tier}" for kind in ("success", "quality", "actual_cost") for tier in TIERS)
TRAINER_COLUMNS = ["sample_id", "duplicate_group_id", "model_input_text", "difficulty_label",
                   "sample_weight", "split", "label_origin"]
FEEDBACK_ORIGIN = "simulated_observed_escalation"
CONFIRMED_EASY_ORIGIN = "simulated_confirmed_first_try_easy"
HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE.parents[1] / "src" / "llm_router_labeling"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def exact_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Existing artifact differs; use a new run directory: {path}")
        return
    with path.open("x") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def file_pin(path, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Real SHA-256 required for {path}")
    if sha(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def text_key(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def stable_sample(frame, n, seed):
    """Order independent selection keyed by sample_id."""
    keys = frame.sample_id.map(lambda x: digest([seed, str(x)]))
    return frame.assign(_order=keys).sort_values("_order").head(n).drop(columns="_order")


def per_label(frame, n, seed):
    return pd.concat([stable_sample(frame[frame.difficulty_label.eq(label)], n, seed)
                      for label in LABELS], ignore_index=True)


def run_arms(config):
    return list(config.get("arms", DEFAULT_ARMS))


def steps_per_epoch(rows, config):
    return math.ceil(math.ceil(rows / config["train_batch_size"]) / config["gradient_accumulation_steps"])


def validate_frames(frames):
    seen = {"sample_id": set(), "duplicate_group_id": set(), "text": set()}
    for role, frame in frames.items():
        required = {"sample_id", "duplicate_group_id", "model_input_text", "difficulty_label",
                    "source_dataset", "category"}
        if required - set(frame.columns) or frame.empty:
            raise ValueError(f"{role}: empty or missing columns {required - set(frame.columns)}")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"{role}: null identity/label fields")
        for column in required:
            if not frame[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
                raise ValueError(f"{role}: {column} must contain nonempty strings")
        if not frame.difficulty_label.isin(LABELS).all():
            raise ValueError(f"{role}: invalid difficulty label")
        values = {"sample_id": frame.sample_id.astype(str),
                  "duplicate_group_id": frame.duplicate_group_id.astype(str),
                  "text": frame.model_input_text.map(text_key)}
        for key, series in values.items():
            if (key != "duplicate_group_id" and series.duplicated().any()) or seen[key] & set(series):
                raise ValueError(f"Duplicate/leakage in {role}: {key}; keep identities unique and groups within one role")
            seen[key].update(series)
        if role in OUTCOME_ROLES:
            missing = set(OUTCOME_COLUMNS) - set(frame.columns)
            if missing:
                raise ValueError(f"{role}: recorded outcomes are required: {sorted(missing)}")
            values = frame[list(OUTCOME_COLUMNS)].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"{role}: recorded outcomes must be finite; no silent imputation")
            success = frame[[f"success_{t}" for t in TIERS]].to_numpy(float)
            if not np.isin(success, (0.0, 1.0)).all():
                raise ValueError(f"{role}: success columns must be 0/1")
            if (frame[[f"actual_cost_{t}" for t in TIERS]].to_numpy(float) < 0).any():
                raise ValueError(f"{role}: negative recorded cost")


def load_inputs(config_path, pins_path, data_dir):
    config, pins = read(config_path), read(pins_path)
    if config["promotion_mode"] != "direct_research_only" or config["train_max_length"] != 512:
        raise ValueError("Protocol requires research-only direct promotion and 512-token training")
    if config["evaluation_max_length"] != 8192:
        raise ValueError("Protocol evaluates at 8192 tokens only")
    if config["seeds"] != [42, 43, 44] or config["windows"] != 6 or config["updates_per_major"] != 3:
        raise ValueError("Protocol requires three adaptation seeds, six windows and 3-update cycles")
    for name in ("f2_epochs", "minimum_feedback_per_window", "train_batch_size",
                 "gradient_accumulation_steps", "checkpoint_every_steps", "keep_last_checkpoints"):
        if not isinstance(config[name], int) or config[name] <= 0:
            raise ValueError(f"Positive integer required: {name}")
    arms = run_arms(config)
    if not arms or set(arms) - set(ARMS) or len(set(arms)) != len(arms):
        raise ValueError(f"Unknown or duplicated arms: {arms}")
    if "f1_cycle_only" in arms and "f2_replay" not in arms:
        raise ValueError("F1 matches F2's optimizer steps; prepare them together")
    if "f3_balanced_easy" in arms and not isinstance(config.get("easy_supplement_seed"), int):
        raise ValueError("F3 needs an integer easy_supplement_seed")
    if config.get("feedback_per_window_cap") is not None:
        raise ValueError("2026-09-28 decision: every valid feedback event is used; no per-window cap")
    pins = json.loads(json.dumps(pins))
    parent = Path(pins_path).resolve().parent
    runtime = pins["runtime_contract"]
    if not runtime.get("path"):
        raise ValueError("Fill source pins: runtime_contract.path")
    runtime["path"] = str((parent / runtime["path"]).resolve())
    file_pin(runtime["path"], runtime["sha256"])
    source = pins["models"]["r0"]
    if not source.get("path"):
        raise ValueError("Fill source pins: r0.path")
    root = (parent / source["path"]).resolve()
    source["path"] = str(root)
    required = {"model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "training_config.json"}
    if not required.issubset(source["files"]):
        raise ValueError("Pin anchor weights, config, training provenance and tokenizer")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(source["files"]):
        raise ValueError("Pin the entire anchor model directory")
    for name, expected in source["files"].items():
        p = (root / name).resolve()
        if not p.is_relative_to(root):
            raise ValueError("Checkpoint path traversal")
        file_pin(p, expected)
    model_config = read(root / "config.json")
    if model_config.get("model_type") != "modernbert" or model_config.get("router_objective", "ordinal") != "ordinal":
        raise ValueError("Anchor must be the ModernBERT R0 (ordinal) checkpoint")
    cohort = source["cohort"]
    if cohort.get("training_max_length") != 512 or not re.fullmatch(r"[0-9a-f]{40}", str(cohort.get("backbone_revision"))):
        raise ValueError("Anchor cohort needs the original 512 length and immutable backbone revision")
    frames = {}
    for role in INPUT_ROLES:
        path = Path(data_dir) / f"{role}.parquet"
        file_pin(path, pins["data_sha256"][role])
        frames[role] = pd.read_parquet(path).reset_index(drop=True)
    validate_frames(frames)
    return config, pins, frames


def simulate(frame, initial):
    """Replay recorded outcomes: start at the routed tier, escalate on failure.

    Only tiers actually called are charged and revealed. resolved_tier is -1
    when even the highest tier fails.
    """
    initial = np.asarray(initial, dtype=int)
    if initial.shape != (len(frame),) or ((initial < 0) | (initial > 2)).any():
        raise ValueError("Initial tiers must be 0/1/2 for every row")
    success = frame[[f"success_{t}" for t in TIERS]].to_numpy(float)
    cost = frame[[f"actual_cost_{t}" for t in TIERS]].to_numpy(float)
    quality = frame[[f"quality_{t}" for t in TIERS]].to_numpy(float)
    n = len(frame)
    resolved = np.full(n, -1)
    depth = np.zeros(n, dtype=int)
    total = np.zeros(n)
    active = np.ones(n, dtype=bool)
    for tier in range(3):
        tried = active & (initial <= tier)
        depth[tried] += 1
        total[tried] += cost[tried, tier]
        hit = tried & (success[:, tier] == 1)
        resolved[hit] = tier
        active &= ~hit
    outcome = np.where(resolved < 0, "fail_all", np.where(resolved == initial, "first_try", "escalated_success"))
    return pd.DataFrame({"initial_tier": initial, "resolved_tier": resolved, "call_depth": depth,
                         "simulated_cost": total, "outcome": outcome,
                         "initial_quality": quality[np.arange(n), initial]}, index=frame.index)


def route_metrics(frame, initial):
    sim = simulate(frame, initial)
    return {"initial_quality": float(sim.initial_quality.mean()),
            "first_try_rate": float(sim.outcome.eq("first_try").mean()),
            "resolved_rate": float(sim.resolved_tier.ge(0).mean()),
            "escalated_success_rate": float(sim.outcome.eq("escalated_success").mean()),
            "mean_call_depth": float(sim.call_depth.mean()),
            "mean_cost": float(sim.simulated_cost.mean()),
            "total_cost": float(sim.simulated_cost.sum())}


def label_metrics(frame, predicted):
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    y = frame.difficulty_label.map(LABELS.index).to_numpy()
    p = np.asarray(predicted, dtype=int)
    result = {"rows": int(len(frame)), "accuracy": float(accuracy_score(y, p)),
              "macro_f1": float(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0)),
              "hard_recall": float(recall_score(y, p, labels=[2], average="macro", zero_division=0)),
              "under_route_rate": float((p < y).mean()), "over_route_rate": float((p > y).mean()),
              "easy_to_hard_rate_all_rows": float(((y == 0) & (p == 2)).mean())}
    for i, name in enumerate(LABELS):
        result[f"predicted_{name}_share"] = float((p == i).mean())
    scores = []
    for source in sorted(frame.source_dataset.unique()):
        mask = frame.source_dataset.eq(source).to_numpy()
        scores.append(f1_score(y[mask], p[mask], labels=[0, 1, 2], average="macro", zero_division=0))
    result["source_macro_f1"] = float(np.mean(scores))
    return result


def all_metrics(frame, predicted):
    result = label_metrics(frame, predicted)
    if set(OUTCOME_COLUMNS).issubset(frame.columns) and np.isfinite(frame[list(OUTCOME_COLUMNS)].to_numpy(float)).all():
        result.update(route_metrics(frame, predicted))
    return result


def infer(model_path, frame, length, batch_size=1):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("Use the explicitly selected personal Colab GPU runtime")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, attn_implementation="eager").to("cuda").eval()
    logits = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(frame), batch_size):
            batch = tokenizer(frame.model_input_text.iloc[start:start + batch_size].tolist(),
                              padding=True, truncation=True, max_length=length, return_tensors="pt")
            logits.append(model(**{k: v.to("cuda") for k, v in batch.items()}).logits.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(logits)


def logger_receipt(pins, log_path):
    return {"kind": "fixed_logger_anchor_r0", "feedback_sha256": pins["data_sha256"]["feedback"],
            "log_sha256": sha(log_path), "logger_model_digest": digest(pins["models"]["r0"]),
            "logging_max_length": 8192}


def build_logger_log(args):
    config, pins, frames = load_inputs(args.config, args.pins, args.data_dir)
    path = Path(args.output_file)
    if path.exists() or path.with_suffix(".json").exists():
        raise ValueError("Logger output already exists; do not overwrite the frozen stream")
    frame = frames["feedback"]
    logits = infer(pins["models"]["r0"]["path"], frame, config["evaluation_max_length"], config["eval_batch_size"])
    out = frame[["sample_id"]].copy()
    for i, name in enumerate(LABELS):
        out[f"logit_{name}"] = logits[:, i]
    out["initial_label"] = np.asarray(LABELS)[logits.argmax(1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    exact_json(path.with_suffix(".json"), logger_receipt(pins, path))


def parent_update(update, config):
    """Operational successfulHttpRequest rule: each major cycle restarts from its .0 anchor."""
    cycle_start = ((update - 1) // config["updates_per_major"]) * config["updates_per_major"]
    return "anchor" if cycle_start == 0 else f"update-{cycle_start:02d}"


def make_updates(config, frames, log):
    feedback = frames["feedback"]
    if log.sample_id.duplicated().any() or set(log.sample_id) != set(feedback.sample_id):
        raise ValueError("Logger log must cover the pinned feedback partition exactly once")
    if not log.initial_label.isin(LABELS).all():
        raise ValueError("Unknown logger labels")
    stream = stable_sample(feedback, len(feedback), config["stream_seed"]).reset_index(drop=True)
    stream = stream.merge(log, on="sample_id", validate="one_to_one", sort=False)
    stream = pd.concat([stream, simulate(stream, stream.initial_label.map(LABELS.index).to_numpy())], axis=1)
    stream["window"] = 0
    for window, ids in enumerate(np.array_split(np.arange(len(stream)), config["windows"]), 1):
        stream.loc[ids, "window"] = window
    events = stream[stream.outcome.eq("escalated_success")].copy()
    events["benchmark_difficulty_label"] = events.difficulty_label
    events["difficulty_label"] = np.asarray(LABELS)[events.resolved_tier.to_numpy()]
    events["label_origin"] = FEEDBACK_ORIGIN
    counts = events.window.value_counts().reindex(range(1, config["windows"] + 1), fill_value=0)
    short = {int(k): int(v) for k, v in counts.items() if v < config["minimum_feedback_per_window"]}
    if short:
        raise ValueError(f"Windows {short} have fewer than {config['minimum_feedback_per_window']} feedback rows. "
                         "Revise and version the design BEFORE fitting.")
    seed = config["rehearsal_seed"]
    validation = per_label(frames["dev"], config["validation_per_label"], seed).assign(
        split="validation", sample_weight=1.0, label_origin="benchmark_difficulty")
    base = per_label(frames["train"], config["base_per_label"], seed).assign(label_origin="benchmark_difficulty")
    arms = run_arms(config)
    updates = {arm: {} for arm in arms}
    # Arrived requests the logger sent to the low tier and that were solved first try.
    confirmed = stream[stream.outcome.eq("first_try") & stream.initial_tier.eq(0)].copy()
    confirmed["benchmark_difficulty_label"] = confirmed.difficulty_label
    confirmed["difficulty_label"] = "easy"
    confirmed["label_origin"] = CONFIRMED_EASY_ORIGIN
    used_easy, supplement = set(), {}
    per = config["updates_per_major"]
    for update in range(1, config["windows"] + 1):
        name = f"update-{update:02d}"
        cycle_start = ((update - 1) // per) * per
        current = events[events.window.between(cycle_start + 1, update)]
        history = per_label(events[events.window.le(cycle_start)], config["history_per_label"], seed)
        replay = pd.concat([current, history, base], ignore_index=True)
        rows = {"f1_cycle_only": current, "f2_replay": replay}
        if "f3_balanced_easy" in arms:
            counts = replay.difficulty_label.value_counts().reindex(LABELS, fill_value=0)
            target = (int(counts["medium"]) + int(counts["hard"]) + 1) // 2
            need = max(0, target - int(counts["easy"]))
            pool = confirmed[confirmed.window.le(update) & ~confirmed.sample_id.isin(used_easy)]
            added = stable_sample(pool, need, config["easy_supplement_seed"] + update)
            used_easy.update(added.sample_id)
            supplement[name] = {"target_easy": target, "base_easy": int(counts["easy"]), "added_easy": int(len(added)),
                                "pool_available": int(len(pool)), "shortfall": int(need - len(added))}
            rows["f3_balanced_easy"] = pd.concat([replay, added], ignore_index=True)
        for arm in arms:
            train = rows[arm].assign(split="train", sample_weight=1.0)
            updates[arm][name] = pd.concat([train[TRAINER_COLUMNS], validation[TRAINER_COLUMNS]], ignore_index=True)
    return updates, stream, events, supplement


def training_budget(config, updates):
    """F2/F3 use the operational 3 epochs of their own rows; F1 repeats epochs to match F2's steps."""
    budget = {arm: {} for arm in updates}
    for arm in ("f2_replay", "f3_balanced_easy"):
        for name, data in updates.get(arm, {}).items():
            rows = int(data.split.eq("train").sum())
            budget[arm][name] = {"train_rows": rows, "epochs": config["f2_epochs"],
                                 "max_train_steps": steps_per_epoch(rows, config) * config["f2_epochs"]}
    for name, data in updates.get("f1_cycle_only", {}).items():
        steps = budget["f2_replay"][name]["max_train_steps"]
        rows = int(data.split.eq("train").sum())
        budget["f1_cycle_only"][name] = {"train_rows": rows, "epochs": math.ceil(steps / steps_per_epoch(rows, config)),
                                         "max_train_steps": steps}
    return budget


def prepare(args):
    config, pins, frames = load_inputs(args.config, args.pins, args.data_dir)
    log_path = Path(args.logger_log)
    receipt = read(log_path.with_suffix(".json"))
    if receipt != logger_receipt(pins, log_path):
        raise ValueError("Logger provenance mismatch")
    updates, stream, events, supplement = make_updates(config, frames, pd.read_parquet(log_path))
    root = Path(args.output_dir).resolve()
    if root.exists():
        raise ValueError("Prepare needs a NEW directory; other notebooks reuse that sealed plan")
    root.mkdir(parents=True)
    hashes = {}
    for arm, phases in updates.items():
        for name, frame in phases.items():
            path = root / "shared-data" / arm / f"{name}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            hashes[f"{arm}/{name}"] = sha(path)
    windows_path = root / "shared-data" / "stream-windows.parquet"
    stream.drop(columns=["model_input_text"]).to_parquet(windows_path, index=False)
    hashes["stream-windows"] = sha(windows_path)
    archives = Path(args.pins).resolve().parent / "model-archives.json"
    plan = {"config": config, "pins": pins, "data_dir": str(Path(args.data_dir).resolve()),
            "logger_receipt": receipt, "data_sha256": hashes,
            "anchor_archive": read(archives)["r0"] if archives.exists() else None,
            "training_budget": training_budget(config, updates),
            "easy_supplement": supplement,
            "feedback_per_window": {str(k): int(v) for k, v in events.window.value_counts().sort_index().items()},
            "feedback_label_counts": {k: int(v) for k, v in events.difficulty_label.value_counts().items()},
            "feedback_source_counts": {k: int(v) for k, v in events.source_dataset.value_counts().items()},
            "logger_stream_outcomes": {k: int(v) for k, v in stream.outcome.value_counts().items()},
            "helper_sha256": sha(__file__),
            "source_code_sha256": {p.relative_to(SOURCE_ROOT).as_posix(): sha(p) for p in sorted(SOURCE_ROOT.rglob("*.py"))},
            "production_approved": False, "status": "prepared_not_trained"}
    exact_json(root / "plan.json", plan)
    exact_json(root / "plan.sha256.json", {"sha256": sha(root / "plan.json")})
    print(json.dumps({"run_dir": str(root), "feedback_per_window": plan["feedback_per_window"],
                      "feedback_label_counts": plan["feedback_label_counts"],
                      "easy_supplement": plan["easy_supplement"],
                      "training_budget": plan["training_budget"]}, indent=2))


def load_plan(root):
    root = Path(root)
    file_pin(root / "plan.json", read(root / "plan.sha256.json")["sha256"])
    plan = read(root / "plan.json")
    file_pin(__file__, plan["helper_sha256"])
    for name, expected in plan["source_code_sha256"].items():
        file_pin(SOURCE_ROOT / name, expected)
    for name, expected in plan["data_sha256"].items():
        file_pin(root / "shared-data" / f"{name}.parquet", expected)
    return plan


def verify_source(source):
    for name, expected in source["files"].items():
        file_pin(Path(source["path"]) / name, expected)


def model_inventory(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(Path(root).rglob("*")) if p.is_file()}


def verify_finished(directory):
    record = read(directory / "completed.json")
    if model_inventory(directory / "model") != record["model_files"]:
        raise ValueError("Completed model inventory changed")
    return record


def train(args):
    from dataclasses import asdict
    from llm_router_labeling.training.modernbert_training import TrainingConfig, run_training
    from llm_router_labeling.training.modernbert_runtime import load_runtime_contract
    from llm_router_labeling.training.modernbert_checkpointing import CheckpointOptions, latest_completed_checkpoint
    root = Path(args.run_dir)
    plan = load_plan(root)
    cfg, pins = plan["config"], plan["pins"]
    if args.arm not in run_arms(cfg) or args.seed not in cfg["seeds"]:
        raise ValueError("Only this run's arms with declared adaptation seeds can train")
    if (root / "final-lock.json").exists():
        raise ValueError("Experiment locked; no further fitting")
    source = pins["models"]["r0"]
    verify_source(source)
    runtime_pin = pins["runtime_contract"]
    file_pin(runtime_pin["path"], runtime_pin["sha256"])
    runtime = load_runtime_contract(Path(runtime_pin["path"]))
    arm_root = root / "arms" / args.arm / f"seed-{args.seed}"
    names = [f"update-{i:02d}" for i in range(1, cfg["windows"] + 1)]
    if args.until_update not in names:
        raise ValueError("Unknown stopping update")
    for index, name in enumerate(names, 1):
        directory = arm_root / name
        parent = parent_update(index, cfg)
        initial = Path(source["path"]) if parent == "anchor" else arm_root / parent / "model"
        if parent != "anchor":
            verify_finished(arm_root / parent)
        if (directory / "completed.json").exists():
            if verify_finished(directory)["plan_sha256"] != sha(root / "plan.json"):
                raise ValueError("Completed update belongs to a different plan")
        else:
            budget = plan["training_budget"][args.arm][name]
            training = TrainingConfig(training_version=f"{cfg['experiment_id']}-{args.arm}-{args.seed}-{name}",
                model_id="answerdotai/ModernBERT-base", model_revision=source["cohort"]["backbone_revision"],
                train_split="train", validation_split="validation", max_length=cfg["train_max_length"], truncation=True,
                train_batch_size=cfg["train_batch_size"], eval_batch_size=cfg["eval_batch_size"],
                epochs=budget["epochs"], max_train_steps=budget["max_train_steps"],
                gradient_accumulation_steps=cfg["gradient_accumulation_steps"], gradient_checkpointing=True,
                learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"], warmup_ratio=cfg["warmup_ratio"],
                seed=args.seed + index, device="cuda", precision=cfg["precision"],
                attention_implementation=cfg["attention_implementation"], train_sampling_strategy="shuffle",
                gradient_clip_norm=cfg["gradient_clip_norm"])
            directory.mkdir(parents=True, exist_ok=True)
            exact_json(directory / "stage-contract.json", {"plan_sha256": sha(root / "plan.json"),
                        "parent": parent, "initial_model_path": str(initial), "training": asdict(training)})
            checkpoints = directory / "checkpoints"
            resume = latest_completed_checkpoint(checkpoints) if checkpoints.exists() else None
            options = CheckpointOptions(checkpoint_dir=checkpoints, save_every_optimizer_steps=cfg["checkpoint_every_steps"],
                        keep_last_checkpoints=cfg["keep_last_checkpoints"], resume_from=resume)
            start = time.monotonic()
            import torch
            attempt = {"started_at": datetime.now(timezone.utc).isoformat(),
                       "resume_from": str(resume) if resume else None, "status": "running"}
            attempt_path = directory / "attempts" / f"{time.time_ns()}.json"
            exact_json(attempt_path, attempt)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            try:
                result = run_training(root / "shared-data" / args.arm / f"{name}.parquet", directory / "model", training,
                                      runtime_contract=runtime, initial_model_dir=None if resume else initial,
                                      checkpoint_options=options, objective_method="ordinal")
                actual_steps = result.metrics["optimization"]["optimizer_steps"]
                if actual_steps != budget["max_train_steps"]:
                    raise RuntimeError(f"Training budget mismatch: {actual_steps} != {budget['max_train_steps']}")
                attempt.update(status="completed", optimizer_steps=actual_steps)
            except BaseException as error:
                attempt.update(status="failed", error_type=type(error).__name__, error=str(error))
                raise
            finally:
                attempt.update(ended_at=datetime.now(timezone.utc).isoformat(),
                               wall_seconds=time.monotonic() - start,
                               peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
                attempt_path.write_text(json.dumps(attempt, indent=2) + "\n")
            exact_json(directory / "completed.json", {"plan_sha256": sha(root / "plan.json"),
                "arm": args.arm, "seed": args.seed, "update": name, "parent": parent,
                "optimizer_steps": budget["max_train_steps"], "train_rows": budget["train_rows"],
                "this_invocation_seconds": time.monotonic() - start,
                "cumulative_training_seconds": result.metrics["optimization"]["training_seconds"],
                "model_files": model_inventory(directory / "model")})
        if name == args.until_update:
            break


def write_scores(directory, frame, logits, identity, plan_sha, data_sha, model_digest, seconds):
    out = frame.drop(columns=["model_input_text"]).copy()
    predicted = logits.argmax(1)
    out["prediction"] = np.asarray(LABELS)[predicted]
    for i, name in enumerate(LABELS):
        out[f"logit_{name}"] = logits[:, i]
    directory.mkdir(parents=True, exist_ok=True)
    out.to_parquet(directory / "predictions.parquet", index=False)
    exact_json(directory / "receipt.json", {**identity, "plan_sha256": plan_sha, "data_sha256": data_sha,
        "model_digest": model_digest, "inference_seconds": seconds,
        "predictions_sha256": sha(directory / "predictions.parquet"), "metrics": all_metrics(out, predicted)})


def existing_receipt(directory, plan_sha, model_digest):
    if not (directory / "receipt.json").exists():
        return False
    receipt = read(directory / "receipt.json")
    if receipt["plan_sha256"] != plan_sha or receipt["model_digest"] != model_digest:
        raise ValueError("Existing prediction provenance differs")
    file_pin(directory / "predictions.parquet", receipt["predictions_sha256"])
    return True


def score_windows(args):
    """Prequential: the model after update j predicts window j+1 before learning from it."""
    root = Path(args.run_dir)
    plan = load_plan(root)
    cfg, pins = plan["config"], plan["pins"]
    if args.arm not in run_arms(cfg) or args.seed not in cfg["seeds"]:
        raise ValueError("Unsupported arm/seed")
    path = Path(plan["data_dir"]) / "feedback.parquet"
    file_pin(path, pins["data_sha256"]["feedback"])
    windows = pd.read_parquet(root / "shared-data" / "stream-windows.parquet")[["sample_id", "window"]]
    frame = pd.read_parquet(path).merge(windows, on="sample_id", validate="one_to_one")
    plan_sha = sha(root / "plan.json")
    for update in range(1, cfg["windows"]):
        phase = root / "arms" / args.arm / f"seed-{args.seed}" / f"update-{update:02d}"
        verify_finished(phase)
        model_digest = digest(model_inventory(phase / "model"))
        directory = root / "scores" / args.arm / f"seed-{args.seed}" / f"window-{update + 1:02d}"
        if existing_receipt(directory, plan_sha, model_digest):
            continue
        rows = frame[frame.window.eq(update + 1)].reset_index(drop=True)
        start = time.monotonic()
        logits = infer(phase / "model", rows, cfg["evaluation_max_length"], cfg["eval_batch_size"])
        write_scores(directory, rows, logits, {"arm": args.arm, "seed": args.seed, "model_update": update,
                     "role": "window", "window": update + 1}, plan_sha, pins["data_sha256"]["feedback"],
                     model_digest, time.monotonic() - start)


def final_lock(args):
    root = Path(args.run_dir)
    plan = load_plan(root)
    cfg = plan["config"]
    digests, missing = {}, []
    last = f"update-{cfg['windows']:02d}"
    for arm in run_arms(cfg):
        for seed in cfg["seeds"]:
            digests[f"{arm}/{seed}"] = digest(verify_finished(root / "arms" / arm / f"seed-{seed}" / last)["model_files"])
            for window in range(2, cfg["windows"] + 1):
                if not (root / "scores" / arm / f"seed-{seed}" / f"window-{window:02d}" / "receipt.json").exists():
                    missing.append(f"{arm}/{seed}/window-{window:02d}")
    if missing:
        raise ValueError(f"Finish prequential window scoring before final evaluation: {missing}")
    exact_json(root / "final-lock.json", {"plan_sha256": sha(root / "plan.json"),
        "final_model_digests": digests, "production_approved": False})


def score_final(args):
    root = Path(args.run_dir)
    plan = load_plan(root)
    cfg, pins = plan["config"], plan["pins"]
    if args.role not in FINAL_ROLES:
        raise ValueError("Unsupported final role")
    if not (root / "final-lock.json").exists():
        raise ValueError("Final evaluation needs final-lock.json after all training and window scoring")
    lock = read(root / "final-lock.json")
    if lock["plan_sha256"] != sha(root / "plan.json"):
        raise ValueError("Final lock belongs to another plan")
    if args.arm == FROZEN:
        if args.seed != 0:
            raise ValueError("F0 is the untouched anchor, measured once as seed 0")
        verify_source(pins["models"]["r0"])
        model = Path(pins["models"]["r0"]["path"])
    elif args.arm in run_arms(cfg) and args.seed in cfg["seeds"]:
        model = root / "arms" / args.arm / f"seed-{args.seed}" / f"update-{cfg['windows']:02d}" / "model"
        if digest(model_inventory(model)) != lock["final_model_digests"][f"{args.arm}/{args.seed}"]:
            raise ValueError("Final checkpoint changed after the lock")
    else:
        raise ValueError("Unsupported arm/seed")
    path = Path(plan["data_dir"]) / f"{args.role}.parquet"
    file_pin(path, pins["data_sha256"][args.role])
    frame = pd.read_parquet(path)
    plan_sha = sha(root / "plan.json")
    model_digest = digest(model_inventory(model))
    directory = root / "scores" / args.arm / f"seed-{args.seed}" / "final" / args.role
    if existing_receipt(directory, plan_sha, model_digest):
        return
    start = time.monotonic()
    logits = infer(model, frame, cfg["evaluation_max_length"], cfg["eval_batch_size"])
    write_scores(directory, frame, logits, {"arm": args.arm, "seed": args.seed, "role": args.role},
                 plan_sha, pins["data_sha256"][args.role], model_digest, time.monotonic() - start)


def paired_bootstrap(frame, left, right, draws, seed):
    """Mean(right - left) with duplicate groups resampled within each source."""
    diff = frame[right].to_numpy(float) - frame[left].to_numpy(float)
    work = pd.DataFrame({"source": frame.source_dataset.to_numpy(), "group": frame.duplicate_group_id.to_numpy(), "d": diff})
    clusters = work.groupby(["source", "group"]).d.agg(["sum", "count"]).reset_index()
    rng = np.random.default_rng(seed)
    by_source = [g[["sum", "count"]].to_numpy(float) for _, g in clusters.groupby("source")]
    samples = []
    for _ in range(draws):
        total = np.zeros(2)
        for values in by_source:
            total += values[rng.integers(0, len(values), len(values))].sum(0)
        samples.append(total[0] / total[1])
    return float(diff.mean()), float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def reference_sources(plan, reference_run, current_stream=None):
    """Reuse a finished run's arms only when its sealed stream and logger are identical."""
    if not reference_run:
        return {}
    ref = Path(reference_run)
    file_pin(ref / "plan.json", read(ref / "plan.sha256.json")["sha256"])
    ref_plan = read(ref / "plan.json")
    same = {key: ref_plan["config"][key] == plan["config"][key]
            for key in ("seeds", "windows", "stream_seed", "evaluation_max_length")}
    if ref_plan["logger_receipt"] != plan["logger_receipt"] or not all(same.values()):
        raise ValueError("Reference run used a different stream, logger or seeds")
    file_pin(ref / "shared-data" / "stream-windows.parquet", ref_plan["data_sha256"]["stream-windows"])
    if current_stream is not None:
        # Parquet bytes may differ across writers; the sealed rows themselves must match.
        keys = ["sample_id", "window", "initial_label", "outcome", "resolved_tier"]
        theirs = pd.read_parquet(ref / "shared-data" / "stream-windows.parquet", columns=keys)
        if not theirs.reset_index(drop=True).equals(current_stream[keys].reset_index(drop=True)):
            raise ValueError("Reference run used a different stream, logger or seeds")
    return {arm: (ref, sha(ref / "plan.json")) for arm in run_arms(ref_plan["config"])}


def compare(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(args.run_dir)
    plan = load_plan(root)
    cfg = plan["config"]
    plan_sha = sha(root / "plan.json")
    out = root / "report"
    out.mkdir(exist_ok=True)
    stream = pd.read_parquet(root / "shared-data" / "stream-windows.parquet")
    windows = range(1, cfg["windows"] + 1)
    sources = reference_sources(plan, getattr(args, "reference_run", None), stream)
    sources.update({arm: (root, plan_sha) for arm in run_arms(cfg)})
    arms = [arm for arm in ARMS if arm in sources]
    rows, missing = [], []
    per_row = {}
    logger_pred = stream.initial_label.map(LABELS.index).to_numpy()
    for window in windows:
        mask = stream.window.eq(window).to_numpy()
        part = stream[mask].reset_index(drop=True)
        rows.append({"arm": FROZEN, "seed": 0, "window": window, **all_metrics(part, logger_pred[mask])})
    base = stream[stream.window.ge(2)].sort_values("sample_id").reset_index(drop=True)
    sim0 = simulate(base, base.initial_label.map(LABELS.index).to_numpy())
    per_row[FROZEN] = sim0
    for arm in arms:
        arm_root, arm_plan = sources[arm]
        seed_rows = []
        for seed in cfg["seeds"]:
            parts = []
            for window in range(2, cfg["windows"] + 1):
                directory = arm_root / "scores" / arm / f"seed-{seed}" / f"window-{window:02d}"
                if not (directory / "receipt.json").exists():
                    missing.append(f"{arm}/{seed}/window-{window:02d}")
                    continue
                receipt = read(directory / "receipt.json")
                if receipt["plan_sha256"] != arm_plan:
                    raise ValueError("Cannot combine results from different plans")
                file_pin(directory / "predictions.parquet", receipt["predictions_sha256"])
                frame = pd.read_parquet(directory / "predictions.parquet")
                predicted = frame.prediction.map(LABELS.index).to_numpy()
                rows.append({"arm": arm, "seed": seed, "window": window, **all_metrics(frame, predicted)})
                parts.append(frame)
            if len(parts) == cfg["windows"] - 1:
                merged = pd.concat(parts).sort_values("sample_id").reset_index(drop=True)
                if not merged.sample_id.equals(base.sample_id):
                    raise ValueError("Prequential rows differ between arms")
                seed_rows.append(simulate(merged, merged.prediction.map(LABELS.index).to_numpy()))
        if len(seed_rows) == len(cfg["seeds"]):
            per_row[arm] = sum(s[["initial_quality", "simulated_cost", "call_depth"]] for s in seed_rows) / len(seed_rows)
    prequential = pd.DataFrame(rows)
    prequential.to_csv(out / "prequential-metrics.csv", index=False)
    contrasts = []
    pooled = base[["sample_id", "source_dataset", "duplicate_group_id"]].copy()
    for arm, values in per_row.items():
        pooled[f"{arm}_quality"] = values.initial_quality.to_numpy()
        pooled[f"{arm}_cost"] = values.simulated_cost.to_numpy()
        pooled[f"{arm}_depth"] = values.call_depth.to_numpy()
    pairs = [(FROZEN, arm) for arm in arms] + [("f1_cycle_only", "f2_replay"), ("f2_replay", "f3_balanced_easy")]
    for left, right in pairs:
        if left not in per_row or right not in per_row:
            continue
        for metric in ("quality", "cost", "depth"):
            value, low, high = paired_bootstrap(pooled, f"{left}_{metric}", f"{right}_{metric}",
                                                cfg["bootstrap_draws"], cfg["bootstrap_seed"])
            contrasts.append({"contrast": f"{right}_minus_{left}", "metric": metric, "windows": "2-6",
                              "difference": value, "conditional_ci95_low": low, "conditional_ci95_high": high})
    contrasts = pd.DataFrame(contrasts)
    contrasts.to_csv(out / "paired-comparisons.csv", index=False)
    final_rows = []
    predictions = {}
    for arm, seeds in [(FROZEN, [0])] + [(a, cfg["seeds"]) for a in arms]:
        arm_root, arm_plan = (root, plan_sha) if arm == FROZEN else sources[arm]
        for seed in seeds:
            for role in FINAL_ROLES:
                directory = arm_root / "scores" / arm / f"seed-{seed}" / "final" / role
                if not (directory / "receipt.json").exists():
                    missing.append(f"{arm}/{seed}/final/{role}")
                    continue
                receipt = read(directory / "receipt.json")
                if receipt["plan_sha256"] != arm_plan:
                    raise ValueError("Cannot combine results from different plans")
                file_pin(directory / "predictions.parquet", receipt["predictions_sha256"])
                frame = pd.read_parquet(directory / "predictions.parquet")
                predictions[(arm, seed, role)] = frame
                final_rows.append({"arm": arm, "seed": seed, "role": role,
                                   **all_metrics(frame, frame.prediction.map(LABELS.index).to_numpy())})
    final = pd.DataFrame(final_rows)
    final.to_csv(out / "final-metrics.csv", index=False)
    retention = []
    baseline_key = (FROZEN, 0, "forget")
    if baseline_key in predictions:
        baseline = label_metrics(predictions[baseline_key], predictions[baseline_key].prediction.map(LABELS.index))["source_macro_f1"]
        for arm in arms:
            for seed in cfg["seeds"]:
                if (arm, seed, "forget") in predictions:
                    frame = predictions[(arm, seed, "forget")]
                    value = label_metrics(frame, frame.prediction.map(LABELS.index))["source_macro_f1"]
                    retention.append({"arm": arm, "seed": seed, "f0_source_macro_f1": baseline,
                                      "final_source_macro_f1": value, "drop": baseline - value,
                                      "maximum_drop": cfg["maximum_forgetting_drop"],
                                      "within_tolerance": baseline - value <= cfg["maximum_forgetting_drop"]})
    retention = pd.DataFrame(retention)
    retention.to_csv(out / "retention-check.csv", index=False)
    judgments = {}
    for arm in arms:
        q = contrasts[(contrasts.contrast == f"{arm}_minus_{FROZEN}") & (contrasts.metric == "quality")] if len(contrasts) else contrasts
        c = contrasts[(contrasts.contrast == f"{arm}_minus_{FROZEN}") & (contrasts.metric == "cost")] if len(contrasts) else contrasts
        r = retention[retention.arm == arm] if len(retention) else retention
        judgments[arm] = {
            "quality_minus_f0": float(q.difference.iloc[0]) if len(q) else None,
            "effect": bool(q.difference.iloc[0] > cfg["minimum_improvement"]) if len(q) else None,
            "cost_minus_f0": float(c.difference.iloc[0]) if len(c) else None,
            "cost_increase": bool(c.difference.iloc[0] > 0) if len(c) else None,
            "retention_within_tolerance": bool(r.within_tolerance.all()) if len(r) == len(cfg["seeds"]) else None}
    f2_f1 = contrasts[(contrasts.contrast == "f2_replay_minus_f1_cycle_only") & (contrasts.metric == "quality")] if len(contrasts) else contrasts
    if len(f2_f1):
        low, high = f2_f1.conditional_ci95_low.iloc[0], f2_f1.conditional_ci95_high.iloc[0]
        judgments["f2_vs_f1_quality"] = "no_difference_confirmed" if low <= 0 <= high else ("f2_higher" if low > 0 else "f1_higher")
    f3_f2 = contrasts[(contrasts.contrast == "f3_balanced_easy_minus_f2_replay")] if len(contrasts) else contrasts
    for metric in ("quality", "cost"):
        row = f3_f2[f3_f2.metric == metric] if len(f3_f2) else f3_f2
        if len(row):
            low, high = row.conditional_ci95_low.iloc[0], row.conditional_ci95_high.iloc[0]
            judgments[f"f3_vs_f2_{metric}"] = "no_difference_confirmed" if low <= 0 <= high else ("f3_higher" if low > 0 else "f2_higher")
    status = {"complete": not missing, "missing": missing, "judgments": judgments,
              "production_approved": False, "plan_sha256": plan_sha}
    (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, label in zip(axes, ("initial_quality", "mean_cost", "mean_call_depth"),
                                 ("Initial selection quality", "Mean simulated cost", "Mean call depth")):
        for arm, group in prequential.groupby("arm"):
            summary = group.groupby("window")[metric].agg(["mean", "std"])
            ax.errorbar(summary.index, summary["mean"], yerr=summary["std"].fillna(0), marker="o", label=arm)
        ax.set(xlabel="Window (predicted before learning it)", ylabel=label, xticks=list(windows))
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "prequential-curves.png", dpi=160)
    plt.close(fig)
    from sklearn.metrics import confusion_matrix
    for (arm, seed, role), frame in predictions.items():
        if role != "forget":
            continue
        fig, ax = plt.subplots(figsize=(4, 4))
        matrix = confusion_matrix(frame.difficulty_label, frame.prediction, labels=LABELS)
        ax.imshow(matrix, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        ax.set(xticks=range(3), xticklabels=LABELS, yticks=range(3), yticklabels=LABELS,
               xlabel="Predicted", ylabel="Reference", title=f"{arm} / forget / seed {seed}")
        fig.tight_layout()
        fig.savefig(out / f"confusion-{arm}-{seed}-forget.png", dpi=140)
        plt.close(fig)
    summary = prequential[prequential.window.ge(2)].groupby("arm")[
        ["predicted_easy_share", "initial_quality", "first_try_rate", "resolved_rate", "mean_call_depth", "mean_cost", "source_macro_f1"]].mean()
    (out / "REPORT.md").write_text("# R0 재학습 효과 실험 결과\n\n"
        "생성된 실제 예측만 집계했다. 누락 항목은 완료로 해석하지 않으며 status.json에 표시한다. "
        "F0는 운영 1.0(anchor)이자 고정 로거이고, F1/F2는 seed 42/43/44의 adaptation 반복이다.\n\n"
        f"완료 여부: {not missing}\n\n"
        "## Prequential 요약 (window 2~6, 업데이트 이전 예측)\n\n" + summary.to_csv() + "\n"
        "## Paired 비교 (질문 그룹 bootstrap, 고정 anchor·seed 조건부)\n\n" + contrasts.to_csv(index=False) + "\n"
        "## 망각 (forget source-macro-F1, F0 대비)\n\n" + retention.to_csv(index=False) + "\n"
        "## 판정 (계획서 8절)\n\n```json\n" + json.dumps(judgments, indent=2, ensure_ascii=False) + "\n```\n\n"
        "운영 설정·CURRENT는 이 결과로 자동 변경하지 않는다.\n", encoding="utf-8")
    print(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "build-logger-log"):
        p = subs.add_parser(name)
        for flag in ("config", "pins", "data-dir"):
            p.add_argument("--" + flag, required=True)
        if name == "prepare":
            p.add_argument("--logger-log", required=True)
            p.add_argument("--output-dir", required=True)
        else:
            p.add_argument("--output-file", required=True)
    p = subs.add_parser("train")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--until-update", default="update-06")
    p = subs.add_parser("score-windows")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p = subs.add_parser("score-final")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--arm", choices=(FROZEN, *ARMS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--role", choices=FINAL_ROLES, required=True)
    subs.add_parser("final-lock").add_argument("--run-dir", required=True)
    p = subs.add_parser("compare")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--reference-run", help="finished run whose F1/F2 used the same sealed stream and logger")
    args = parser.parse_args()
    {"prepare": prepare, "build-logger-log": build_logger_log, "train": train,
     "score-windows": score_windows, "final-lock": final_lock, "score-final": score_final,
     "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
