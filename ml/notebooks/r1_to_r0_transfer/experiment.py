"""Local/Colab research harness. Never submits cloud jobs or publishes a model.

Inputs are explicitly pinned; source checkpoints are never edited. Heavy ML
imports happen only for training/inference. See the adjacent notebooks and the
separate protocol for the limits of the three-arm comparison.
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
ARMS = ("r0_continued", "r1_to_r0", "r1_frozen")
ROLES = ("train", "dev", "gate", "feedback", "forget", "routerarena")
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
    """Order independent: no sampling from protected roles or future windows."""
    keys = frame.sample_id.map(lambda x: digest([seed, str(x)]))
    return frame.assign(_order=keys).sort_values("_order").head(n).drop(columns="_order")


def per_label(frame, n, seed):
    return pd.concat([stable_sample(frame[frame.difficulty_label.eq(label)], n, seed)
                      for label in LABELS], ignore_index=True)


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
        if not frame.model_input_text.astype(str).str.strip().ne("").all():
            raise ValueError(f"{role}: empty input")
        values = {"sample_id": frame.sample_id.astype(str),
                  "duplicate_group_id": frame.duplicate_group_id.astype(str),
                  "text": frame.model_input_text.map(text_key)}
        for key, series in values.items():
            if (key != "duplicate_group_id" and series.duplicated().any()) or seen[key] & set(series):
                raise ValueError(f"Duplicate/leakage in {role}: {key}; keep identities unique and groups within one role")
            seen[key].update(series)


def load_inputs(config_path, pins_path, data_dir):
    config, pins = read(config_path), read(pins_path)
    if config["automatic_promotion"] or config["train_max_length"] != 512:
        raise ValueError("Primary protocol requires research-only 512-token training")
    if config["seeds"] != [42, 43, 44] or config["cycles"] != 6:
        raise ValueError("Primary protocol requires three adaptation seeds and six windows")
    for name in ("bridge_steps", "cycle_steps", "new_feedback_per_cycle", "train_batch_size",
                 "gradient_accumulation_steps", "checkpoint_every_steps"):
        if not isinstance(config[name], int) or config[name] <= 0:
            raise ValueError(f"Positive integer required: {name}")
    pins = json.loads(json.dumps(pins))
    parent = Path(pins_path).resolve().parent
    for key in ("r1_policy", "runtime_contract"):
        value = pins[key]
        if not value.get("path"):
            raise ValueError(f"Fill source pins: {key}.path")
        value["path"] = str((parent / value["path"]).resolve())
        file_pin(value["path"], value["sha256"])
    for key in ("r0", "r1"):
        source = pins["models"][key]
        if not source.get("path"):
            raise ValueError(f"Fill source pins: {key}.path")
        root = (parent / source["path"]).resolve()
        source["path"] = str(root)
        required = {"model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "training_config.json"}
        if not required.issubset(source["files"]):
            raise ValueError("Pin checkpoint weights, config, training provenance and tokenizer")
        # Pin every file that from_pretrained could consume, not just weights.
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        if actual != set(source["files"]):
            raise ValueError(f"Pin the entire source model directory: {key}")
        for name, expected in source["files"].items():
            p = (root / name).resolve()
            if not p.is_relative_to(root):
                raise ValueError("Checkpoint path traversal")
            file_pin(p, expected)
        mc = read(root / "config.json")
        if mc.get("model_type") != "modernbert" or mc.get("router_objective", "ordinal") != {"r0": "ordinal", "r1": "pointwise"}[key]:
            raise ValueError(f"Unexpected checkpoint architecture/objective: {key}")
        cohort = source["cohort"]
        if any(v is None for v in cohort.values()):
            raise ValueError("Initial training cohort evidence is incomplete")
        for field in ("training_rows_sha256", "split_manifest_sha256", "label_definition_sha256", "candidate_pool_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(cohort[field])):
                raise ValueError(f"Invalid original cohort digest: {field}")
    r0, r1 = pins["models"]["r0"], pins["models"]["r1"]
    if r0["cohort"] != r1["cohort"]:
        raise ValueError("Confounded source cohorts: primary A R0/A R1 must match")
    if r0["cohort"]["training_max_length"] != 512 or not re.fullmatch(r"[0-9a-f]{40}", str(r0["cohort"]["backbone_revision"])):
        raise ValueError("Source cohort needs the original 512 length and immutable backbone revision")
    if r0["files"]["tokenizer.json"] != r1["files"]["tokenizer.json"]:
        raise ValueError("Source tokenizers differ")
    frames = {}
    for role in ROLES:
        path = Path(data_dir) / f"{role}.parquet"
        file_pin(path, pins["data_sha256"][role])
        frames[role] = pd.read_parquet(path).reset_index(drop=True)
    validate_frames(frames)
    policy = read(pins["r1_policy"]["path"])
    for source in set(pd.concat([frames[x] for x in ROLES if x != "routerarena"]).source_dataset):
        p = policy["sources"][source]
        costs = np.asarray(p["decision_costs"], dtype=float)
        if costs.shape != (3,) or not np.isfinite(costs).all() or (costs < 0).any() or not math.isfinite(p["lambda"]) or p["lambda"] < 0:
            raise ValueError("Invalid frozen R1 policy")
    return config, pins, frames


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


def build_feedback_log(args):
    config, pins, frames = load_inputs(args.config, args.pins, args.data_dir)
    path = Path(args.output_file)
    if path.exists() or path.with_suffix(".json").exists():
        raise ValueError("Feedback output already exists; do not overwrite the frozen stream")
    frame = frames["feedback"]
    logits = infer(pins["models"]["r0"]["path"], frame, 8192, config["eval_batch_size"])
    out = frame[["sample_id"]].copy()
    out["initial_label"] = np.asarray(LABELS)[logits.argmax(1)]
    out["final_label"] = frame.difficulty_label.to_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    exact_json(path.with_suffix(".json"), {"kind": "benchmark_upward_proxy",
        "feedback_sha256": pins["data_sha256"]["feedback"], "log_sha256": sha(path),
        "logger_model_digest": digest(pins["models"]["r0"]), "logging_max_length": 8192})


def make_phases(config, frames, log):
    feedback = frames["feedback"]
    if log.sample_id.duplicated().any() or set(log.sample_id) != set(feedback.sample_id):
        raise ValueError("Feedback log must cover the pinned feedback partition exactly once")
    if not log.initial_label.isin(LABELS).all() or not log.final_label.isin(LABELS).all():
        raise ValueError("Unknown logging labels")
    stream = stable_sample(feedback, len(feedback), config["stream_seed"])
    stream = stream.merge(log, on="sample_id", validate="one_to_one", sort=False)
    if not stream.final_label.eq(stream.difficulty_label).all():
        raise ValueError("This experiment supports benchmark proxy labels only")
    validation = per_label(frames["dev"], config["validation_per_label"], config["stream_seed"])
    phases = {"bridge": frames["train"].copy()}
    history = stream.iloc[:0].copy()
    for i, ids in enumerate(np.array_split(np.arange(len(stream)), config["cycles"]), 1):
        window = stream.iloc[ids]
        eligible = window[window.final_label.map(LABELS.index) > window.initial_label.map(LABELS.index)]
        if len(eligible) < config["new_feedback_per_cycle"]:
            raise ValueError(f"Window {i} has {len(eligible)} eligible rows; need {config['new_feedback_per_cycle']}. Revise and version the design BEFORE fitting.")
        new = stable_sample(eligible, config["new_feedback_per_cycle"], config["stream_seed"] + i).assign(label_origin="benchmark_upward_proxy")
        replay = per_label(history, config["history_per_label"], config["stream_seed"] + i)
        base = per_label(frames["train"], config["base_per_label"], config["stream_seed"] + i).assign(label_origin="benchmark_difficulty")
        phases[f"cycle-{i:02d}"] = pd.concat([new, replay, base], ignore_index=True)
        history = pd.concat([history, new], ignore_index=True)
    out = {}
    for phase, rows in phases.items():
        rows = rows.assign(split="train", sample_weight=1.0)
        if "label_origin" not in rows:
            rows["label_origin"] = "benchmark_difficulty"
        val = validation.assign(split="validation", sample_weight=1.0, label_origin="benchmark_difficulty")
        # Only model inputs and R0 labels enter the trainer. Candidate outcomes
        # and observed escalation metadata never become model features.
        cols = ["sample_id", "duplicate_group_id", "model_input_text", "difficulty_label", "sample_weight", "split", "label_origin"]
        out[phase] = pd.concat([rows[cols], val[cols]], ignore_index=True)
    return out


def prepare(args):
    config, pins, frames = load_inputs(args.config, args.pins, args.data_dir)
    log_path = Path(args.feedback_log)
    receipt = read(log_path.with_suffix(".json"))
    expected = {"kind": "benchmark_upward_proxy", "feedback_sha256": pins["data_sha256"]["feedback"],
                "log_sha256": sha(log_path), "logger_model_digest": digest(pins["models"]["r0"]), "logging_max_length": 8192}
    if receipt != expected:
        raise ValueError("Feedback logging provenance mismatch")
    phases = make_phases(config, frames, pd.read_parquet(log_path))
    root = Path(args.output_dir).resolve()
    if root.exists():
        raise ValueError("Prepare needs a NEW directory; other notebooks reuse that sealed plan")
    root.mkdir(parents=True)
    hashes = {}
    for phase, frame in phases.items():
        path = root / "shared-data" / f"{phase}.parquet"
        path.parent.mkdir(exist_ok=True)
        frame.to_parquet(path, index=False)
        hashes[phase] = sha(path)
    plan = {"config": config, "pins": pins, "data_dir": str(Path(args.data_dir).resolve()),
            "feedback_receipt": receipt, "phase_sha256": hashes,
            "phase_train_rows": {k: int(v.split.eq("train").sum()) for k, v in phases.items()},
            "phase_label_counts": {k: v[v.split.eq("train")].difficulty_label.value_counts().to_dict() for k, v in phases.items()},
            "helper_sha256": sha(__file__),
            "source_code_sha256": {p.relative_to(SOURCE_ROOT).as_posix(): sha(p) for p in sorted(SOURCE_ROOT.rglob("*.py"))},
            "status": "prepared_not_trained"}
    exact_json(root / "plan.json", plan)
    exact_json(root / "plan.sha256.json", {"sha256": sha(root / "plan.json")})
    print(json.dumps({"run_dir": str(root), "phase_train_rows": plan["phase_train_rows"]}, indent=2))


def load_plan(root):
    root = Path(root)
    file_pin(root / "plan.json", read(root / "plan.sha256.json")["sha256"])
    plan = read(root / "plan.json")
    file_pin(__file__, plan["helper_sha256"])
    for name, expected in plan["source_code_sha256"].items():
        file_pin(SOURCE_ROOT / name, expected)
    for phase, expected in plan["phase_sha256"].items():
        file_pin(root / "shared-data" / f"{phase}.parquet", expected)
    return plan


def verify_source(source):
    for name, expected in source["files"].items():
        file_pin(Path(source["path"]) / name, expected)


def reset_r1_head(model, seed):
    """ModernBERT encoder stays intact; reset both task head and final classifier."""
    import torch
    if model.config.model_type != "modernbert" or model.config.router_objective != "pointwise":
        raise ValueError("Conversion requires a ModernBERT R1 checkpoint")
    if not all(hasattr(model, name) for name in ("model", "head", "classifier")):
        raise ValueError("Unexpected ModernBERT architecture; inspect before conversion")
    torch.manual_seed(seed)
    model.head.apply(model._init_weights)
    model.classifier.apply(model._init_weights)
    model.config.router_objective = "ordinal"
    model.config.router_score_semantics = "difficulty_class_logits"
    model.config.problem_type = "single_label_classification"
    model.config.id2label = dict(enumerate(LABELS))
    model.config.label2id = {x: i for i, x in enumerate(LABELS)}
    model.config.router_research_only = True


def model_inventory(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def verify_finished(directory):
    record = read(directory / "completed.json")
    if model_inventory(directory / "model") != record["model_files"]:
        raise ValueError("Completed model inventory changed")
    for name, expected in record["model_files"].items():
        file_pin(directory / "model" / name, expected)
    return record


def train(args):
    from dataclasses import asdict
    from llm_router_labeling.training.modernbert_training import TrainingConfig, run_training
    from llm_router_labeling.training.modernbert_runtime import load_runtime_contract
    from llm_router_labeling.training.modernbert_checkpointing import CheckpointOptions, latest_completed_checkpoint
    root = Path(args.run_dir)
    plan = load_plan(root)
    cfg, pins = plan["config"], plan["pins"]
    if args.arm not in ARMS[:2] or args.seed not in cfg["seeds"]:
        raise ValueError("Only A/B with declared adaptation seeds can train")
    if (root / "selection-lock.json").exists():
        raise ValueError("Experiment locked; no further fitting")
    source = pins["models"]["r0" if args.arm == ARMS[0] else "r1"]
    verify_source(source)
    initial = Path(source["path"])
    arm_root = root / "arms" / args.arm / f"seed-{args.seed}"
    if args.arm == "r1_to_r0":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        converted = arm_root / "conversion"
        receipt_path = converted / "conversion.json"
        identity = {"parent_model_digest": digest(source), "seed": args.seed,
                    "reset_modules": ["head", "classifier"], "encoder_reused": True}
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt["identity"] != identity:
                raise ValueError("Conversion provenance changed")
            for name, expected in receipt["model_files"].items():
                file_pin(converted / "model" / name, expected)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(initial, local_files_only=True, attn_implementation="eager")
            tokenizer = AutoTokenizer.from_pretrained(initial, local_files_only=True)
            reset_r1_head(model, args.seed)
            model.save_pretrained(converted / "model", safe_serialization=True)
            tokenizer.save_pretrained(converted / "model")
            exact_json(receipt_path, {"identity": identity, "model_files": model_inventory(converted / "model")})
            del model
        initial = converted / "model"
    phase_names = ["bridge"] + [f"cycle-{i:02d}" for i in range(1, cfg["cycles"] + 1)]
    if args.until_phase not in phase_names:
        raise ValueError("Unknown stopping phase")
    runtime_pin = pins["runtime_contract"]
    file_pin(runtime_pin["path"], runtime_pin["sha256"])
    runtime = load_runtime_contract(Path(runtime_pin["path"]))
    for index, phase in enumerate(phase_names):
        directory = arm_root / phase
        if (directory / "completed.json").exists():
            finished = verify_finished(directory)
            if finished["plan_sha256"] != sha(root / "plan.json"):
                raise ValueError("Completed phase belongs to a different plan")
        else:
            steps = cfg["bridge_steps"] if phase == "bridge" else cfg["cycle_steps"]
            count = plan["phase_train_rows"][phase]
            steps_per_epoch = math.ceil(math.ceil(count / cfg["train_batch_size"]) / cfg["gradient_accumulation_steps"])
            training = TrainingConfig(training_version=f"{cfg['experiment_id']}-{args.arm}-{args.seed}-{phase}",
                model_id="answerdotai/ModernBERT-base", model_revision=source["cohort"]["backbone_revision"],
                train_split="train", validation_split="validation", max_length=cfg["train_max_length"], truncation=True,
                train_batch_size=cfg["train_batch_size"], eval_batch_size=cfg["eval_batch_size"],
                epochs=math.ceil(steps / steps_per_epoch), max_train_steps=steps,
                gradient_accumulation_steps=cfg["gradient_accumulation_steps"], gradient_checkpointing=True,
                learning_rate=cfg["learning_rate"], weight_decay=cfg["weight_decay"], warmup_ratio=cfg["warmup_ratio"],
                seed=args.seed + index, device="cuda", precision=cfg["precision"],
                attention_implementation=cfg["attention_implementation"], train_sampling_strategy="shuffle",
                gradient_clip_norm=cfg["gradient_clip_norm"])
            directory.mkdir(parents=True, exist_ok=True)
            exact_json(directory / "stage-contract.json", {"plan_sha256": sha(root / "plan.json"),
                        "initial_model_path": str(initial), "training": asdict(training)})
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
                result = run_training(root / "shared-data" / f"{phase}.parquet", directory / "model", training,
                                      runtime_contract=runtime, initial_model_dir=None if resume else initial,
                                      checkpoint_options=options, objective_method="ordinal")
                actual_steps = result.metrics["optimization"]["optimizer_steps"]
                if actual_steps != steps:
                    raise RuntimeError(f"Training budget incomplete: {actual_steps} != {steps}")
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
                "arm": args.arm, "seed": args.seed, "phase": phase, "optimizer_steps": steps,
                "this_invocation_seconds": time.monotonic() - start,
                "cumulative_training_seconds": result.metrics["optimization"]["training_seconds"],
                "model_bytes": sum(p.stat().st_size for p in (directory / "model").rglob("*") if p.is_file()),
                "model_files": model_inventory(directory / "model")})
        initial = directory / "model"
        if phase == args.until_phase:
            break


def predictions(logits, frame, arm, policy, role):
    if arm != "r1_frozen":
        return logits.argmax(1)
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    if role == "routerarena":
        return probabilities.argmax(1)  # diagnostic only: no service pool outcomes
    adjusted = probabilities.copy()
    for source in frame.source_dataset.unique():
        rule = policy["sources"][source]  # no silent default for unknown pools
        mask = frame.source_dataset.eq(source).to_numpy()
        adjusted[mask] -= float(rule["lambda"]) * np.asarray(rule["decision_costs"])
    return adjusted.argmax(1)


def metrics(frame):
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    y, p = frame.difficulty_label.map(LABELS.index), frame.prediction.map(LABELS.index)
    result = {"rows": len(frame), "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0),
        "hard_recall": recall_score(y, p, labels=[2], average="macro", zero_division=0),
        "under_route_rate": float((p < y).mean()),
        "hard_to_easy_rate_all_rows": float(((y == 2) & (p == 0)).mean())}
    result["source_macro_f1"] = float(np.mean([
        f1_score(g.difficulty_label.map(LABELS.index), g.prediction.map(LABELS.index),
                 labels=[0, 1, 2], average="macro", zero_division=0)
        for _, g in frame.groupby("source_dataset")]))
    for kind in ("quality", "actual_cost"):
        cols = [f"{kind}_{x}" for x in ("low", "medium", "high")]
        if set(cols).issubset(frame):
            values = frame[cols].to_numpy(float)
            if np.isfinite(values).all():
                result[f"selected_{kind}"] = float(values[np.arange(len(frame)), p].mean())
    return result


def score(args):
    root = Path(args.run_dir)
    plan = load_plan(root)
    cfg, pins = plan["config"], plan["pins"]
    if args.role not in ("dev", "gate", "forget", "routerarena") or args.arm not in ARMS:
        raise ValueError("Unsupported scoring role/arm")
    if args.arm == "r1_frozen":
        if args.seed != 0 or args.phase != "baseline":
            raise ValueError("Frozen R1 is measured once, seed 0 / baseline")
        verify_source(pins["models"]["r1"])
        model = Path(pins["models"]["r1"]["path"])
    else:
        if args.seed not in cfg["seeds"]:
            raise ValueError("Undeclared seed")
        if args.phase == "baseline":
            if args.arm != "r0_continued":
                raise ValueError("R1 before conversion is the frozen baseline, not an R0 classifier")
            verify_source(pins["models"]["r0"])
            model = Path(pins["models"]["r0"]["path"])
        else:
            phase = root / "arms" / args.arm / f"seed-{args.seed}" / args.phase
            verify_finished(phase)
            model = phase / "model"
    if args.role != "dev":
        if not args.allow_final_evaluation or not (root / "selection-lock.json").exists():
            raise ValueError("Final evaluation needs a selection lock and explicit flag")
        selection = read(root / "selection-lock.json")
        if selection["plan_sha256"] != sha(root / "plan.json"):
            raise ValueError("Selection lock belongs to another plan")
        if args.phase == "cycle-06":
            expected = selection["final_model_digests"][f"{args.arm}/{args.seed}"]
            if digest(model_inventory(model)) != expected:
                raise ValueError("Final checkpoint changed after selection lock")
        if args.phase not in ("baseline", "cycle-06"):
            raise ValueError("Protected data may only evaluate baseline and fixed final checkpoint")
    path = Path(plan["data_dir"]) / f"{args.role}.parquet"
    file_pin(path, pins["data_sha256"][args.role])
    frame = pd.read_parquet(path)
    file_pin(pins["r1_policy"]["path"], pins["r1_policy"]["sha256"])
    policy = read(pins["r1_policy"]["path"])
    model_digest = digest(model_inventory(model))
    for length in cfg["evaluation_max_lengths"]:
        directory = root / "scores" / args.arm / f"seed-{args.seed}" / args.phase / args.role / str(length)
        if (directory / "receipt.json").exists():
            receipt = read(directory / "receipt.json")
            if (receipt["plan_sha256"] != sha(root / "plan.json") or receipt["model_digest"] != model_digest
                    or receipt["data_sha256"] != pins["data_sha256"][args.role]):
                raise ValueError("Existing prediction provenance differs")
            file_pin(directory / "predictions.parquet", receipt["predictions_sha256"])
            continue
        import torch
        start = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        logits = infer(model, frame, length, cfg["eval_batch_size"])
        inference_seconds = time.monotonic() - start
        peak_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        out = frame.drop(columns=["model_input_text"]).copy()
        out["prediction"] = np.asarray(LABELS)[predictions(logits, frame, args.arm, policy, args.role)]
        for i, name in enumerate(LABELS):
            out[f"logit_{name}"] = logits[:, i]
        directory.mkdir(parents=True, exist_ok=True)
        out.to_parquet(directory / "predictions.parquet", index=False)
        exact_json(directory / "receipt.json", {"arm": args.arm, "seed": args.seed, "phase": args.phase,
            "role": args.role, "max_length": length, "plan_sha256": sha(root / "plan.json"),
            "data_sha256": pins["data_sha256"][args.role], "model_digest": model_digest,
            "inference_seconds": inference_seconds, "peak_cuda_allocated_bytes": peak_bytes,
            "predictions_sha256": sha(directory / "predictions.parquet"), "metrics": metrics(out)})


def dev_score(root, arm, seed, phase, expected_model_digest):
    path = root / "scores" / arm / f"seed-{seed}" / phase / "dev" / "8192"
    receipt = read(path / "receipt.json")
    if receipt["plan_sha256"] != sha(root / "plan.json") or receipt["model_digest"] != expected_model_digest:
        raise ValueError("Development predictions belong to another plan")
    file_pin(path / "predictions.parquet", receipt["predictions_sha256"])
    frame = pd.read_parquet(path / "predictions.parquet")
    return float(np.mean([metrics(group)["macro_f1"] for _, group in frame.groupby("source_dataset")]))


def lock(args):
    root = Path(args.run_dir)
    plan = load_plan(root)
    values = {}
    checkpoints = {}
    for arm in ARMS[:2]:
        values[arm] = []
        for seed in plan["config"]["seeds"]:
            directory = root / "arms" / arm / f"seed-{seed}" / "cycle-06"
            checkpoints[f"{arm}/{seed}"] = digest(verify_finished(directory)["model_files"])
            values[arm].append(dev_score(root, arm, seed, "cycle-06", checkpoints[f"{arm}/{seed}"]))
    verify_source(plan["pins"]["models"]["r1"])
    frozen_digest = digest(model_inventory(Path(plan["pins"]["models"]["r1"]["path"])))
    values["r1_frozen"] = [dev_score(root, "r1_frozen", 0, "baseline", frozen_digest)]
    delta = float(np.mean(values["r1_to_r0"]) - np.mean(values["r0_continued"]))
    selected = "r1_to_r0" if delta > plan["config"]["minimum_dev_improvement"] else "r0_continued"
    exact_json(root / "selection-lock.json", {"plan_sha256": sha(root / "plan.json"),
        "primary_metric": "dev_source_macro_f1_8192", "values": values,
        "selected_adaptive_arm": selected, "delta_B_minus_A": delta,
        "final_model_digests": checkpoints, "production_approved": False})


def compare(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(args.run_dir)
    plan = load_plan(root)
    rows, grouped = [], []
    predictions_by_key = {}
    for path in sorted((root / "scores").rglob("receipt.json")):
        record = read(path)
        if record["plan_sha256"] != sha(root / "plan.json"):
            raise ValueError("Cannot combine results from different plans")
        file_pin(path.parent / "predictions.parquet", record["predictions_sha256"])
        frame = pd.read_parquet(path.parent / "predictions.parquet")
        identity = {k: record[k] for k in ("arm", "seed", "phase", "role", "max_length")}
        predictions_by_key[tuple(identity.values())] = frame
        rows.append({**identity, **metrics(frame)})
        for keys, group in frame.groupby(["source_dataset", "category"]):
            grouped.append({**identity, "source_dataset": keys[0], "category": keys[1], **metrics(group)})
    if not rows:
        raise ValueError("No verified predictions; no invented results")
    out = root / "report"
    out.mkdir(exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(grouped).to_csv(out / "source-category-metrics.csv", index=False)
    dev = results[(results.role == "dev") & (results.max_length == 8192)]
    fig, ax = plt.subplots(figsize=(9, 4))
    phase_order = [p for p in ["baseline", "bridge"] + [f"cycle-{i:02d}" for i in range(1, 7)] if p in set(dev.phase)]
    for arm, group in dev.groupby("arm"):
        summary = group.groupby("phase").source_macro_f1.agg(["mean", "std"])
        ordered = [p for p in phase_order if p in summary.index]
        xs = [phase_order.index(p) for p in ordered]
        ys = summary.loc[ordered, "mean"].to_numpy()
        if arm == "r1_frozen":
            ax.axhline(ys[0], label=arm, linestyle="--")
        else:
            ax.errorbar(xs, ys, yerr=summary.loc[ordered, "std"].fillna(0), label=arm, marker="o")
    ax.set(xticks=range(len(phase_order)), xticklabels=phase_order, ylabel="Dev source-macro F1", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "adaptation-curve.png", dpi=160)
    plt.close(fig)
    from sklearn.metrics import confusion_matrix
    # One confusion matrix per actual run; no majority-vote ensemble invented.
    for key, frame in predictions_by_key.items():
        arm, seed, phase, role, length = key
        if length != 8192 or phase not in ("baseline", "cycle-06"):
            continue
        fig, ax = plt.subplots(figsize=(4, 4))
        matrix = confusion_matrix(frame.difficulty_label, frame.prediction, labels=LABELS)
        ax.imshow(matrix, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        ax.set(xticks=range(3), xticklabels=LABELS, yticks=range(3), yticklabels=LABELS,
               xlabel="Predicted", ylabel="Reference", title=f"{arm} / {role} / seed {seed}")
        fig.tight_layout()
        fig.savefig(out / f"confusion-{arm}-{seed}-{phase}-{role}.png", dpi=140)
        plt.close(fig)
    category = pd.DataFrame(grouped)
    category = category[(category.max_length == 8192) & category.phase.isin(["baseline", "cycle-06"])]
    for role, group in category.groupby("role"):
        group = group.copy()
        group["cell"] = group.source_dataset + " / " + group.category
        group["series"] = group.arm + " / " + group.phase
        pivot = group.pivot_table(index="cell", columns="series", values="macro_f1", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(max(7, len(pivot.columns) * 1.5), max(4, len(pivot) * .3)))
        im = ax.imshow(np.ma.masked_invalid(pivot.to_numpy()), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set(xticks=range(len(pivot.columns)), xticklabels=pivot.columns,
               yticks=range(len(pivot)), yticklabels=pivot.index, title=f"{role}: category macro F1 (seed mean)")
        ax.tick_params(axis="x", rotation=35)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out / f"source-category-{role}.png", dpi=140)
        plt.close(fig)
    contrasts = []
    missing = []
    for role in ("dev", "gate", "forget", "routerarena"):
        keys_a = [("r0_continued", seed, "cycle-06", role, 8192) for seed in plan["config"]["seeds"]]
        keys_b = [("r1_to_r0", seed, "cycle-06", role, 8192) for seed in plan["config"]["seeds"]]
        c = ("r1_frozen", 0, "baseline", role, 8192)
        for key in keys_a + keys_b + [c]:
            if key not in predictions_by_key:
                missing.append(list(key))
        if any(key not in predictions_by_key for key in keys_a + keys_b + [c]):
            continue
        aligned = [predictions_by_key[k].sort_values("sample_id").reset_index(drop=True) for k in keys_a + keys_b + [c]]
        for frame in aligned[1:]:
            cols = ["sample_id", "duplicate_group_id", "source_dataset", "difficulty_label"]
            if not frame[cols].equals(aligned[0][cols]):
                raise ValueError("Paired comparison rows or labels differ")
        # Resample whole duplicate groups, including multiple questions sharing a passage.
        # Stratify the paired group bootstrap by source; variation is conditional
        # on the fixed source checkpoints and the three adaptation seeds.
        rng = np.random.default_rng(20260927)
        strata = [g.index.to_numpy() for _, g in aligned[0].groupby("source_dataset")]
        clusters = [[g.index.to_numpy() for _, g in source.groupby("duplicate_group_id")]
                    for _, source in aligned[0].groupby("source_dataset")]
        from sklearn.metrics import f1_score
        def score_indices(frame, indices):
            return f1_score(frame.difficulty_label.iloc[indices], frame.prediction.iloc[indices],
                            labels=list(LABELS), average="macro", zero_division=0)
        n = len(keys_a)
        def contrast(a, b, sampled):
            av = np.mean([np.mean([score_indices(f, ix) for ix in sampled]) for f in a])
            bv = np.mean([np.mean([score_indices(f, ix) for ix in sampled]) for f in b])
            return float(bv - av)
        for name, left, right in [("B_minus_A", aligned[:n], aligned[n:2*n]),
                                   ("B_minus_C", aligned[-1:], aligned[n:2*n])]:
            samples = [contrast(left, right, [np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))]) for groups in clusters])
                       for _ in range(1000)]
            contrasts.append({"role": role, "contrast": name, "metric": "source_macro_f1",
                "difference": contrast(left, right, strata), "conditional_ci95_low": float(np.quantile(samples, .025)),
                "conditional_ci95_high": float(np.quantile(samples, .975)), "bootstrap_draws": 1000})
    pd.DataFrame(contrasts).to_csv(out / "paired-comparisons.csv", index=False)
    retention = []
    missing_retention = []
    for arm in ARMS[:2]:
        baseline_key = (("r0_continued", plan["config"]["seeds"][0], "baseline", "forget", 8192)
                        if arm == "r0_continued" else ("r1_frozen", 0, "baseline", "forget", 8192))
        if baseline_key not in predictions_by_key:
            missing_retention.append(list(baseline_key))
            continue
        baseline = metrics(predictions_by_key[baseline_key])["source_macro_f1"]
        for seed in plan["config"]["seeds"]:
            key = (arm, seed, "cycle-06", "forget", 8192)
            if key not in predictions_by_key:
                missing_retention.append(list(key))
                continue
            final = metrics(predictions_by_key[key])["source_macro_f1"]
            retention.append({"arm": arm, "seed": seed, "baseline_arm": baseline_key[0],
                "baseline_source_macro_f1": baseline, "final_source_macro_f1": final,
                "drop": baseline - final, "maximum_drop": plan["config"]["maximum_forgetting_drop"],
                "within_research_tolerance": baseline - final <= plan["config"]["maximum_forgetting_drop"]})
    pd.DataFrame(retention).to_csv(out / "retention-check.csv", index=False)
    diagnostics = [list((*key[:-1], 512)) for key in predictions_by_key
                   if 512 in plan["config"]["evaluation_max_lengths"] and key[-1] == 8192 and (*key[:-1], 512) not in predictions_by_key]
    status = {"complete_primary_final_predictions": not missing, "missing_primary_predictions": missing,
              "complete_retention_comparison": not missing_retention,
              "missing_retention_predictions": missing_retention, "missing_512_diagnostics": diagnostics,
              "all_retention_seeds_within_tolerance": (all(x["within_research_tolerance"] for x in retention)
                                                       if retention and not missing_retention else None),
              "production_approved": False, "plan_sha256": sha(root / "plan.json")}
    # Report status is derived and may be regenerated as additional roles arrive.
    (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    phase_final = results[(results.max_length == 8192) & results.phase.isin(["cycle-06", "baseline"])]
    (out / "REPORT.md").write_text("# R1→R0 전이 실험 결과\n\n"
        "생성된 실제 예측만 집계했다. 누락 arm/seed/role은 완료로 해석하지 않는다. "
        "오차막대는 adaptation seed 간 표준편차이며 독립 초기학습 반복의 신뢰구간이 아니다.\n\n"
        "전체 수치는 metrics.csv, 출처·카테고리별 수치는 source-category-metrics.csv에 있다. "
        "selection-lock.json은 개발검증 선택이며 서비스 승격 승인이 아니다.\n\n"
        + f"실제 완료 여부: {not missing}. 누락 목록: status.json.\n\n"
        + "paired-comparisons.csv의 CI는 고정 초기 모델·adaptation seed에 조건부인 질문 그룹 bootstrap이다.\n\n"
        + "retention-check.csv는 A의 원본 R0, B의 원본 R1 정책 대비 망각을 비교한다. "
          "B의 변화에는 출력 의미·정책 전환 효과도 포함된다. 허용폭은 seed별로 확인하며 자동 배포를 승인하지 않는다.\n\n"
        + phase_final.to_csv(index=False), encoding="utf-8")
    print(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "build-feedback-log"):
        p = subs.add_parser(name)
        for flag in ("config", "pins", "data-dir"):
            p.add_argument("--" + flag, required=True)
        if name == "prepare":
            p.add_argument("--feedback-log", required=True)
            p.add_argument("--output-dir", required=True)
        else:
            p.add_argument("--output-file", required=True)
    p = subs.add_parser("train")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--arm", choices=ARMS[:2], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--until-phase", default="cycle-06")
    p = subs.add_parser("score")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--phase", required=True)
    p.add_argument("--role", choices=("dev", "gate", "forget", "routerarena"), required=True)
    p.add_argument("--allow-final-evaluation", action="store_true")
    for name in ("lock", "compare"):
        subs.add_parser(name).add_argument("--run-dir", required=True)
    args = parser.parse_args()
    {"prepare": prepare, "build-feedback-log": build_feedback_log, "train": train,
     "score": score, "lock": lock, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
