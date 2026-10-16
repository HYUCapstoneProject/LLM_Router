"""EV-10 05_finetune_track and 06_released: routers that fine-tune their own encoder, and RouteLLM's released BERT.

Same training rows as 03 and 04: the 03_ours blended train split (text and the three group success
probabilities), with the three model-group tiers as three virtual models. Each setting writes per-tier
scores for every dev, test and RouterArena question to <output>/<method>/<setting>/ in the 04 format;
07_lock calibrates them per source on dev and picks one setting per method.

  routerdc            (05a) LLMRouter's RouterDC: RouterModule, DCDataset, add_clusters and DCTrainer,
                            mDeBERTa-v3-base. Two configurations (LLMRouter YAML, RouterDC RouterBench script).
  routellm_retrained  (05b) RouteLLM's BERT router architecture retrained on our data with the training
                            arguments stored in the released checkpoint.
  routellm_released   (06)  RouteLLM's released checkpoint routellm/bert_gpt4_augmented, no training.

External code is fetched at pinned commits (SHA-256 per file) and not vendored. Test outcomes are never
read: test questions come from the text column of 00_prepare's test.parquet only.
"""
from __future__ import annotations

import argparse
import copy
import functools
import importlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_track as C  # noqa: E402

TIERS = C.TIERS
SEED = C.SEED

EXTERNAL = {
    "llmrouter": {
        "repo": "ulab-uiuc/LLMRouter", "commit": "c65ee0122bc9e84f71d571fcdec4a948481b6e88", "license": "MIT",
        "files": {
            "LICENSE": "905c5b3e88f78dc96ed4d0b8ec1c0a20fbb3122988b7d992be64d751836ecc92",
            "llmrouter/models/base_trainer.py": "8bcde8ea1847646e9b7d60b03321b94d5b05363c1f0c52f63c38a771ea1c8af4",
            "llmrouter/models/routerdc/__init__.py": "06860e1bd4bea8a4549427d8975723103b9fe5777eeb474c7949f237157289e7",
            "llmrouter/models/routerdc/router.py": "00ed8f09d546df13d2cb768d1283ed12d1791dfbd7014a3d60b575e764a6da8a",
            "llmrouter/models/routerdc/trainer.py": "7d98f6e99a7e9a3e9b9eba16bb155ca07f9f9e9008de299df76bf8c2d3b45299",
            "llmrouter/models/routerdc/dcmodel.py": "7c51cd3f9c683f659127707e9112802e5e7f824b3846953ffe6b7fa541a7fcf4",
            "llmrouter/models/routerdc/dcdataset.py": "0cbbad3d3fa2b5ecfe8217df6eb505cc0b71bc35978e96afb43a186c0ef036f7",
            "llmrouter/models/routerdc/dcutils.py": "55524ec344815d983190654a6086c7c6294c30f4c7d8736e2586faf18cd7882d",
            "llmrouter/models/routerdc/dcdata_utils.py": "88d7dcab17122a8d284bc4d7aed644a4608087e7befa11d32b7ff31b26806a28",
        },
    },
    "routellm": {
        "repo": "lm-sys/RouteLLM", "commit": "0b64fdafe049e596a3f5657c219329f24af24198", "license": "Apache-2.0",
        "files": {
            "LICENSE": "43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1",
            "routellm/routers/routers.py": "2393da77f9049970f57dd6d8eada3cc68e99506b2a547e7662e7d2d56b5065b8",
        },
    },
}
HF_MODELS = {  # Hugging Face repository, pinned revision
    "mdeberta": ("microsoft/mdeberta-v3-base", "a0484667b22365f84929a935b5e50a51f71f159d"),
    "xlmr": ("FacebookAI/xlm-roberta-base", "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"),
    "released": ("routellm/bert_gpt4_augmented", "86237e3df400762178ea98379477b8296e66d5e4"),
}
# RouterDC hyperparameters. llmrouter: configs/model_config_train/dcrouter.yaml at the pinned commit.
# official_routerbench: train_scripts/router_train_routerbench.sh of github.com/shuhao02/RouterDC
# (commit 168c026d) -- top_k/last_k 4 there (11 models) are capped at our three tiers.
ROUTERDC_CONFIGS = {
    "llmrouter": {"training_steps": 500, "batch_size": 32, "eval_steps": 50},
    "official_routerbench": {"training_steps": 1000, "batch_size": 64, "eval_steps": 100},
}
ROUTERDC_COMMON = {"learning_rate": 5e-5, "top_k": 3, "last_k": 3, "temperature": 1.0, "sample_loss_weight": 0.0,
                   "cluster_loss_weight": 1.0, "H": 3, "gradient_accumulation": 1, "n_clusters": 3,
                   "hidden_state_dim": 768, "similarity_function": "cos", "max_len": 512}
# RouteLLM retraining: values read from the released checkpoint's training_args.bin and trainer_state.json
# (inspected with pickletools, not unpickled) and the paper's "~2000 steps".
ROUTELLM_ARGS = {"learning_rate": 1e-5, "weight_decay": 0.01, "per_device_batch": 16, "accumulation": 2,
                 "schedule_epochs": 4, "stop_at_step": 2000, "eval_steps": 500, "max_len": 512}

SETTINGS = {
    "routerdc": [{"config": c, "checkpoint": k} for c in ROUTERDC_CONFIGS for k in ("best_dev", "final")],
    "routellm_retrained": [{"checkpoint": "best_dev_loss"}],
    "routellm_released": [{}],
}
STEP_OF = {"routerdc": "05a", "routellm_retrained": "05b", "routellm_released": "06"}

DEVIATIONS = {
    "all": [
        "Training rows: 03_ours blended train (same rows and targets as R1-group and the 04 routers); input is the "
        "question text (model_input_text); no task names.",
        "Each method outputs raw scores; the routing decision is the shared rule of 07_lock/08_test.",
        "Test questions' text is read from 00_prepare test.parquet without its outcome columns.",
    ],
    "routerdc": [
        "LLMRouter RouterDC (RouterModule, DCDataset, add_clusters, DCTrainer) with mDeBERTa-v3-base; the three tiers "
        "are the three candidate LLMs with score = group success probability; clusters = k-means(3) on those score "
        "vectors (dcdata_utils.add_clusters).",
        "config=llmrouter is LLMRouter's dcrouter.yaml (500 steps, batch 32, lr 5e-5, top_k 3, last_k 3, cluster loss "
        "1). config=official_routerbench is the RouterDC repository's RouterBench script (1000 steps, batch 64, eval "
        "every 100; its top_k/last_k 4 for 11 models capped at our 3 tiers).",
        "Our dev split stands in for the 'test' split DCTrainer uses to keep best_model.pth (plan 4.3). Its per-step "
        "evaluation of the whole training set (only for best_training_model.pth, which inference ignores when "
        "best_model.pth exists) is skipped.",
        "checkpoint=best_dev is the checkpoint LLMRouter's route_batch loads; checkpoint=final is the last step.",
        "Gradient checkpointing and per-batch removal of all-padding columns: memory and speed only, same outputs.",
        "Seed 42 (the plan's seed) instead of the configs' 1 and 7. Left truncation at 512 tokens as in the original.",
    ],
    "routellm_retrained": [
        "XLM-RoBERTa-base (the released checkpoint's base) fine-tuned with the released run's arguments: lr 1e-5, "
        "weight decay .01, AdamW, linear schedule over 4 epochs, effective batch 32 (16 x 2 GPUs there, 16 x 2 "
        "accumulation here), evaluation and save every 500 steps, best by evaluation loss, fp32, seed 42, stopped at "
        "2000 steps (paper: about 2000 steps; the released run stopped at 1500).",
        "Labels are the original three classes (strong wins / tie / weak wins) with strong = high group and weak = "
        "low group, as soft targets: strong = p_high(1 - p_low), weak = p_low(1 - p_high), tie = the rest; soft "
        "cross-entropy (plan 4.3).",
        "Model selection on our dev split (the paper uses a held-out validation set).",
        "Score = strong win rate as BERTRouter.calculate_strong_win_rate (1 - P(tie) - P(weak wins)), batched; "
        "checked against the original function on sample questions. Single score: P(the high group is needed).",
        "Gradient checkpointing, dynamic padding and model-only checkpoints (no optimizer state; training is not "
        "resumed): memory, disk and speed only.",
    ],
    "routellm_released": [
        "routellm/bert_gpt4_augmented at a pinned revision, no training on our data.",
        "Score = strong win rate as BERTRouter.calculate_strong_win_rate, batched; checked against the original "
        "function on sample questions. Single score: P(the strong model is needed).",
    ],
}


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------

def load_inputs(run_dir: Path, smoke: bool) -> dict:
    """Training rows (03_ours) and the text of every scored question, in the 01_embed id order."""
    embed_dir, data_dir, prepare_dir = run_dir / "embed", run_dir / "ours" / "data", run_dir / "prepare"
    embed_manifest = json.loads((embed_dir / "manifest.json").read_text())
    if C.sha256_file(embed_dir / "ids.parquet") != embed_manifest["outputs"]["ids.parquet"]:
        raise SystemExit("01_embed ids changed")
    if C.sha256_file(prepare_dir / "manifest.json") != embed_manifest["prepare_manifest_sha256"]:
        raise SystemExit("00_prepare manifest differs from the one 01_embed used")
    prepare_manifest = json.loads((prepare_dir / "manifest.json").read_text())
    for name in ("dev.parquet", "test.parquet"):
        if C.sha256_file(prepare_dir / name) != prepare_manifest["outputs"][name]["sha256"]:
            raise SystemExit(f"00_prepare output changed: {name}")
    arena_path = run_dir / "inputs" / "routerarena-full.parquet"
    if C.sha256_file(arena_path) != embed_manifest["routerarena_sha256"]:
        raise SystemExit("RouterArena input changed")
    data_manifest = json.loads((data_dir / "manifest.json").read_text())
    for name, digest in data_manifest["variants"]["blended"]["files"].items():
        if C.sha256_file(data_dir / "blended" / name) != digest:
            raise SystemExit(f"03_ours data changed: blended/{name}")

    columns = ["sample_id", "source_dataset", "model_input_text"] + [f"success_{t}" for t in TIERS]
    train = pd.read_parquet(data_dir / "blended" / "train.parquet", columns=columns)
    dev = pd.read_parquet(data_dir / "blended" / "validation.parquet", columns=columns)
    ids = pd.read_parquet(embed_dir / "ids.parquet")
    scored = ids[ids.role.isin(C.SCORED_ROLES)].reset_index(drop=True)
    text = {}
    for role in ("dev", "test"):  # text column only: outcome columns are not read
        part = pd.read_parquet(prepare_dir / f"{role}.parquet", columns=["sample_id", "model_input_text"])
        text.update(zip(part.sample_id, part.model_input_text.astype(str)))
    arena = pd.read_parquet(arena_path, columns=["Global Index", "Question"])
    text.update(zip("routerarena:" + arena["Global Index"].astype(str), arena["Question"].astype(str)))
    if smoke:
        train = train.groupby("source_dataset", group_keys=False).head(32).reset_index(drop=True)
        dev = dev.groupby("source_dataset", group_keys=False).head(16).reset_index(drop=True)
        scored = scored.groupby("role", group_keys=False).head(40).reset_index(drop=True)
    missing = [s for s in scored.sample_id if s not in text]
    if missing:
        raise SystemExit(f"{len(missing)} scored questions have no text, e.g. {missing[:3]}")
    return {"train": train, "dev": dev, "scored": scored, "texts": [text[s] for s in scored.sample_id],
            "smoke": smoke,
            "hashes": {"embed_manifest": C.sha256_file(embed_dir / "manifest.json"),
                       "ours_data_manifest": C.sha256_file(data_dir / "manifest.json"),
                       "prepare_manifest": C.sha256_file(prepare_dir / "manifest.json")}}


def soft_three_class(p_low: np.ndarray, p_high: np.ndarray) -> np.ndarray:
    """RouteLLM classes [strong wins, tie, weak wins] from the group probabilities (strong = high, weak = low)."""
    strong = p_high * (1 - p_low)
    weak = p_low * (1 - p_high)
    return np.column_stack([strong, 1 - strong - weak, weak])


def trim_padding(batch):
    """Drop columns that are padding in every row of a right-padded batch (outputs unchanged, faster)."""
    for container in (batch[0] if isinstance(batch, (list, tuple)) else batch,):
        width = int(container["attention_mask"].sum(dim=1).max())
        for key in ("input_ids", "attention_mask"):  # the only inputs RouterDC passes to the encoder
            container[key] = container[key][:, :width]
    return batch


def trimmed_collate(items):
    from torch.utils.data import default_collate
    return trim_padding(default_collate(items))


def length_order(texts: list) -> np.ndarray:
    return np.argsort([len(t) for t in texts], kind="stable")


def cpu_latency(fn, texts: list, n: int = 50) -> float:
    """Median milliseconds per question on CPU, one question at a time."""
    times = []
    for t in texts[:n]:
        start = time.perf_counter()
        fn([t])
        times.append(1000 * (time.perf_counter() - start))
    return float(np.median(times))


class _Scored:  # what common_track.write_setting needs
    def __init__(self, scored, device):
        self.scored, self.device = scored, device


def write(output_dir: Path, inputs: dict, method: str, setting: dict, scores: np.ndarray, extra: dict, info: dict,
          device: str) -> dict:
    folder = output_dir / method / C.setting_id(method, setting)
    record = C.write_setting(folder, _Scored(inputs["scored"], device), method, setting, scores, extra,
                             info, deviations=DEVIATIONS["all"] + DEVIATIONS[method], external=EXTERNAL)
    print(f"[{method}] {folder.name}: fit {info['fit_seconds']:.0f}s, score {info['score_seconds']:.0f}s", flush=True)
    return record


# ---------------------------------------------------------------------------------------------
# RouteLLM: the original strong-win-rate function, loaded from the pinned routers.py
# ---------------------------------------------------------------------------------------------

def load_routellm_bert_router(external_dir: Path):
    root = external_dir / "routellm" / "routellm"
    unused = C._unused
    stubs = {
        "datasets": C._module("datasets", concatenate_datasets=unused, load_dataset=unused),
        "routellm": C._module("routellm", root),
        "routellm.routers": C._module("routellm.routers", root / "routers"),
        "routellm.routers.causal_llm": C._module("routellm.routers.causal_llm"),
        "routellm.routers.causal_llm.configs": C._module("routellm.routers.causal_llm.configs", RouterModelConfig=object),
        "routellm.routers.causal_llm.llm_utils": C._module("routellm.routers.causal_llm.llm_utils",
                                                           load_prompt_format=unused, to_openai_api_messages=unused),
        "routellm.routers.causal_llm.model": C._module("routellm.routers.causal_llm.model", CausalLLMClassifier=object),
        "routellm.routers.matrix_factorization": C._module("routellm.routers.matrix_factorization"),
        "routellm.routers.matrix_factorization.model": C._module("routellm.routers.matrix_factorization.model",
                                                                 MODEL_IDS={}, MFModel=object),
        "routellm.routers.similarity_weighted": C._module("routellm.routers.similarity_weighted"),
        "routellm.routers.similarity_weighted.utils": C._module(
            "routellm.routers.similarity_weighted.utils", OPENAI_CLIENT=None, compute_elo_mle_with_tie=unused,
            compute_tiers=unused, preprocess_battles=unused),
    }
    with C.temporary_modules(stubs):
        module = C._load_file("ev10_routellm_routers", root / "routers" / "routers.py")
    return module.BERTRouter


def strong_win_rates(model, tokenizer, texts: list, device: str, batch: int = 32) -> np.ndarray:
    """Batched BERTRouter.calculate_strong_win_rate: class probabilities [strong, tie, weak] per question."""
    import torch
    model.eval()
    order = length_order(texts)
    out = np.zeros((len(texts), 3))
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            idx = order[i:i + batch]
            enc = tokenizer([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True).to(device)
            out[idx] = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()
    return out


def check_against_original(bert_router_cls, checkpoint_dir: Path, texts: list) -> float:
    """Max |batched strong win rate - original calculate_strong_win_rate| on sample questions (CPU)."""
    original = bert_router_cls(checkpoint_path=str(checkpoint_dir))
    ours = strong_win_rates(original.model, original.tokenizer, texts, "cpu", batch=len(texts))[:, 0]
    reference = np.array([original.calculate_strong_win_rate(t) for t in texts])
    return float(np.abs(ours - reference).max())


def routellm_scores(probs: np.ndarray):
    strong = probs[:, 0]  # = 1 - P(tie) - P(weak wins), the router's strong win rate
    return np.repeat(strong[:, None], 3, axis=1), {"p_strong_win": probs[:, 0], "p_tie": probs[:, 1],
                                                   "p_weak_win": probs[:, 2]}


def run_routellm_released(inputs: dict, external_dir: Path, output_dir: Path, device: str) -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    repo, revision = HF_MODELS["released"]
    local = Path(snapshot_download(repo, revision=revision, allow_patterns=[
        "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model",
        "special_tokens_map.json", "LICENSE"]))
    tokenizer = AutoTokenizer.from_pretrained(local)
    model = AutoModelForSequenceClassification.from_pretrained(local, num_labels=3).to(device)
    start = time.time()
    probs = strong_win_rates(model, tokenizer, inputs["texts"], device)
    score_seconds = time.time() - start
    diff = check_against_original(load_routellm_bert_router(external_dir), local, inputs["texts"][:8])
    if diff > 1e-4:
        raise SystemExit(f"batched strong win rate differs from the original function by {diff}")
    cpu_model = copy.deepcopy(model).to("cpu")
    latency = cpu_latency(lambda t: strong_win_rates(cpu_model, tokenizer, t, "cpu"), inputs["texts"])
    scores, extra = routellm_scores(probs)
    write(output_dir, inputs, "routellm_released", {}, scores, extra, {
        "fit_seconds": 0.0, "score_seconds": score_seconds, "train_rows": 0, "single_score": "P(strong needed)",
        "hf_model": {"repo": repo, "revision": revision}, "max_abs_diff_vs_original": diff,
        "cpu_ms_per_question": latency, "external_key": "routellm",
        "parameters": sum(p.numel() for p in model.parameters())}, device)


def run_routellm_retrained(inputs: dict, external_dir: Path, output_dir: Path, device: str, work: Path) -> None:
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainerCallback,
                              TrainingArguments)
    repo, revision = HF_MODELS["xlmr"]
    args_ = ROUTELLM_ARGS
    C.seed_everything()
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(repo, revision=revision, num_labels=3)

    def encode(frame):
        ids = tokenizer(frame.model_input_text.astype(str).tolist(), truncation=True, max_length=args_["max_len"])["input_ids"]
        target = soft_three_class(frame.success_low.to_numpy(float), frame.success_high.to_numpy(float))
        return [{"input_ids": i, "soft_labels": t.tolist()} for i, t in zip(ids, target)]

    def collate(features):
        batch = tokenizer.pad([{"input_ids": f["input_ids"]} for f in features], return_tensors="pt")
        batch["soft_labels"] = torch.tensor([f["soft_labels"] for f in features], dtype=torch.float32)
        return batch

    class SoftLabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs)
            target = inputs.pop("soft_labels")
            outputs = model(**inputs)
            loss = -(target * torch.log_softmax(outputs.logits.float(), dim=-1)).sum(dim=-1).mean()
            return (loss, outputs) if return_outputs else loss

    stop_at = 2 if inputs["smoke"] else args_["stop_at_step"]
    every = 1 if inputs["smoke"] else args_["eval_steps"]

    class StopAt(TrainerCallback):  # the released run stopped early inside a 4-epoch schedule
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= stop_at:
                control.should_training_stop = True
            return control

    training_args = TrainingArguments(
        output_dir=str(work / "routellm-retrained"), per_device_train_batch_size=args_["per_device_batch"],
        gradient_accumulation_steps=args_["accumulation"], per_device_eval_batch_size=32,
        learning_rate=args_["learning_rate"], weight_decay=args_["weight_decay"], adam_beta1=0.9, adam_beta2=0.999,
        adam_epsilon=1e-8, max_grad_norm=1.0, num_train_epochs=args_["schedule_epochs"], lr_scheduler_type="linear",
        warmup_steps=0, eval_strategy="steps", eval_steps=every, save_strategy="steps", save_steps=every,
        save_total_limit=2, save_only_model=True, load_best_model_at_end=True, metric_for_best_model="loss", greater_is_better=False,
        logging_steps=every, seed=SEED, fp16=False, bf16=False, report_to=[], remove_unused_columns=False,
        label_names=["soft_labels"], prediction_loss_only=True, gradient_checkpointing=True,
        use_cpu=(device == "cpu"))
    trainer = SoftLabelTrainer(model=model, args=training_args, train_dataset=encode(inputs["train"]),
                               eval_dataset=encode(inputs["dev"]), data_collator=collate, callbacks=[StopAt()])
    start = time.time()
    trainer.train()
    fit = time.time() - start
    final_dir = work / "routellm-retrained-best"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    model = trainer.model.to(device)
    start = time.time()
    probs = strong_win_rates(model, tokenizer, inputs["texts"], device)
    score_seconds = time.time() - start
    diff = check_against_original(load_routellm_bert_router(external_dir), final_dir, inputs["texts"][:8])
    if diff > 1e-4:
        raise SystemExit(f"batched strong win rate differs from the original function by {diff}")
    cpu_model = copy.deepcopy(model).to("cpu")
    latency = cpu_latency(lambda t: strong_win_rates(cpu_model, tokenizer, t, "cpu"), inputs["texts"])
    log = [e for e in trainer.state.log_history if "eval_loss" in e]
    scores, extra = routellm_scores(probs)
    write(output_dir, inputs, "routellm_retrained", {"checkpoint": "best_dev_loss"}, scores, extra, {
        "fit_seconds": fit, "score_seconds": score_seconds, "train_rows": len(inputs["train"]),
        "single_score": "P(high group needed)", "hf_model": {"repo": repo, "revision": revision},
        "training_args": args_, "global_step": trainer.state.global_step,
        "best_checkpoint": trainer.state.best_model_checkpoint, "eval_history": log,
        "max_abs_diff_vs_original": diff, "cpu_ms_per_question": latency, "external_key": "routellm",
        "parameters": sum(p.numel() for p in model.parameters())}, device)


# ---------------------------------------------------------------------------------------------
# RouterDC
# ---------------------------------------------------------------------------------------------

def dc_records(frame: pd.DataFrame) -> list:
    return [{"question": str(q), "scores": {t: float(v) for t, v in zip(TIERS, row)}}
            for q, row in zip(frame.model_input_text, frame[[f"success_{t}" for t in TIERS]].to_numpy(float))]


def dc_scores(model, tokenizer, texts: list, device: str, batch: int = 64) -> np.ndarray:
    import torch
    model.eval()
    order = length_order(texts)
    out = np.zeros((len(texts), 3))
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            idx = order[i:i + batch]
            enc = tokenizer([texts[j] for j in idx], max_length=ROUTERDC_COMMON["max_len"], padding=True,
                            truncation=True, return_tensors="pt")
            scores, _ = model(t=ROUTERDC_COMMON["temperature"], input_ids=enc["input_ids"].to(device),
                              attention_mask=enc["attention_mask"].to(device))
            out[idx] = scores.float().cpu().numpy()
    return out


def run_routerdc(inputs: dict, external_dir: Path, output_dir: Path, settings: list, device: str, work: Path) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import DebertaV2Model, DebertaV2Tokenizer
    meta_router = C.register_llmrouter(external_dir)
    dc = importlib.import_module("llmrouter.models.routerdc")
    dcmodel = importlib.import_module("llmrouter.models.routerdc.dcmodel")
    dcdataset = importlib.import_module("llmrouter.models.routerdc.dcdataset")
    dcdata = importlib.import_module("llmrouter.models.routerdc.dcdata_utils")
    trainer_module = importlib.import_module("llmrouter.models.routerdc.trainer")
    trainer_module.DataLoader = functools.partial(DataLoader, collate_fn=trimmed_collate)  # speed only
    repo, revision = HF_MODELS["mdeberta"]
    for config in dict.fromkeys(s["config"] for s in settings):
        hp = ROUTERDC_COMMON | ROUTERDC_CONFIGS[config]
        if inputs["smoke"]:
            # batch 16: the cluster loss needs H=3 other-cluster samples in a batch
            hp = hp | {"training_steps": 2, "batch_size": 16, "eval_steps": 1}
        folder = work / f"routerdc-{config}"
        folder.mkdir(parents=True, exist_ok=True)
        C.seed_everything()
        tokenizer = DebertaV2Tokenizer.from_pretrained(repo, revision=revision, truncation_side="left", padding=True)
        encoder = DebertaV2Model.from_pretrained(repo, revision=revision)
        encoder.gradient_checkpointing_enable()
        train_data = dcdata.add_clusters(dc_records(inputs["train"]), n_clusters=hp["n_clusters"])
        train_set = dcdataset.DCDataset(data=train_data, source_max_token_len=hp["max_len"],
                                        target_max_token_len=hp["max_len"], dataset_id=0)
        dev_set = dcdataset.DCDataset(data=dc_records(inputs["dev"]), source_max_token_len=hp["max_len"],
                                      target_max_token_len=hp["max_len"], dataset_id=1)
        train_set.register_tokenizer(tokenizer)
        dev_set.register_tokenizer(tokenizer)
        if train_set.router_node != list(TIERS):
            raise SystemExit("RouterDC nodes must be low, medium, high")
        model = dcmodel.RouterModule(backbone=encoder, hidden_state_dim=hp["hidden_state_dim"], node_size=3,
                                     similarity_function=hp["similarity_function"])
        meta_router.context = {}
        router = dc.DCRouter.__new__(dc.DCRouter)  # DCRouter.__init__ reads YAML and files; set what it would set
        meta_router.__init__(router, model=model)
        router.cfg = {"hparam": hp | {"device": device},
                      "model_path": {"save_model_path": str(folder / "dcrouter_model.pth"), "backbone_model": repo}}
        router.project_root, router.tokenizer = str(folder), tokenizer
        router.train_dataset, router.test_dataset = train_set, dev_set
        trainer = dc.DCTrainer(router, device=device)
        trainer.eval_datasets = [(dev_set, "probability", "test_data")]
        start = time.time()
        trainer.train()
        fit = time.time() - start
        best = folder / "best_model.pth"  # route_batch's choice; without it (no evaluation ran) the final weights
        weights = {"final": folder / "dcrouter_model.pth",
                   "best_dev": best if best.exists() else folder / "dcrouter_model.pth"}
        for setting in [s for s in settings if s["config"] == config]:
            router.model.load_state_dict(torch.load(weights[setting["checkpoint"]], map_location=device))
            router.model.to(device)
            start = time.time()
            scores = dc_scores(router.model, tokenizer, inputs["texts"], device)
            score_seconds = time.time() - start
            cpu_model = copy.deepcopy(router.model).to("cpu")
            latency = cpu_latency(lambda t: dc_scores(cpu_model, tokenizer, t, "cpu"), inputs["texts"])
            write(output_dir, inputs, "routerdc", setting, scores, {}, {
                "fit_seconds": fit, "score_seconds": score_seconds, "train_rows": len(inputs["train"]),
                "hyperparameters": hp, "hf_model": {"repo": repo, "revision": revision},
                "training_log": trainer.training_log, "cpu_ms_per_question": latency, "external_key": "llmrouter",
                "shared_training_per_config": True,
                "parameters": sum(p.numel() for p in router.model.parameters())}, device)


# ---------------------------------------------------------------------------------------------

def run(method: str, run_dir: Path, external_dir: Path, output_dir: Path, device: str, smoke: bool, work: Path) -> dict:
    import torch
    import transformers
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    external = C.verify_external(external_dir, EXTERNAL)
    inputs = load_inputs(run_dir, smoke)
    output_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in SETTINGS[method] if not (output_dir / method / C.setting_id(method, s) / "_SUCCESS").exists()]
    print(f"[{method}] {len(SETTINGS[method]) - len(todo)} done, {len(todo)} to run", flush=True)
    if todo:
        if method == "routerdc":
            run_routerdc(inputs, external_dir, output_dir, todo, device, work)
        elif method == "routellm_retrained":
            run_routellm_retrained(inputs, external_dir, output_dir, device, work)
        else:
            run_routellm_released(inputs, external_dir, output_dir, device)
    outputs = {}
    for setting in SETTINGS[method]:
        folder = output_dir / method / C.setting_id(method, setting)
        if (folder / "_SUCCESS").exists():
            outputs[folder.name] = json.loads((folder / "metrics.json").read_text())["scores_sha256"]
    manifest = {"experiment": f"EV-10 {STEP_OF[method]} {method}", "smoke": smoke, "seed": SEED, "device": device,
                "inputs": inputs["hashes"], "external": external, "hf_models": HF_MODELS,
                "settings": [C.setting_id(method, s) for s in SETTINGS[method]], "deviations": DEVIATIONS,
                "train_rows": inputs["train"].groupby("source_dataset").size().to_dict(),
                "scored_rows": inputs["scored"].groupby(["role", "source_dataset"]).size().unstack(fill_value=0).to_dict("index"),
                "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                             "transformers": transformers.__version__, "numpy": np.__version__},
                "outputs": outputs, "complete": len(outputs) == len(SETTINGS[method])}
    (output_dir / method / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return {k: manifest[k] for k in ("complete", "outputs")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--external-dir", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--method", choices=sorted(SETTINGS), required=True)
    r.add_argument("--run-dir", type=Path, required=True)
    r.add_argument("--external-dir", type=Path, required=True)
    r.add_argument("--output-dir", type=Path, required=True)
    r.add_argument("--work-dir", type=Path, default=Path("/tmp/ev10-finetune-work"), help="checkpoints (local disk)")
    r.add_argument("--device", default="auto")
    r.add_argument("--smoke", action="store_true", help="32 training rows per source, 2 steps, 40 scored rows per role")
    a = p.parse_args(argv)
    if a.command == "fetch":
        out = C.fetch(a.external_dir, EXTERNAL)
    else:
        out = run(a.method, a.run_dir, a.external_dir, a.output_dir, a.device, a.smoke, a.work_dir)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
