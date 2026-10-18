"""EV-10 07a_score_ours: score the ModernBERT routers on every dev, test and RouterArena question.

03_ours wrote validation scores only for its own validation rows (R1-group-A: the A split only) and 02_rt08
wrote none, so 07_lock could not calibrate them for every source. This step runs each trained model once
over the same questions as 04-06, with the training's own tokenisation (512 tokens, right truncation) and
the shared batch scorer of the objective-comparison code, and writes raw logits in the 04 format:

  r0_group, r1_group, r1_group_a   03_ours models (low/medium/high outputs)
  rt08                             02_rt08 model (easy/medium/hard outputs, old definition)

Test outcomes are never read: questions come from the text columns only (finetune_track.load_inputs).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # ml/src: the objective-comparison scorer
import common_track as C  # noqa: E402
import finetune_track as F  # noqa: E402

MODELS = {  # name -> (model folder under the run, output class order)
    "r0_group": ("ours/models/r0_group/model", ("low", "medium", "high")),
    "r1_group": ("ours/models/r1_group/model", ("low", "medium", "high")),
    "r1_group_a": ("ours/models/r1_group_a/model", ("low", "medium", "high")),
    "rt08": ("rt08/model/full/final-model", ("easy", "medium", "hard")),
}
SEED_RUNS = ("r1_group", "r1_group_a")  # 09_seeds: H5 only (plan 4.3: seeds only near a decision boundary)


def seed_models(seed: int) -> dict:
    return {f"{base}_s{seed}": (f"ours/models/seed{seed}/{base}/model", MODELS[base][1]) for base in SEED_RUNS}


MAX_LENGTH = 512  # routing_objective_continual_v1.json and modernbert_initial_v1_training.json
BATCH = 32


def model_labels(model_dir: Path) -> tuple:
    id2label = json.loads((model_dir / "config.json").read_text())["id2label"]
    return tuple(id2label[str(i)] for i in range(len(id2label)))


def score(name: str, run_dir: Path, inputs: dict, output_dir: Path, device: str, models: dict = MODELS) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from llm_router_labeling.objective_comparison.training import _predict_logits
    folder = output_dir / name
    if (folder / "_SUCCESS").exists():
        print(f"[{name}] done", flush=True)
        return json.loads((folder / "metrics.json").read_text())
    rel, labels = models[name]
    model_dir = run_dir / rel
    if model_labels(model_dir) != labels:
        raise SystemExit(f"{name}: unexpected output classes {model_labels(model_dir)}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    frame = inputs["scored"][["sample_id", "source_dataset", "role"]].assign(model_input_text=inputs["texts"])
    order = F.length_order(inputs["texts"])  # batches of similar length; results are put back in place
    start = time.time()
    logits = np.zeros((len(frame), 3))
    logits[order] = _predict_logits(model, tokenizer, frame.iloc[order], batch_size=BATCH, max_length=MAX_LENGTH,
                                     device=device)
    seconds = time.time() - start
    tokens = [len(ids) for ids in tokenizer(inputs["texts"], truncation=False)["input_ids"]]
    table = frame[["sample_id", "source_dataset", "role"]].copy()
    for k, tier in enumerate(C.TIERS):
        table[f"score_{tier}"] = logits[:, k]
    table["input_token_count"] = tokens
    table["was_truncated"] = np.array(tokens) > MAX_LENGTH
    folder.mkdir(parents=True, exist_ok=True)
    table.to_parquet(folder / "scores.parquet", index=False)
    cpu = model.to("cpu")
    latency = F.cpu_latency(lambda t: _predict_logits(cpu, tokenizer, pd.DataFrame({"model_input_text": t}),
                                                      batch_size=1, max_length=MAX_LENGTH, device="cpu"),
                            inputs["texts"])
    record = {"model": name, "model_dir": rel, "output_classes": list(labels), "max_length": MAX_LENGTH,
              "model_sha256": C.sha256_file(model_dir / "model.safetensors"), "device": device,
              "score_seconds": seconds, "score_rows": int(len(table)), "cpu_ms_per_question": latency,
              "truncated_share": float(table.was_truncated.mean()), "torch": torch.__version__,
              "scores_sha256": C.sha256_file(folder / "scores.parquet")}
    (folder / "metrics.json").write_text(json.dumps(record, indent=2))
    (folder / "_SUCCESS").write_text("")
    print(f"[{name}] {len(table)} rows in {seconds:.0f}s, truncated {record['truncated_share']:.1%}", flush=True)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True, help="normally <run>/scores-modernbert")
    p.add_argument("--models", nargs="+", choices=sorted(MODELS), default=list(MODELS))
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke", action="store_true", help="40 scored rows per role")
    p.add_argument("--seed", type=int, default=None, help="09_seeds: score ours/models/seed<N>/{r1_group,r1_group_a}")
    a = p.parse_args(argv)
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    inputs = F.load_inputs(a.run_dir, a.smoke)
    models = MODELS if a.seed is None else {n: v for n, v in seed_models(a.seed).items() if (a.run_dir / v[0]).exists()}
    if a.seed is not None:
        print(f"[seed {a.seed}] models trained so far: {sorted(models)}", flush=True)
    for m in (a.models if a.seed is None else list(models)):
        score(m, a.run_dir, inputs, a.output_dir, device, models)
    out = {m: json.loads((a.output_dir / m / "metrics.json").read_text())["scores_sha256"]
           for m in models if (a.output_dir / m / "_SUCCESS").exists()}
    manifest = {"experiment": "EV-10 07a_score_ours" if a.seed is None else f"EV-10 09_seeds scores (seed {a.seed})",
                "seed": a.seed, "smoke": a.smoke, "inputs": inputs["hashes"], "outputs": out,
                "scored_rows": inputs["scored"].groupby("role").size().to_dict()}
    (a.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
