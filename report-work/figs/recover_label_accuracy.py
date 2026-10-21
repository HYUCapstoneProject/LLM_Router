"""Recover the existing 4,434-question comparison's per-question predictions.

Uses saved retraining predictions and the original frozen R1/R2/R3 checkpoints.
No training, tuning, or change of evaluation population is performed.
The historical policy and inference recipe are preserved explicitly below.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "report-work/figs/data/label-accuracy"
DRIVE = Path(
    "/Users/yoocheolmin/Library/CloudStorage/"
    "GoogleDrive-alexyoo0506@gmail.com/내 드라이브/LLM_Router"
)
TIERS = ["low", "medium", "high"]
LABELS = {"easy": 0, "medium": 1, "hard": 2}
EXPECTED = {
    "R0": 60.329, "R1": 26.432, "R2": 35.521, "R3": 34.529,
    "F1": 25.342, "F2": 26.214, "F3": 39.866,
}
MODELS = {
    "R1": DRIVE / "transfer-experiments/inputs/experiment-A/models/r1",
    "R2": Path(tempfile.gettempdir()) / "codex-label-accuracy-checkpoints/r2/model",
    "R3": DRIVE / "routing-generalization-v1-20260925/r3-session/r3-natural/model",
}


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    source = DRIVE / (
        "retraining-effect-f3/runs/retraining-effect-v2-f3-run001/"
        "shared-data/stream-windows.parquet"
    )
    audit = ROOT / "ml/artifacts/benchmark-outcome-audit-v1/routerbench-exclusions.parquet"
    w = pd.read_parquet(source)
    w = w[w.window >= 2].copy().reset_index(drop=True)
    excluded = set(pd.read_parquet(audit).sample_id)
    w["excluded"] = (
        w.source_dataset.eq("routerbench")
        & w.sample_id.str.removeprefix("routerbench:").isin(excluded)
    )
    assert len(w) == 4473 and (~w.excluded).sum() == 4434
    assert w.sample_id.is_unique and w.difficulty_label.isin(LABELS).all()
    w.to_parquet(DATA / "stream-population.parquet", index=False)
    source_text = pd.read_parquet(
        ROOT / "ml/artifacts/modernbert-initial-v1/dataset/sealed/stream.parquet",
        columns=["sample_id", "model_input_text"],
    )
    inputs = w[["sample_id"]].merge(source_text, on="sample_id", validate="one_to_one")
    assert len(inputs) == len(w) and inputs.model_input_text.notna().all()
    inputs.to_parquet(DATA / "stream-inputs.parquet", index=False)
    pd.DataFrame(
        w[~w.excluded].groupby(["source_dataset", "category"]).size(),
        columns=["n"],
    ).to_csv(DATA / "source-category-counts.csv")
    manifest = {
        "population": "stream windows 2–6, audited RouterBench exclusions removed",
        "n_before_exclusions": len(w),
        "n": 4434,
        "label": "original difficulty_label; easy/medium/hard mapped to low/medium/high",
        "accuracy": "fraction of selected initial tiers matching the difficulty label",
        "retraining_aggregation": "mean of per-seed accuracy for seeds 42, 43, 44",
        "historical_summary_percent": EXPECTED,
        "historical_execution": {
            "project_session": "83a1b657-e83f-4493-9986-98cf681f8f6d",
            "tool_id": "toolu_01LocfMshRLa1H6NvAn8aTRS",
            "timestamp": "2026-10-01T17:14:24.527Z",
        },
        "sources": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source, audit)
        },
        "inference": {
            "max_length": 8192, "batch_size": 16,
            "dtype": "float32", "attention": "eager",
            "order": "ascending character length via numpy.argsort",
        },
        "policies": {
            "R0": "saved initial_tier",
            "R1": "argmax(sigmoid(logits) - lambda * decision_cost); RB=.10, xRB=.03",
            "R2": "argmax(logits - lambda * decision_cost); RB=1.0, xRB=.10",
            "R3": "argmax(logits)",
        },
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Prepared 4,473 inference rows; 4,434 comparison rows.", flush=True)


def extract_r2():
    MODELS["R2"].mkdir(parents=True, exist_ok=True)
    if (MODELS["R2"] / "model.safetensors").exists():
        return
    source = DRIVE / "routing-objectives-continual-v1-A-20260924/pairwise-final-results.tar"
    with tarfile.open(source) as archive:
        # Only checkpoint files; never execute or extract scripts from the archive.
        members = [
            m for m in archive.getmembers()
            if m.isfile() and Path(m.name).parent.as_posix().strip("./") == "model"
        ]
        assert any(Path(m.name).name == "model.safetensors" for m in members)
        for member in members:
            target = MODELS["R2"] / Path(member.name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".partial")
            with archive.extractfile(member) as src, partial.open("wb") as dst:
                import shutil
                shutil.copyfileobj(src, dst)
            partial.replace(target)
    print("Recovered frozen R2 checkpoint.", flush=True)


def infer(name):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    target = DATA / f"stream-{name.lower()}-scores.parquet"
    if target.exists():
        print(f"Reusing {target.name}", flush=True)
        return
    if name == "R2":
        extract_r2()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODELS[name], local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODELS[name], attn_implementation="eager",
        torch_dtype=torch.float32, local_files_only=True,
    ).to(device).eval()
    frame = pd.read_parquet(DATA / "stream-inputs.parquet")
    texts = frame.model_input_text.astype(str)
    order = np.argsort(texts.str.len().to_numpy())
    scores = np.zeros((len(frame), 3), dtype=np.float32)
    start = time.time()
    with torch.no_grad():
        for i in range(0, len(order), 16):
            idx = order[i:i + 16]
            encoded = tokenizer(
                texts.iloc[idx].tolist(), padding=True, truncation=True,
                max_length=8192, return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            scores[idx] = model(**encoded).logits.float().cpu().numpy()
            if i % 320 == 0:
                print(f"{name}: {i}/{len(frame)} ({time.time() - start:.0f}s)", flush=True)
    result = pd.DataFrame(scores, columns=[f"score_{t}" for t in TIERS])
    result.insert(0, "sample_id", frame.sample_id.to_numpy())
    result.to_parquet(target, index=False)
    print(f"{name}: complete ({time.time() - start:.0f}s)", flush=True)


def aggregate():
    frame = pd.read_parquet(DATA / "stream-population.parquet")
    frame = frame[~frame.excluded].reset_index(drop=True)
    y = frame.difficulty_label.map(LABELS).to_numpy(int)
    decision_cost = frame[[f"decision_cost_{t}" for t in TIERS]].to_numpy(float)
    is_rb = frame.source_dataset.eq("routerbench").to_numpy()
    methods = {"R0": [(0, frame.initial_tier.to_numpy(int))]}
    cached_path = DATA / "per-question-predictions.parquet"
    cached = pd.read_parquet(cached_path) if cached_path.exists() else pd.DataFrame()
    for method, lambdas in (("R1", (.1, .03)), ("R2", (1., .1)), ("R3", (0., 0.))):
        path = DATA / f"stream-{method.lower()}-scores.parquet"
        if not path.exists():
            continue
        scores = pd.read_parquet(path).set_index("sample_id").loc[
            frame.sample_id, [f"score_{t}" for t in TIERS]
        ].to_numpy(float)
        if method == "R1":
            scores = 1 / (1 + np.exp(-np.clip(scores, -30, 30)))
        penalty = np.where(is_rb, *lambdas)[:, None] * decision_cost
        methods[method] = [(0, (scores - penalty).argmax(axis=1))]
    for method, run, arm in (
        ("F1", "retraining-effect/runs/retraining-effect-v1-run001", "f1_cycle_only"),
        ("F2", "retraining-effect/runs/retraining-effect-v1-run001", "f2_replay"),
        ("F3", "retraining-effect-f3/runs/retraining-effect-v2-f3-run001", "f3_balanced_easy"),
    ):
        methods[method] = []
        for seed in (42, 43, 44):
            if len(cached):
                prior = cached[cached.method.eq(method) & cached.seed.eq(seed)]
                if len(prior) == len(frame) and set(prior.sample_id) == set(frame.sample_id):
                    aligned = prior.set_index("sample_id").loc[frame.sample_id]
                    np.testing.assert_array_equal(aligned.label.to_numpy(), y)
                    methods[method].append((seed, aligned.initial_tier.to_numpy(int)))
                    continue
            prediction = pd.concat([
                pd.read_parquet(
                    DRIVE / run / f"scores/{arm}/seed-{seed}/window-0{k}/predictions.parquet",
                    columns=["sample_id", "prediction"],
                ) for k in range(2, 7)
            ])
            assert prediction.sample_id.is_unique
            initial = prediction.set_index("sample_id").loc[
                frame.sample_id, "prediction"
            ].map(LABELS).to_numpy(int)
            methods[method].append((seed, initial))
    rows = []
    for method, predictions in methods.items():
        accuracy = np.mean([(pred == y).mean() * 100 for _, pred in predictions])
        print(f"{method}: {accuracy:.6f}% (historical {EXPECTED[method]:.3f}%)", flush=True)
        assert round(accuracy, 3) == EXPECTED[method], (method, accuracy)
        for seed, pred in predictions:
            subset = frame[["sample_id", "source_dataset", "category", "source_task"]].copy()
            subset["method"] = method
            subset["seed"] = seed
            subset["label"] = y
            subset["initial_tier"] = pred
            subset["correct"] = pred == y
            rows.append(subset)
    pd.concat(rows).to_parquet(DATA / "per-question-predictions.parquet", index=False)
    summary = pd.DataFrame([
        {"method": method, "n": len(frame), "seeds": len(seeds),
         "accuracy": np.mean([(pred == y).mean() * 100 for _, pred in seeds])}
        for method, seeds in methods.items()
    ])
    summary.to_csv(DATA / "overall.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "infer", "aggregate"])
    parser.add_argument("--method", choices=["R1", "R2", "R3"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "infer":
        infer(args.method)
    else:
        aggregate()
